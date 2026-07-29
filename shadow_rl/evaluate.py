#!/usr/bin/env python3
"""Evaluate one model role on the seven Search-R1 QA sets and append to results.csv.

Generation reproduces Search-R1's ``LLMGenerationManager.run_llm_loop``: up to
``--max-turns`` search rounds, each response truncated at the first ``</search>``
or ``</answer>``, retrieval results wrapped in ``<information>`` blocks, followed
by one final answer-only rollout.

Scoring is *not* reimplemented.  ``verl/utils/reward_score/qa_em.py`` is loaded
straight from a Search-R1 checkout and used as-is, because answer parsing has a
subtlety that is easy to get wrong: ``extract_solution`` ignores sequences with
fewer than two ``<answer>`` matches, since the prompt itself contains one
``<answer> Beijing </answer>`` example.  The scored string is therefore the full
prompt + rollout, exactly as verl's reward manager decodes it.

Example
-------
    python shadow_rl/evaluate.py \
        --search-r1-root ~/Search-R1 \
        --pair ppo-nosearch-3b-v0.2 \
        --role rl_on_base \
        --out shadow_rl/results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs import DATASETS, MODEL_ROLES, PAIRS_BY_ID, Pair  # noqa: E402

# The seven columns the experiment specifies, plus n_questions -- without it a
# sampled run and a full run are indistinguishable in the CSV, and the two are
# not comparable.
CSV_FIELDS = ["version", "size", "algo", "with_search", "model_role", "dataset",
              "em", "n_questions"]

# Fixed so that every model role sees exactly the same subsample. Comparing roles
# across different question sets would be meaningless.
SAMPLE_SEED = 20240917

# The Search-R1 agentic prompt, byte-identical to make_prefix() in
# scripts/data_process/qa_search_test_merge.py.  Do not reflow: the checkpoints
# were trained against this exact string.
SEARCH_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# The R1 baseline: same reasoning/answer contract, no search tool.  Used for the
# `R1-*` checkpoints, which were trained with retrieval disabled.
NOSEARCH_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first. \
After reasoning, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# qa_em.extract_solution returns the *last* <answer>...</answer> match and gives up
# on sequences with fewer than two.  The prompt itself supplies two matches -- the
# "inside <answer> and </answer>" phrasing and the "<answer> Beijing </answer>"
# example -- so the model's own answer lands last and is the one extracted.  A
# reworded prompt that dropped them would silently score every model at zero, so
# the invariant is asserted at import.
#
# Known quirk of the official scorer, preserved deliberately: a rollout that never
# answers falls back to the prompt's "Beijing", so such a rollout counts as correct
# on the rare question whose gold answer is Beijing.  We reuse the harness as-is
# rather than diverge from the published numbers; the per-dataset "no parseable
# <answer>" count printed below is what to watch instead.
for _name, _p in (("SEARCH_PROMPT", SEARCH_PROMPT), ("NOSEARCH_PROMPT", NOSEARCH_PROMPT)):
    _matches = re.findall(r"<answer>(.*?)</answer>", _p, re.DOTALL)
    assert len(_matches) >= 2 and _matches[-1].strip() == "Beijing", (
        f"{_name} must keep the <answer> example the official scorer relies on"
    )

INVALID_ACTION_OBS = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
    "Let me try again.\n"
)


# --------------------------------------------------------------------------- #
# official scorer
# --------------------------------------------------------------------------- #
def load_qa_em(search_r1_root: str):
    """Import qa_em.py from a Search-R1 checkout rather than vendoring it."""
    path = os.path.join(search_r1_root, "verl", "utils", "reward_score", "qa_em.py")
    if not os.path.exists(path):
        sys.exit(
            f"[fail] {path} not found.\n"
            "Clone the harness first:\n"
            "  git clone https://github.com/PeterGriffinJin/Search-R1.git\n"
            "and pass its path with --search-r1-root."
        )
    spec = importlib.util.spec_from_file_location("qa_em", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_questions(dataset: str, limit: Optional[int], sample: Optional[int]) -> List[Dict]:
    """Load a test split the way scripts/data_process/qa_search_test_merge.py does.

    `sample` takes a deterministic random subset -- the same questions for every
    model role, so the four roles stay comparable. `limit` takes the first N
    instead, which is only for smoke tests: these files are not shuffled, so a
    prefix is a biased sample.
    """
    import datasets as hfds

    ds = hfds.load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset)
    for split in ("test", "dev", "train"):
        if split in ds:
            rows = ds[split]
            print(f"[data] {dataset}: using '{split}' split ({len(rows)} rows)")
            break
    else:
        raise KeyError(f"no usable split for {dataset}")

    indices = range(len(rows))
    if sample and sample < len(rows):
        # Seeded on the dataset name too, so different sets draw different rows
        # while any given set is identical across roles and across runs.
        rng = random.Random(f"{SAMPLE_SEED}:{dataset}")
        indices = sorted(rng.sample(range(len(rows)), sample))
        print(f"[data] {dataset}: sampled {sample} of {len(rows)} (seed {SAMPLE_SEED})")
    elif sample:
        print(f"[data] {dataset}: {len(rows)} rows, smaller than --sample {sample}; using all")

    out = []
    for i in indices:
        row = rows[int(i)]
        question = row["question"].strip()
        if question[-1] != "?":
            question += "?"
        out.append({"question": question, "golden_answers": row["golden_answers"]})
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, url: str, topk: int):
        self.url, self.topk = url, topk

    def check(self) -> None:
        try:
            self.search(["who wrote hamlet?"])
        except Exception as exc:
            sys.exit(
                f"[fail] retrieval server at {self.url} is not answering: {exc}\n"
                "Launch it with shadow_rl/launch_bm25_retriever.sh first."
            )

    def search(self, queries: List[str]) -> List[str]:
        import requests

        if not queries:
            return []
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        results = requests.post(self.url, json=payload, timeout=600).json()["result"]
        return [self._to_string(r) for r in results]

    @staticmethod
    def _to_string(retrieval_result) -> str:
        # Identical formatting to LLMGenerationManager._passages2string.
        out = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            out += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return out


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def truncate_response(text: str) -> str:
    """Cut the response at its first completed action, as Search-R1 does."""
    if "</search>" in text:
        return text.split("</search>")[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    return text


def parse_action(text: str):
    """First <search>/<answer> block decides the action (Search-R1 semantics)."""
    match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
    if not match:
        return None, ""
    return match.group(1), match.group(2).strip()


def run_rollouts(
    llm,
    tokenizer,
    prompts: List[str],
    retriever: Optional[Retriever],
    max_turns: int,
    max_response_length: int,
    max_obs_length: int,
) -> List[str]:
    """Return the full prompt+rollout string for each question."""
    from vllm import SamplingParams

    # Greedy: EM on a fixed checkpoint should not move between runs.
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_response_length,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
    )

    sequences = list(prompts)
    active = list(range(len(prompts)))

    def generate(indices: List[int]) -> List[str]:
        outs = llm.generate([sequences[i] for i in indices], params)
        return [o.outputs[0].text for o in outs]

    for turn in range(max_turns):
        if not active:
            break
        responses = generate(active)

        pending_search, search_slots = [], []
        obs_for: Dict[int, str] = {}
        still_active = []

        for idx, raw in zip(active, responses):
            response = truncate_response(raw)
            sequences[idx] += response
            action, content = parse_action(response)

            if action == "answer":
                continue                       # done
            if action == "search" and retriever is not None:
                pending_search.append(content)
                search_slots.append(idx)
                still_active.append(idx)
            else:
                # Malformed action, or a search request with no retriever
                # available: nudge and let it retry, exactly as the harness does.
                obs_for[idx] = INVALID_ACTION_OBS
                still_active.append(idx)

        if pending_search:
            for idx, result in zip(search_slots, retriever.search(pending_search)):
                obs_for[idx] = f"\n\n<information>{result.strip()}</information>\n\n"

        for idx, obs in obs_for.items():
            ids = tokenizer(obs, add_special_tokens=False)["input_ids"]
            if len(ids) > max_obs_length:
                obs = tokenizer.decode(ids[:max_obs_length])
            sequences[idx] += obs

        active = still_active
        print(f"[rollout] turn {turn + 1}/{max_turns}: {len(active)} still active", flush=True)

    # Final answer-only rollout for trajectories that never committed an answer.
    if active:
        for idx, raw in zip(active, generate(active)):
            sequences[idx] += truncate_response(raw)

    return sequences


# --------------------------------------------------------------------------- #
def existing_rows(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {
            (r["version"], r["size"], r["algo"], r["with_search"], r["model_role"], r["dataset"])
            for r in csv.DictReader(fh)
        }


def append_rows(path: str, rows: List[Dict]) -> None:
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-r1-root", required=True, help="path to a Search-R1 checkout")
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS_BY_ID))
    ap.add_argument("--role", required=True, choices=MODEL_ROLES)
    ap.add_argument("--model-path", default=None,
                    help="override the model location; required for --role shadow")
    ap.add_argument("--out", default="shadow_rl/results.csv")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-datasets", default="",
                    help="comma list of datasets to omit, e.g. popqa,triviaqa")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions per dataset; biased, smoke tests only")
    ap.add_argument("--sample", type=int, default=None,
                    help="deterministic random subsample of N questions per dataset; "
                         "identical across roles, so the four stay comparable")
    ap.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=4, help="max action budget")
    ap.add_argument("--max-response-length", type=int, default=500)
    ap.add_argument("--max-obs-length", type=int, default=500)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dump-generations", default=None, help="optional .jsonl of raw rollouts")
    ap.add_argument("--overwrite", action="store_true", help="re-run rows already in results.csv")
    args = ap.parse_args()

    pair: Pair = PAIRS_BY_ID[args.pair]
    model_path = args.model_path or pair.repo(args.role)
    if model_path is None:
        sys.exit("[fail] --role shadow needs --model-path (produced by shadow_rl/merge.py)")

    qa_em = load_qa_em(args.search_r1_root)

    skip = {d.strip() for d in args.skip_datasets.split(",") if d.strip()}
    unknown = skip - set(DATASETS)
    if unknown:
        sys.exit(f"[fail] --skip-datasets names unknown dataset(s): {', '.join(sorted(unknown))}")
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip() and d.strip() not in skip]
    if skip:
        print(f"[info] skipping {', '.join(sorted(skip))}; "
              f"averages will cover {len(wanted)} of {len(DATASETS)} datasets")
    key = (pair.version, pair.size, pair.algo, str(pair.with_search), args.role)
    done = set() if args.overwrite else existing_rows(args.out)
    todo = [d for d in wanted if (*key, d) not in done]
    if not todo:
        print(f"[skip] {args.pair}/{args.role}: all datasets already in {args.out}")
        return
    if len(todo) < len(wanted):
        print(f"[resume] skipping {sorted(set(wanted) - set(todo))}, already present")

    retriever = None
    if pair.with_search:
        retriever = Retriever(args.retriever_url, args.topk)
        retriever.check()
        print(f"[ok] retrieval server live at {args.retriever_url}")
    else:
        print("[info] no-search pair: retrieval disabled")

    prompt_template = SEARCH_PROMPT if pair.with_search else NOSEARCH_PROMPT

    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"[load] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    dump = open(args.dump_generations, "a") if args.dump_generations else None
    rows = []

    for dataset in todo:
        questions = load_questions(dataset, args.limit, args.sample)
        prompts = []
        for q in questions:
            text = prompt_template.format(question=q["question"])
            # Search-R1 stores prompts as chat messages and lets verl apply the
            # template, so every role -- including the base-side checkpoints and
            # the merged model -- goes through the chat path.
            if tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            prompts.append(text)

        print(f"\n[eval] {args.pair} / {args.role} / {dataset}  ({len(prompts)} questions)")
        sequences = run_rollouts(
            llm, tokenizer, prompts, retriever,
            args.max_turns, args.max_response_length, args.max_obs_length,
        )

        # Scored on prompt + rollout, matching verl's reward manager.
        scores = [
            qa_em.compute_score_em(
                solution_str=seq,
                ground_truth={"target": q["golden_answers"]},
            )
            for seq, q in zip(sequences, questions)
        ]
        em = sum(scores) / len(scores)
        no_answer = sum(1 for s in sequences if qa_em.extract_solution(s) is None)
        print(f"[result] {dataset}: EM = {em:.4f}   "
              f"({no_answer}/{len(sequences)} produced no parseable <answer>)")

        if dump:
            for seq, q, s in zip(sequences, questions, scores):
                dump.write(json.dumps({
                    "pair": args.pair, "role": args.role, "dataset": dataset,
                    "question": q["question"], "golden": q["golden_answers"],
                    "sequence": seq, "score": s,
                }) + "\n")
            dump.flush()

        row = {
            "version": pair.version, "size": pair.size, "algo": pair.algo,
            "with_search": pair.with_search, "model_role": args.role,
            "dataset": dataset, "em": round(em, 4), "n_questions": len(questions),
        }
        rows.append(row)
        append_rows(args.out, [row])   # write as we go; long runs get interrupted

    if dump:
        dump.close()

    avg = sum(r["em"] for r in rows) / len(rows)
    print(f"\n[done] {args.pair} / {args.role}: "
          f"Avg over {len(rows)} dataset(s) = {avg:.4f}  ->  {args.out}")


if __name__ == "__main__":
    main()
