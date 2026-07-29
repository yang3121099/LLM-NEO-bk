#!/usr/bin/env python3
"""Tests for the rollout loop, the EM scoring contract, and the aggregator.

vLLM and the retrieval server are stubbed, so this runs on CPU with no model and
no index.  What it actually pins down is the part that is easy to get silently
wrong: that we feed `qa_em.compute_score_em` the same string verl feeds it
(prompt + rollout), so the two-`<answer>` rule in `extract_solution` resolves the
way the official harness intends.

Run:  python shadow_rl/tests/test_evaluate.py --search-r1-root ~/Search-R1
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class FakeOutput:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class ScriptedLLM:
    """Replays a canned response per prompt per turn."""

    def __init__(self, script):
        self.script = script      # prompt_index -> list of responses by turn
        self.turn_of = {}
        self.calls = 0

    def generate(self, prompts, params):
        self.calls += 1
        out = []
        for p in prompts:
            idx = int(p.split("#")[1].split(" ")[0])
            turn = self.turn_of.get(idx, 0)
            self.turn_of[idx] = turn + 1
            responses = self.script[idx]
            out.append(FakeOutput(responses[min(turn, len(responses) - 1)]))
        return out


class FakeTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, queries):
        self.queries.extend(queries)
        return [f"Doc 1(Title: T-{q}) body about {q}\n" for q in queries]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-r1-root", default=os.environ.get("SEARCH_R1_ROOT", ""))
    args = ap.parse_args()

    # evaluate.py imports vllm lazily inside run_rollouts; stub it out.
    sys.modules["vllm"] = types.SimpleNamespace(
        SamplingParams=lambda **kw: types.SimpleNamespace(**kw), LLM=object
    )
    import evaluate as ev

    print("\n[1] prompt templates satisfy the qa_em last-match contract")
    import re
    for name, prompt in (("search", ev.SEARCH_PROMPT), ("no-search", ev.NOSEARCH_PROMPT)):
        m = re.findall(r"<answer>(.*?)</answer>", prompt, re.DOTALL)
        check(f"{name} prompt yields >=2 answer matches ending in Beijing",
              len(m) >= 2 and m[-1].strip() == "Beijing", f"{len(m)} matches")

    print("\n[2] response truncation matches Search-R1")
    check("cuts at first </search>",
          ev.truncate_response("<think>a</think><search>q</search> trailing junk")
          == "<think>a</think><search>q</search>")
    check("cuts at first </answer>",
          ev.truncate_response("<think>a</think><answer>Paris</answer> more")
          == "<think>a</think><answer>Paris</answer>")
    check("search wins when both present",
          ev.truncate_response("<search>q</search><answer>x</answer>").endswith("</search>"))
    check("passes through when neither present",
          ev.truncate_response("<think>unfinished") == "<think>unfinished")

    print("\n[3] action parsing")
    check("search action", ev.parse_action("<search> who? </search>") == ("search", "who?"))
    check("answer action", ev.parse_action("<answer> Paris </answer>") == ("answer", "Paris"))
    check("malformed -> None", ev.parse_action("<think>nothing</think>") == (None, ""))

    print("\n[4] rollout loop: search / answer / malformed")
    script = {
        0: ["<think>t</think><search>capital of france</search>",
            "<think>t</think><answer>Paris</answer>"],          # searches once, answers
        1: ["<think>t</think><answer>Berlin</answer>"],          # answers immediately
        2: ["<think>i am confused</think>",                      # malformed, then answers
            "<think>t</think><answer>Rome</answer>"],
    }
    prompts = [f"PROMPT #{i} <answer> Beijing </answer> Question: q{i}?\n" for i in range(3)]
    retriever = FakeRetriever()
    llm = ScriptedLLM(script)

    seqs = ev.run_rollouts(llm, FakeTokenizer(), prompts, retriever,
                           max_turns=4, max_response_length=500, max_obs_length=500)

    check("search result wrapped in <information>", "<information>" in seqs[0]
          and "Doc 1(Title: T-capital of france)" in seqs[0])
    check("retriever saw the query", retriever.queries == ["capital of france"])
    check("answered trajectory has no <information>", "<information>" not in seqs[1])
    check("malformed gets the retry nudge", "My previous action is invalid" in seqs[2])
    check("malformed trajectory still answers", "<answer>Rome</answer>" in seqs[2])
    check("prompt is retained in the scored string",
          all(seqs[i].startswith(prompts[i]) for i in range(3)))

    print("\n[5] observation truncated to max_obs_length")
    long_ret = types.SimpleNamespace(search=lambda qs: ["word " * 400 for _ in qs])
    seqs_trunc = ev.run_rollouts(
        ScriptedLLM({0: ["<search>q</search>", "<answer>A</answer>"]}),
        FakeTokenizer(), ["PROMPT #0 <answer> Beijing </answer>\n"], long_ret,
        max_turns=4, max_response_length=500, max_obs_length=50,
    )
    obs_len = len(seqs_trunc[0].split("<information>")[1].split("</information>")[0].split())
    check("observation clipped", obs_len <= 50, f"{obs_len} tokens")

    print("\n[6] max_turns budget is respected")
    forever = ScriptedLLM({0: ["<search>q</search>"]})
    ev.run_rollouts(forever, FakeTokenizer(), ["PROMPT #0 x\n"], retriever,
                    max_turns=4, max_response_length=500, max_obs_length=500)
    check("4 loop turns + 1 final rollout", forever.calls == 5, f"{forever.calls} generate calls")

    # ---------------------------------------------------------------- scoring
    if not args.search_r1_root or not os.path.isdir(args.search_r1_root):
        print("\n[7] SKIPPED scoring checks (pass --search-r1-root to enable)")
    else:
        print("\n[7] scoring against the official qa_em")
        qa_em = ev.load_qa_em(args.search_r1_root)
        golden = {"target": ["Paris"]}

        check("correct answer scores 1",
              qa_em.compute_score_em(solution_str=seqs[0], ground_truth=golden) == 1)
        check("wrong answer scores 0",
              qa_em.compute_score_em(solution_str=seqs[1], ground_truth=golden) == 0)

        # The trap: scored on its own, a response has a single <answer> match and
        # extract_solution gives up on fewer than two -- everything would score 0.
        response_only = "<think>t</think><answer>Paris</answer>"
        check("response-only string is unscoreable (why we keep the prompt)",
              qa_em.extract_solution(response_only) is None
              and qa_em.compute_score_em(solution_str=response_only, ground_truth=golden) == 0)
        check("prompt+response extracts the model's answer",
              qa_em.extract_solution(seqs[0]) == "Paris")

        # The real Search-R1 prompt contributes two matches on its own, so an
        # unanswered rollout falls back to the example. Pinned so the quirk is
        # visible rather than surprising.
        real_prompt = ev.SEARCH_PROMPT.format(question="What is the capital of France?")
        check("real prompt alone falls back to the Beijing example",
              qa_em.extract_solution(real_prompt) == "Beijing")
        check("real prompt + answer extracts the model's answer",
              qa_em.extract_solution(real_prompt + "<answer> Paris </answer>") == "Paris")
        check("normalisation is applied (case/punctuation)",
              qa_em.compute_score_em(
                  solution_str=prompts[0] + "<answer> the PARIS. </answer>",
                  ground_truth={"target": ["Paris"]}) == 1)

    # ------------------------------------------------------------- aggregator
    print("\n[8] aggregator renders FINDINGS.md")
    from pairs import DATASETS
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "results.csv")
        md_path = os.path.join(tmp, "FINDINGS.md")
        em_by_role = {"instruct_baseline": 0.10, "rl_on_instruct": 0.224,
                      "rl_on_base": 0.229, "shadow": 0.250}
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ev.CSV_FIELDS)
            w.writeheader()
            for role, em in em_by_role.items():
                for d in DATASETS:
                    w.writerow({"version": "v0.2", "size": "3b", "algo": "ppo",
                                "with_search": False, "model_role": role,
                                "dataset": d, "em": em})

        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "aggregate.py"),
             "--results", csv_path, "--out", md_path],
            capture_output=True, text=True,
        )
        check("aggregate.py succeeded", proc.returncode == 0, proc.stderr.strip()[:200])
        md = open(md_path).read() if os.path.exists(md_path) else ""
        check("summary lists the pair", "`ppo-nosearch-3b-v0.2`" in md)
        check("shadow - rl_on_instruct margin", "+0.026" in md)
        check("shadow - rl_on_base margin", "+0.021" in md)
        check("harness validation section present", "Harness validation" in md)
        check("reproduction delta computed", "+0.000" in md)
        check("unevaluated pairs listed", "Not yet evaluated" in md)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
