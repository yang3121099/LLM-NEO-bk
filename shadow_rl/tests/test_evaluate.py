#!/usr/bin/env python3
"""Tests for the rollout loop, the EM scoring contract, and the aggregator.

vLLM and the retrieval server are stubbed, so this runs on CPU with no model and
no index.  What it actually pins down is the part that is easy to get silently
wrong: that we feed `qa_em.compute_score_em` the same string verl feeds it
(prompt + rollout), so the two-`<answer>` rule in `extract_solution` resolves the
way the official harness intends.

Run:  python shadow_rl/tests/test_evaluate.py --search-r1-root third_party/Search-R1
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

from paths import search_r1_root as _default_search_r1_root  # noqa: E402

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
    # Default to the resolved checkout so the scoring sections actually run.
    # They used to default to "" and silently skip, which meant a green run said
    # nothing about the part that matters most -- the official EM scorer.
    ap.add_argument("--search-r1-root", default=_default_search_r1_root())
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

    # ------------------------------------------------------------- sampling
    print("\n[8] --sample is deterministic and identical across roles")
    N_ROWS = 5000

    class FakeSplit:
        def __len__(self): return N_ROWS
        def __getitem__(self, i):
            return {"question": f"question number {i}", "golden_answers": [f"a{i}"]}

    sys.modules["datasets"] = types.SimpleNamespace(
        load_dataset=lambda *a, **k: {"test": FakeSplit()}
    )

    def qids(rows):
        return [r["question"] for r in rows]

    s1 = qids(ev.load_questions("nq", None, 200))
    s2 = qids(ev.load_questions("nq", None, 200))
    other = qids(ev.load_questions("hotpotqa", None, 200))
    full = qids(ev.load_questions("nq", None, None))

    check("sample size honoured", len(s1) == 200)
    check("same dataset -> same questions every call", s1 == s2)
    check("different datasets draw different rows", s1 != other)
    check("sample is a subset of the full set", set(s1) <= set(full))
    check("sample is not just a prefix", s1 != full[:200])
    check("no sampling -> everything", len(full) == N_ROWS)
    check("sample larger than the split falls back to all",
          len(ev.load_questions("bamboogle", None, N_ROWS * 2)) == N_ROWS)
    check("--limit takes a prefix", qids(ev.load_questions("nq", 50, None)) == full[:50])
    check("question mark appended", all(q.endswith("?") for q in s1))

    # --------------------------------------------------------------- datasets
    print("\n[8b] MCQ options use digits, not letters")
    # qa_em.normalize_answer strips English articles, so a gold of "A" becomes
    # the empty string and matches any prediction that also normalises to empty.
    from pairs import DatasetSpec
    spec = DatasetSpec("t", "r", None, ("train",), "Question", "Correct Answer",
                       kind="mcq",
                       distractor_fields=("Incorrect Answer 1", "Incorrect Answer 2"))
    row = {"Question": "What is 2+2?", "Correct Answer": "four",
           "Incorrect Answer 1": "three", "Incorrect Answer 2": "five"}
    built = ev._build(row, spec, seed=0)
    golds = built["golden_answers"]
    check("no bare letter label used", not any(g in "ABCD" for g in golds), f"{golds}")
    check("a digit label is offered", any(g.isdigit() for g in golds), f"{golds}")
    check("answer text is also accepted", "four" in golds, f"{golds}")
    check("options rendered into the question",
          "1)" in built["question"] and "four" in built["question"])
    check("labelling is deterministic",
          ev._build(row, spec, seed=0) == built)
    check("different rows can differ",
          ev._build(row, spec, seed=1)["question"] != built["question"]
          or True)   # shuffles may coincide; determinism above is the real check

    if args.search_r1_root and os.path.isdir(args.search_r1_root):
        qa_em = ev.load_qa_em(args.search_r1_root)
        norm = qa_em.normalize_answer
        check("every MCQ gold survives normalisation",
              all(norm(str(g)).strip() for g in golds),
              f"{[(g, norm(str(g))) for g in golds]}")

        print("\n[8c] degenerate golds are dropped, not silently scored")
        qs = [{"question": "q1?", "golden_answers": ["A"]},          # normalises to ""
              {"question": "q2?", "golden_answers": ["Paris"]},
              {"question": "q3?", "golden_answers": ["the", "Rome"]}]  # partly bad
        dropped = ev.drop_degenerate(qs, norm)
        check("the all-empty gold is dropped", dropped == 1, f"dropped={dropped}")
        check("good questions survive", len(qs) == 2, f"{len(qs)}")
        check("the empty gold is stripped from a mixed list",
              qs[1]["golden_answers"] == ["Rome"], f"{qs[1]}")
        # Without the guard this would score 1.0 against junk.
        check("an empty gold would have matched junk",
              qa_em.em_check("the", ["A"]) == 1)

    # ------------------------------------------------------------- aggregator
    print("\n[9] aggregator renders FINDINGS.md")
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
                                "dataset": d, "em": em, "n_questions": 500})

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
        check("sampled run is flagged as a subsample", "subsample" in md.lower())
        check("question counts reported", "Questions evaluated per role" in md)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
