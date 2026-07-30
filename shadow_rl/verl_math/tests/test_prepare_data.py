#!/usr/bin/env python3
"""Tests for the DAPO-Math / AIME parquet builder.

These datasets are mirrored under several names with genuinely different row
shapes -- some store a plain problem string, others already store verl's chat
message list and a reward_model dict. Getting that wrong produces a parquet that
trains on stringified Python objects, which is the kind of thing that only shows
up as a mysteriously flat reward curve hours later.

Run:  python shadow_rl/verl_math/tests/test_prepare_data.py
"""

import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import prepare_data as P  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main():
    print("\n[1] INSTRUCTION is a valid format string")
    # The braces of \boxed{} are literal. If they are not doubled, str.format
    # treats them as an auto-numbered field and raises IndexError.
    fields = [f[1] for f in string.Formatter().parse(P.INSTRUCTION) if f[1] is not None]
    check("only 'problem' is a field", fields == ["problem"], f"{fields}")
    rendered = P.INSTRUCTION.format(problem="What is 2+2?")
    check("formats without raising", "What is 2+2?" in rendered)
    check("\\boxed{} survives verbatim", "\\boxed{}" in rendered,
          repr(rendered[-40:]))

    print("\n[2] plain-string rows (the common mirror shape)")
    rows = [{"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+3?", "answer": "6"}]
    out = P.to_records(rows, "dapo_math")
    check("one record per row", len(out) == 2)
    r = out[0]
    check("prompt is a chat message list",
          isinstance(r["prompt"], list) and r["prompt"][0]["role"] == "user")
    check("instruction applied", "Reason step by step" in r["prompt"][0]["content"])
    check("problem text present", "What is 2+2?" in r["prompt"][0]["content"])
    check("ground truth extracted", r["reward_model"]["ground_truth"] == "4")
    check("data_source recorded", r["data_source"] == "dapo_math")

    print("\n[3] rows already in verl's schema")
    # This is the shape that crashed: `prompt` is a message list, not a string,
    # and `reward_model` is a dict.
    rows = [{"prompt": [{"role": "user", "content": "Compute 5*5. \\boxed{}"}],
             "reward_model": {"style": "rule", "ground_truth": "25"}}]
    out = P.to_records(rows, "dapo_math")
    check("record produced", len(out) == 1)
    content = out[0]["prompt"][0]["content"]
    check("message content taken, not stringified",
          content == "Compute 5*5. \\boxed{}", repr(content))
    check("no dict/list repr leaked into the prompt",
          "role" not in content and "{'" not in content)
    check("instruction NOT applied twice",
          "Reason step by step" not in content)
    check("ground_truth unwrapped from the dict",
          out[0]["reward_model"]["ground_truth"] == "25")

    print("\n[4] mixed and awkward answer shapes")
    check("dict with 'target'",
          P._gold({"target": "7"}) == "7")
    check("list takes the first",
          P._gold(["11", "eleven"]) == "11")
    check("plain string passes through", P._gold("  13 ") == "13")
    check("number becomes text", P._gold(42) == "42")

    print("\n[5] duplicate prompts are collapsed")
    # The expanded DAPO release repeats each prompt many times; taking the first
    # N of that with --limit-train would sample a handful of near-duplicates.
    rows = [{"problem": "same?", "answer": "1"}] * 5 + [{"problem": "other?", "answer": "2"}]
    out = P.to_records(rows, "dapo_math")
    check("6 rows -> 2 unique", len(out) == 2, f"{len(out)}")
    check("indices are renumbered after dedupe",
          [r["extra_info"]["index"] for r in out] == [0, 1])
    kept = P.to_records(rows, "dapo_math", dedupe=False)
    check("dedupe can be turned off", len(kept) == 6, f"{len(kept)}")

    print("\n[6] empty rows are dropped rather than poisoning the reward")
    rows = [{"problem": "", "answer": "1"},
            {"problem": "ok?", "answer": ""},
            {"problem": "good?", "answer": "3"}]
    out = P.to_records(rows, "dapo_math")
    check("only the complete row survives", len(out) == 1, f"{len(out)}")
    check("it is the right one", "good?" in out[0]["prompt"][0]["content"])

    print("\n[7] the Processed release is preferred for training")
    first = P.SOURCES["dapo"][0][0]
    check("first candidate is a Processed mirror", "Processed" in first, first)
    check("the expanded release is still a fallback",
          any("BytedTsinghua" in r for r, _ in P.SOURCES["dapo"]))

    print("\n[8] validation benchmarks are declared")
    for name in ("aime24", "aime25", "amc23"):
        check(f"{name} has candidates", bool(P.SOURCES.get(name)))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
