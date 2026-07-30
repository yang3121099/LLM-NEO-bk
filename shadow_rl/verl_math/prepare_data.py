#!/usr/bin/env python3
"""Build the verl parquet files for DAPO-Math training and math validation.

Training: DAPO-Math-17k-Processed.
Validation: AIME24 + AIME25 + AMC23, concatenated into one file, with
`data_source` preserved so per-benchmark accuracy can be split back out.

    python shadow_rl/verl_math/prepare_data.py --out datasets
"""

from __future__ import annotations

import argparse
import os
import sys

# Candidate repos per benchmark, tried in order: these datasets are mirrored
# under several names and the canonical one moves.
SOURCES = {
    # "Processed" first: those are the ~17k unique prompts. The
    # BytedTsinghua-SIA release is the expanded set (~1.79M rows, each prompt
    # repeated for the DAPO recipe) and would make --limit-train sample a
    # handful of near-duplicates.
    "dapo": [("haizhongzheng/DAPO-Math-17k-Processed", None),
             ("open-r1/DAPO-Math-17k-Processed", None),
             ("BytedTsinghua-SIA/DAPO-Math-17k", None)],
    "aime24": [("HuggingFaceH4/aime_2024", None),
               ("Maxwell-Jia/AIME_2024", None),
               ("math-ai/aime24", None)],
    "aime25": [("yentinglin/aime_2025", None),
               ("math-ai/aime25", None),
               ("opencompass/AIME2025", "AIME2025-I")],
    "amc23": [("math-ai/amc23", None),
              ("zwhe99/amc23", None),
              ("knoveleng/AMC-23", None)],
}

# Field names vary across mirrors; first present wins.
QUESTION_FIELDS = ("prompt", "problem", "question", "Problem", "Question")
ANSWER_FIELDS = ("solution", "answer", "final_answer", "Answer", "reward_model")

# The braces of \boxed{} are literal, so they must be doubled -- str.format
# otherwise reads them as an auto-numbered field and raises IndexError.
INSTRUCTION = (
    "Solve the following math problem. Reason step by step, and put your final "
    "answer within \\boxed{{}}.\n\n{problem}"
)


def pick(row, names):
    for n in names:
        if n in row and row[n] is not None:
            return n
    return None


def load_first(candidates, split_hint=None):
    import datasets as hfds
    errors = []
    for repo, config in candidates:
        try:
            ds = hfds.load_dataset(repo, config) if config else hfds.load_dataset(repo)
            split = split_hint or next(
                (s for s in ("train", "test", "validation") if s in ds), None)
            if split is None:
                errors.append(f"{repo}: no usable split ({list(ds)})")
                continue
            print(f"  [ok] {repo}{'/' + config if config else ''} split={split} "
                  f"rows={len(ds[split])}")
            return ds[split], repo
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {str(exc)[:90]}")
    raise RuntimeError("none of the candidates loaded:\n    " + "\n    ".join(errors))


def _text(value) -> str:
    """Problem text from either a plain string or a verl chat-message list."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        # Already verl-shaped: [{"role": "user", "content": "..."}]
        for msg in value:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        if value and isinstance(value[0], dict):
            return str(value[0].get("content", "")).strip()
    return str(value).strip()


def _gold(value) -> str:
    if isinstance(value, dict):
        value = value.get("ground_truth", value.get("target", value.get("answer")))
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value).strip()


def to_records(rows, data_source, dedupe=True):
    """verl expects prompt as a chat message list plus reward_model.ground_truth.

    Some mirrors already store exactly that, so the question field may hold a
    message list rather than a string and the answer field a dict. Both shapes
    are handled; the instruction is applied only when the prompt is raw text, so
    an already-formatted prompt is not wrapped twice.
    """
    out = []
    seen = set()
    first = rows[0]
    qf = pick(first, QUESTION_FIELDS)
    af = pick(first, ANSWER_FIELDS)
    if qf is None or af is None:
        raise RuntimeError(
            f"{data_source}: cannot find question/answer fields in {list(first)}")
    preformatted = not isinstance(first[qf], str)
    print(f"       using question='{qf}' answer='{af}'"
          f"{'  (already chat-formatted)' if preformatted else ''}")

    for row in rows:
        problem = _text(row[qf])
        answer = _gold(row[af])
        if not problem or not answer:
            continue
        if dedupe:
            key = (problem, answer)
            if key in seen:
                continue
            seen.add(key)
        # Only wrap raw text: a prompt that already carries its own instruction
        # would otherwise get a second one bolted on.
        content = problem if preformatted else INSTRUCTION.format(problem=problem)
        out.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": len(out), "split": data_source},
        })
    if dedupe and len(out) < len(rows):
        print(f"       {len(rows)} row(s) -> {len(out)} unique")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="keep only the first N training prompts (demo runs)")
    ap.add_argument("--limit-val", type=int, default=None,
                    help="keep only the first N problems per validation benchmark")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("[fail] pandas not installed. pip install pandas pyarrow datasets")

    if not args.skip_train:
        print("[train] DAPO-Math-17k-Processed")
        rows, repo = load_first(SOURCES["dapo"])
        records = to_records(rows, "dapo_math")
        if args.limit_train:
            records = records[:args.limit_train]
            print(f"       limited to {len(records)} prompt(s) for a demo run")
        d = os.path.join(args.out, "DAPO-Math-17k-Processed")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "DAPO-Math.parquet")
        pd.DataFrame(records).to_parquet(path)
        print(f"[ok] {len(records)} rows -> {path}  (from {repo})\n")

    print("[val] AIME24 + AIME25 + AMC23")
    val = []
    for name in ("aime24", "aime25", "amc23"):
        print(f"  {name}")
        rows, repo = load_first(SOURCES[name])
        got = to_records(rows, name)
        if args.limit_val:
            got = got[:args.limit_val]
        val.extend(got)
        print(f"       {len(got)} rows from {repo}")
    d = os.path.join(args.out, "math_val")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "val.parquet")
    pd.DataFrame(val).to_parquet(path)
    print(f"\n[ok] {len(val)} validation rows -> {path}")
    print("     data_source is preserved, so per-benchmark accuracy splits back out.")


if __name__ == "__main__":
    main()
