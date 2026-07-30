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
    "dapo": [("BytedTsinghua-SIA/DAPO-Math-17k", None),
             ("haizhongzheng/DAPO-Math-17k-Processed", None),
             ("open-r1/DAPO-Math-17k-Processed", None)],
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

INSTRUCTION = (
    "Solve the following math problem. Reason step by step, and put your final "
    "answer within \\boxed{}.\n\n{problem}"
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


def to_records(rows, data_source):
    """verl expects prompt as a chat message list plus reward_model.ground_truth."""
    out = []
    first = rows[0]
    qf = pick(first, QUESTION_FIELDS)
    af = pick(first, ANSWER_FIELDS)
    if qf is None or af is None:
        raise RuntimeError(
            f"{data_source}: cannot find question/answer fields in {list(first)}")
    print(f"       using question='{qf}' answer='{af}'")

    for idx, row in enumerate(rows):
        answer = row[af]
        if isinstance(answer, dict):                 # already verl-shaped
            answer = answer.get("ground_truth", answer.get("target"))
        out.append({
            "data_source": data_source,
            "prompt": [{"role": "user",
                        "content": INSTRUCTION.format(problem=str(row[qf]).strip())}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
            "extra_info": {"index": idx, "split": data_source},
        })
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
