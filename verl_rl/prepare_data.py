#!/usr/bin/env python3
"""Write the RL training set in the parquet form verl's RLHFDataset reads.

verl expects one row per prompt with the answer carried alongside, so the reward
function can score a rollout without a second lookup:

    data_source   str                     free-form; reward_math.py ignores it
    prompt        list[{role, content}]   chat messages, template applied by verl
    ability       str
    reward_model  {style, ground_truth}   what the scorer compares against
    extra_info    {split, index, ...}

The prompt is stored as *messages*, not as pre-rendered text, and both roles
(base and instruct) go through the same `apply_chat_template` path at training
time. That is deliberate: Shadow-FT grafts the base-side update onto the
instruct backbone, so the two runs have to differ in the weights they start
from and in nothing else. Rendering the base prompt differently would confound
the comparison with a prompt-format difference.

    python verl_rl/prepare_data.py --out results/verl_rl/qwen3-4b-grpo/data
    python verl_rl/prepare_data.py --out /tmp/d --demo      # 200 rows, no network
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


# The instruction appended to every question. It asks for the \boxed{} form that
# reward_math.py reads first and that every math eval harness expects, so the
# behaviour RL rewards is the behaviour the benchmarks measure.
INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Columns DeepMath-103K uses. Kept in one place because the other candidate
# datasets (DAPO, OpenR1-Math) name the same two fields differently.
FIELD_ALIASES = {
    "question": ("question", "problem", "prompt", "query"),
    "answer": ("final_answer", "answer", "solution", "ground_truth"),
}


def pick_field(row: dict, kind: str):
    for name in FIELD_ALIASES[kind]:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def make_row(question: str, answer: str, split: str, index: int, data_source: str) -> dict:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "answer": str(answer),
        },
    }


def demo_rows(n: int, data_source: str):
    """A synthetic set, so the pipeline can be exercised with no network.

    Arithmetic the model can actually get right, which makes a smoke run show a
    reward curve that moves rather than a flat zero.
    """
    rng = random.Random(0)
    rows = []
    for i in range(n):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        rows.append(make_row(f"What is {a} + {b}?", a + b, "train", i, data_source))
    return rows


def load_hf_rows(dataset: str, split: str, limit: int, data_source: str):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rows, skipped = [], 0
    for i, item in enumerate(ds):
        if limit and len(rows) >= limit:
            break
        question = pick_field(item, "question")
        answer = pick_field(item, "answer")
        if question is None or answer is None:
            skipped += 1
            continue
        rows.append(make_row(str(question).strip(), answer, split, i, data_source))
    if skipped:
        print(f"  skipped {skipped} row(s) with no question or no verifiable answer")
    return rows


def write_parquet(rows, path: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required to write parquet: pip install pandas pyarrow")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {len(rows)} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dataset", default="zwhe99/DeepMath-103K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-source", default="deepmath",
                    help="stored on every row; only matters if you swap in "
                         "verl's data_source-based reward registry")
    ap.add_argument("--train-size", type=int, default=0, help="0 = all")
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--demo", action="store_true",
                    help="200 synthetic arithmetic rows, no download")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        print(f"building the demo set ({args.dataset} not downloaded)")
        rows = demo_rows(200, args.data_source)
    else:
        print(f"loading {args.dataset} split={args.split}")
        # Read train_size + val_size, since the validation set is held out of
        # the same pool below.
        limit = (args.train_size + args.val_size) if args.train_size else 0
        rows = load_hf_rows(args.dataset, args.split, limit, args.data_source)

    if not rows:
        sys.exit("no usable rows -- check --dataset and the field names")

    # Held out from the same pool with a fixed seed, so both the base and the
    # instruct run are validated on identical questions. Comparing two RL runs
    # scored on different validation sets would be meaningless.
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(args.val_size, max(0, len(rows) - 1))
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    if args.train_size:
        train_rows = train_rows[: args.train_size]

    for i, row in enumerate(train_rows):
        row["extra_info"]["split"], row["extra_info"]["index"] = "train", i
    for i, row in enumerate(val_rows):
        row["extra_info"]["split"], row["extra_info"]["index"] = "val", i

    write_parquet(train_rows, os.path.join(args.out, "train.parquet"))
    write_parquet(val_rows, os.path.join(args.out, "val.parquet"))

    meta = {
        "dataset": "demo" if args.demo else args.dataset,
        "data_source": args.data_source,
        "instruction": INSTRUCTION,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  train={len(train_rows)}  val={len(val_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
