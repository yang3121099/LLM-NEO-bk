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

# The candidate datasets spell the same two fields differently, and the AIME
# sets capitalise them, so matching is case-insensitive.
FIELD_ALIASES = {
    "question": ("question", "problem", "prompt", "query"),
    "answer": ("final_answer", "answer", "solution", "ground_truth", "expected_answer"),
}

# Validation sets named by the recipe. These ids are the commonly used mirrors;
# they could not be verified from the environment this was written in (no access
# to huggingface.co), so confirm them or point at local files:
#
#   python verl_rl/prepare_data.py --val-set AIME24=/data/aime24.parquet ...
VAL_SET_SOURCES = {
    "AIME24": "Maxwell-Jia/AIME_2024",
    "AIME25": "yentinglin/aime_2025",
    "AMC23": "knoveleng/AMC-23",
}

# A parquet already in verl's shape needs no conversion -- and must not get any,
# because re-deriving the prompt would change the tokens the model trains on.
VERL_COLUMNS = {"prompt", "reward_model"}


def pick_field(row: dict, kind: str):
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in FIELD_ALIASES[kind]:
        value = lowered.get(name)
        if value not in (None, ""):
            return value
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


def load_local(path: str):
    """Rows from a parquet/json already on disk.

    The recipe names a local file (datasets/DAPO-Math-17k-Processed/
    DAPO-Math.parquet), and such files are usually already in verl's shape.
    """
    import pandas as pd

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path)
    return df.to_dict("records"), set(df.columns)


def load_hf(dataset: str, split: str):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    return list(ds), set(ds.column_names)


def load_rows(source: str, split: str, limit: int, data_source: str, tag: str):
    """Rows in verl's shape, from a local file or a hub id.

    Passes an already-verl-shaped table through untouched: rebuilding the prompt
    from a question column would change the exact tokens the model sees, which
    is the difference between reproducing a recipe and approximating it.
    """
    if os.path.exists(source):
        print(f"  reading local file {source}")
        raw, columns = load_local(source)
    else:
        print(f"  loading {source} split={split}")
        raw, columns = load_hf(source, split)

    if VERL_COLUMNS <= columns:
        print(f"  already in verl format ({len(raw)} rows) -- passing through unchanged")
        rows = raw[:limit] if limit else raw
        for i, row in enumerate(rows):
            info = row.get("extra_info")
            row["extra_info"] = dict(info) if isinstance(info, dict) else {}
            row["extra_info"].update({"split": tag, "index": i})
            row.setdefault("data_source", data_source)
            row.setdefault("ability", "math")
        return rows

    rows, skipped = [], 0
    for i, item in enumerate(raw):
        if limit and len(rows) >= limit:
            break
        question = pick_field(item, "question")
        answer = pick_field(item, "answer")
        if question is None or answer is None:
            skipped += 1
            continue
        rows.append(make_row(str(question).strip(), answer, tag, i, data_source))
    if skipped:
        print(f"  skipped {skipped} row(s) with no question or no verifiable answer")
    if not rows:
        sys.exit(f"no usable rows in {source}. Columns present: {sorted(columns)}\n"
                 f"  Expected a question-like column {FIELD_ALIASES['question']}\n"
                 f"  and an answer-like column {FIELD_ALIASES['answer']}.")
    return rows


def build_val(specs, limit_per_set: int, split: str):
    """One validation table from several named benchmark sets.

    Kept as a single file with `data_source` marking the origin, because verl
    reports validation as one number per data_source -- so AIME24, AIME25 and
    AMC23 stay separately visible in the training log.
    """
    rows = []
    for name, source in specs:
        print(f"  [{name}]")
        got = load_rows(source, "test", limit_per_set, name, split)
        for row in got:
            row["data_source"] = name
        print(f"    {len(got)} rows")
        rows.extend(got)
    for i, row in enumerate(rows):
        row["extra_info"]["index"] = i
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
    ap.add_argument("--dataset", default="DAPO-Math-17k-Processed",
                    help="local parquet/json path, or a hub id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-source", default="dapo-math-17k",
                    help="stored on every row; only matters if you swap in "
                         "verl's data_source-based reward registry")
    ap.add_argument("--train-size", type=int, default=0, help="0 = all")
    ap.add_argument("--val-sets", default="AIME24,AIME25,AMC23",
                    help="comma-separated names from the built-in table, or "
                         "empty to hold a slice out of the training data instead")
    ap.add_argument("--val-set", action="append", default=[], metavar="NAME=SOURCE",
                    help="override where one validation set comes from; repeatable")
    ap.add_argument("--val-size", type=int, default=0,
                    help="rows per validation set (0 = all of each)")
    ap.add_argument("--demo", action="store_true",
                    help="200 synthetic arithmetic rows, no download")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    overrides = {}
    for item in args.val_set:
        if "=" not in item:
            sys.exit(f"--val-set expects NAME=SOURCE, got {item!r}")
        name, source = item.split("=", 1)
        overrides[name.strip()] = source.strip()

    val_names = [n.strip() for n in args.val_sets.split(",") if n.strip()]

    if args.demo:
        print(f"building the demo set ({args.dataset} not downloaded)")
        train_rows = demo_rows(180, args.data_source)
        val_rows = demo_rows(20, args.data_source)
        for row in val_rows:
            row["extra_info"]["split"] = "val"
        val_names = ["demo"]
    elif val_names:
        # The recipe validates on three competition sets, not on held-out
        # training rows -- so the training file keeps every row.
        print(f"train: {args.dataset}")
        train_rows = load_rows(args.dataset, args.split, args.train_size,
                               args.data_source, "train")
        specs = []
        for name in val_names:
            source = overrides.get(name) or VAL_SET_SOURCES.get(name)
            if not source:
                sys.exit(f"no source known for validation set {name!r}. "
                         f"Known: {', '.join(VAL_SET_SOURCES)}. "
                         f"Pass --val-set {name}=<path or hub id>.")
            specs.append((name, source))
        print(f"val: {', '.join(n for n, _ in specs)}")
        val_rows = build_val(specs, args.val_size, "val")
    else:
        # Fallback: hold a slice out of the training pool. Seeded, so the base
        # and instruct runs are validated on identical questions -- comparing
        # two RL runs scored on different questions would be meaningless.
        print(f"train+val from {args.dataset}")
        rows = load_rows(args.dataset, args.split, 0, args.data_source, "train")
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        held_out = args.val_size or 500
        val_size = min(held_out, max(0, len(rows) - 1))
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
        "val_sets": val_names,
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
