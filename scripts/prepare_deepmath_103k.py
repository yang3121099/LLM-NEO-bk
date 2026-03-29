#!/usr/bin/env python3
"""
Preprocess DeepMath-103K for LlamaFactory SFT training.

Each sample has 3 R1 solutions (`r1_solution_1/2/3`). This script expands
them into separate training examples (question → r1_solution), tripling
the dataset to ~309K examples in sharegpt format.

For the full dataset, output is split into 4 shards to avoid PyArrow's
2GB offset limit when loading large JSON files.

Usage:
    # Full dataset (~309K examples, 4 shards)
    python3 scripts/prepare_deepmath_103k.py

    # 2K demo (for quick validation on new machines)
    python3 scripts/prepare_deepmath_103k.py --demo

    # Custom size
    python3 scripts/prepare_deepmath_103k.py --max-questions 5000
"""
import argparse
import json
import math
from datasets import load_dataset


SHARD_SIZE = 80000  # ~80K examples per shard, well under 2GB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None,
                        help="Output path (default: auto based on size)")
    parser.add_argument("--max-solutions", type=int, default=3,
                        help="Max R1 solutions per question (1-3)")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Limit number of questions (0=all)")
    parser.add_argument("--demo", action="store_true",
                        help="Generate 2K demo dataset for validation")
    args = parser.parse_args()

    if args.demo:
        args.max_questions = 667   # 667 × 3 solutions ≈ 2000 examples
        if args.output is None:
            args.output = "data/deepmath_2k_demo.json"
    else:
        if args.output is None:
            args.output = "data/deepmath_103k_sft.json"

    log = (lambda *a, **kw: None) if args.demo else print

    log("Loading zwhe99/DeepMath-103K from HuggingFace ...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    log(f"  Loaded {len(ds)} samples")

    if args.max_questions > 0:
        ds = ds.select(range(min(args.max_questions, len(ds))))
        log(f"  Using first {len(ds)} questions")

    records = []
    sol_keys = [f"r1_solution_{i}" for i in range(1, args.max_solutions + 1)]
    for row in ds:
        question = row["question"]
        for key in sol_keys:
            sol = row.get(key, "")
            if not sol or not sol.strip():
                continue
            records.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": sol},
                ]
            })

    log(f"  Expanded to {len(records)} training examples")

    # For small datasets (demo) or when output is explicitly set, write single file
    if args.demo or len(records) <= SHARD_SIZE:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        log(f"  Saved to {args.output}")
    else:
        # Split into shards to avoid PyArrow 2GB offset overflow
        n_shards = math.ceil(len(records) / SHARD_SIZE)
        base = args.output.replace(".json", "")
        shard_files = []
        for i in range(n_shards):
            start = i * SHARD_SIZE
            end = min((i + 1) * SHARD_SIZE, len(records))
            shard_path = f"{base}_shard{i}.json"
            with open(shard_path, "w", encoding="utf-8") as f:
                json.dump(records[start:end], f, ensure_ascii=False, indent=1)
            shard_files.append(shard_path)
            log(f"  Shard {i}: {end - start} examples → {shard_path}")
        log(f"  Total: {len(records)} examples in {n_shards} shards")


if __name__ == "__main__":
    main()
