#!/usr/bin/env python3
"""
Preprocess DeepMath-103K for LlamaFactory SFT training.

Each sample has 3 R1 solutions in `r1_solutions`. This script expands
them into separate training examples (question → r1_solution), tripling
the dataset to ~309K examples in sharegpt format.

Usage:
    python3 scripts/prepare_deepmath_103k.py [--output data/deepmath_103k_sft.json]

Output format (sharegpt):
    [
      {"conversations": [
         {"from": "human", "value": "<question>"},
         {"from": "gpt",   "value": "<r1_solution>"}
      ]},
      ...
    ]
"""
import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/deepmath_103k_sft.json")
    parser.add_argument("--max-solutions", type=int, default=3,
                        help="Max R1 solutions per question (1-3)")
    args = parser.parse_args()

    print("Loading zwhe99/DeepMath-103K from HuggingFace ...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  Loaded {len(ds)} samples")

    records = []
    for row in ds:
        question = row["question"]
        solutions = row["r1_solutions"]
        for sol in solutions[: args.max_solutions]:
            if not sol or not sol.strip():
                continue
            records.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": sol},
                ]
            })

    print(f"  Expanded to {len(records)} training examples")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
