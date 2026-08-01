#!/usr/bin/env python3
"""Evaluate a model on AIME24 / AIME25 / AMC23 and append to a results CSV.

Same shape as shadow_rl/evaluate.py but for the math track: single-turn
generation, `\\boxed{}` answer extraction, and the rule-based reward reused
verbatim from math_reward.py so training and evaluation agree on what counts as
correct.

These sets are tiny (30/30/40), so a single greedy pass is high variance. `--k`
runs k samples per problem and reports avg@k, which is the standard mitigation.

    python shadow_rl/verl_math/eval_math.py \
        --model Qwen/Qwen3-4B-Base --role base_baseline --k 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_reward import compute_score  # noqa: E402
from prepare_data import INSTRUCTION, SOURCES, load_first, pick, QUESTION_FIELDS, ANSWER_FIELDS  # noqa: E402

BENCHMARKS = ("aime24", "aime25", "amc23")
CSV_FIELDS = ["model_role", "model", "benchmark", "accuracy", "n_problems", "k"]
MATH_ROLES = ["base_baseline", "instruct_baseline",
              "rl_on_instruct", "rl_on_base", "shadow"]


def load_problems(name):
    rows, repo = load_first(SOURCES[name])
    qf, af = pick(rows[0], QUESTION_FIELDS), pick(rows[0], ANSWER_FIELDS)
    out = []
    for row in rows:
        answer = row[af]
        if isinstance(answer, dict):
            answer = answer.get("ground_truth", answer.get("target"))
        out.append({"question": str(row[qf]).strip(), "answer": str(answer).strip()})
    return out, repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--role", required=True, choices=MATH_ROLES)
    ap.add_argument("--out", default="shadow_rl/verl_math/math_results.csv")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--k", type=int, default=4, help="samples per problem (avg@k)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N problems per benchmark (demo runs only)")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="ignored when --k 1, which uses greedy decoding")
    ap.add_argument("--max-tokens", type=int, default=31744)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--enforce-eager", action="store_true", default=None,
                    help="disable CUDA graphs; auto-detected from GPU arch if omitted")
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.enforce_eager is None:
        try:
            import torch
            major, _ = torch.cuda.get_device_capability(0)
            args.enforce_eager = major >= 10
        except Exception:
            args.enforce_eager = False

    wanted = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = set(wanted) - set(BENCHMARKS)
    if unknown:
        sys.exit(f"[fail] unknown benchmark(s): {', '.join(sorted(unknown))}")

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True,
              dtype="bfloat16", enforce_eager=args.enforce_eager)

    # k=1 is greedy and reproducible; k>1 needs sampling or every draw is identical.
    params = SamplingParams(
        n=args.k,
        temperature=0.0 if args.k == 1 else args.temperature,
        top_p=1.0, max_tokens=args.max_tokens,
    )

    rows = []
    for name in wanted:
        problems, repo = load_problems(name)
        if args.limit:
            problems = problems[:args.limit]
        prompts = []
        for p in problems:
            text = INSTRUCTION.format(problem=p["question"])
            if tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            prompts.append(text)

        print(f"\n[eval] {name}: {len(prompts)} problems x k={args.k}  ({repo})")
        outputs = llm.generate(prompts, params)

        total = 0.0
        for out, problem in zip(outputs, problems):
            scores = [compute_score("math", c.text, problem["answer"])
                      for c in out.outputs]
            total += sum(scores) / len(scores)      # avg@k, not pass@k
        acc = total / len(problems)
        print(f"[result] {name}: avg@{args.k} = {acc:.4f}")
        rows.append({"model_role": args.role, "model": args.model,
                     "benchmark": name, "accuracy": round(acc, 4),
                     "n_problems": len(problems), "k": args.k})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    is_new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerows(rows)

    avg = sum(r["accuracy"] for r in rows) / len(rows)
    print(f"\n[done] {args.role}: mean over {len(rows)} benchmark(s) = {avg:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
