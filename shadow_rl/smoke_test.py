#!/usr/bin/env python3
"""Confirm a merged model loads and generates coherent text before a full eval.

A grafted model can load fine and still be nonsense -- a dtype mistake or a
mismatched tokenizer shows up as high perplexity and degenerate text long before
it shows up as a bad EM score.  This is the cheap check to run first.

    python shadow_rl/smoke_test.py --model shadow_rl/merged/ppo-nosearch-3b-v0.2
"""

from __future__ import annotations

import argparse
import sys

PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Explain in two sentences why the sky appears blue.",
    "Answer the given question. You must conduct reasoning inside <think> and </think> first. "
    "After reasoning, you can directly provide the answer inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>. Question: Who wrote the play Hamlet?\n",
]

# Coherent English on these prompts sits well under this; a broken merge blows past it.
PPL_LIMIT = 30.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ppl-limit", type=float, default=PPL_LIMIT)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"[ok] loaded, dtype={next(model.parameters()).dtype}, "
          f"chat_template={'yes' if tokenizer.chat_template else 'no'}")

    failures = []

    # 1. Perplexity on plain English -- the sharpest cheap signal of a bad merge.
    text = (
        "The Amazon rainforest is the largest tropical rainforest in the world. "
        "It spans several countries in South America and is home to a vast number "
        "of plant and animal species."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        ppl = torch.exp(model(ids, labels=ids).loss).item()
    ok = ppl < args.ppl_limit
    print(f"\n[perplexity] {ppl:.2f} on reference English  "
          f"({'ok' if ok else f'FAIL, expected < {args.ppl_limit}'})")
    if not ok:
        failures.append(f"perplexity {ppl:.2f} >= {args.ppl_limit}")

    # 2. Generation, through the chat path the eval will use.
    for prompt in PROMPTS:
        text_in = prompt
        if tokenizer.chat_template:
            text_in = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        inputs = tokenizer(text_in, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n--- prompt ---\n{prompt.strip()[:160]}\n--- completion ---\n{completion.strip()[:400]}")
        if not completion.strip():
            failures.append("empty completion")

    print("\n" + "=" * 60)
    if failures:
        print("SMOKE TEST FAILED: " + "; ".join(failures))
        print("Do not evaluate this checkpoint until the merge is fixed.")
        sys.exit(1)
    print("smoke test passed -- model loads and generates coherent text")
    print("Read the completions above before trusting the eval.")


if __name__ == "__main__":
    main()
