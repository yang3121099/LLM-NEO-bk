#!/usr/bin/env python3
"""Download and convert nohurry/Opus-4.6-Reasoning-3000x-filtered to LlamaFactory sharegpt format.

Output: data/opus_reasoning_3k.json
Format: [{"conversations": [{"role": "user", "content": problem}, {"role": "assistant", "content": "<think>thinking</think>\n\nsolution"}]}]
"""
import json
import sys

def main():
    from datasets import load_dataset

    print("[loading] Downloading nohurry/Opus-4.6-Reasoning-3000x-filtered ...")
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")

    output = []
    max_len = 0
    for row in ds:
        # Combine thinking + solution as assistant response
        solution = row.get("solution", "").strip()
        problem = row.get("problem", "").strip()
        response = solution

        total_len = len(problem) + len(response)
        if total_len > max_len:
            max_len = total_len

        output.append({
            "conversations": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": response},
            ]
        })

    out_path = "data/opus_reasoning_3k.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    print(f"[done] Saved {len(output)} samples to {out_path}")
    print(f"[info] Max character length: {max_len} (approx {max_len // 4} tokens)")
    print(f"[info] Recommended cutoff_len: check token distribution below")

    # Token length distribution (char-based approximation: 1 token ≈ 4 chars)
    lengths = []
    for item in output:
        total = sum(len(m["content"]) for m in item["conversations"])
        lengths.append(total // 4)  # rough token estimate

    lengths.sort()
    n = len(lengths)
    print(f"  p50={lengths[n//2]}, p90={lengths[int(n*0.9)]}, p95={lengths[int(n*0.95)]}, p99={lengths[int(n*0.99)]}, max={lengths[-1]}")


if __name__ == "__main__":
    main()
