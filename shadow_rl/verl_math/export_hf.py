#!/usr/bin/env python3
"""Convert a verl FSDP checkpoint into a HuggingFace directory.

merge.py reads safetensors + config + tokenizer, so a verl checkpoint has to be
exported before it can take part in a Shadow-FT graft.

    python shadow_rl/verl_math/export_hf.py \
        --ckpt shadow_rl/verl_math/ckpt/qwen3-4b-base-dapo-math-grpo \
        --out  shadow_rl/verl_math/hf/qwen3-4b-base-rl \
        --base Qwen/Qwen3-4B-Base
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def newest_step(ckpt_dir: str) -> str:
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "global_step_*")),
                   key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not steps:
        return ckpt_dir
    print(f"[info] {len(steps)} checkpoint(s); using {os.path.basename(steps[-1])}")
    return steps[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="verl default_local_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True,
                    help="original model, for config and tokenizer")
    ap.add_argument("--step", default=None, help="global_step_N (default: newest)")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    step_dir = os.path.join(args.ckpt, args.step) if args.step else newest_step(args.ckpt)
    actor = os.path.join(step_dir, "actor")
    if os.path.isdir(actor):
        step_dir = actor

    # verl ships a converter for its sharded FSDP format; prefer it when present.
    merged = os.path.join(step_dir, "huggingface")
    if os.path.isdir(merged) and glob.glob(os.path.join(merged, "*.safetensors")):
        print(f"[info] verl already exported HF weights at {merged}")
        src = merged
    else:
        shards = sorted(glob.glob(os.path.join(step_dir, "model_world_size_*_rank_*.pt")))
        if not shards:
            sys.exit(
                f"[fail] no FSDP shards or exported weights under {step_dir}.\n"
                "       If your verl version ships scripts/model_merger.py, run that "
                "instead:\n"
                f"       python $VERL_ROOT/scripts/model_merger.py --local_dir {step_dir}")
        print(f"[info] merging {len(shards)} FSDP shard(s)")
        state = {}
        for shard in shards:
            part = torch.load(shard, map_location="cpu", weights_only=False)
            state.update(part.get("model", part))
        # Strip the FSDP/actor prefixes verl adds.
        cleaned = {}
        for k, v in state.items():
            for prefix in ("_fsdp_wrapped_module.", "module.", "actor_module."):
                k = k.replace(prefix, "")
            cleaned[k] = v
        config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[warn] {len(missing)} missing key(s), e.g. {missing[:3]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected key(s), e.g. {unexpected[:3]}")
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out, safe_serialization=True)
        src = None

    if src:
        os.makedirs(args.out, exist_ok=True)
        import shutil
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(args.out, f))

    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    print(f"[ok] HuggingFace model at {args.out}")
    print("     Sanity-check it before merging:")
    print(f"       python shadow_rl/smoke_test.py --model {args.out}")


if __name__ == "__main__":
    main()
