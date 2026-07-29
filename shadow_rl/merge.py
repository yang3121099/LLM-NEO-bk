#!/usr/bin/env python3
"""Shadow-FT weight grafting for matched RL checkpoint pairs.

Computes, for a pair of matched Search-R1 releases::

    Delta_K  = RL(W_B) - W_B     # what the base model learned under RL
    W_shadow = W_I + Delta_K     # graft that update onto the instruct backbone

The merge streams safetensors shard by shard and writes incrementally, so at no
point are four models resident in memory -- peak usage is a handful of
individual tensors.  Arithmetic is done in fp32 and cast back to the dtype of
``W_I``: the released RL checkpoints are frequently fp32 while the official Qwen
weights are bf16, and taking a small difference in bf16 throws away most of the
signal.

Tokenizer and config are taken from ``W_I`` -- the merged model is an instruct
model in every respect that matters and should be prompted as one.

Example
-------
    python shadow_rl/merge.py \
        --base        Qwen/Qwen2.5-3B \
        --instruct    Qwen/Qwen2.5-3B-Instruct \
        --rl-base     PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.2 \
        --out         /models/shadow-r1-3b-ppo-v0.2 \
        --rl-instruct PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Working range for sigma reported in the Shadow-FT paper.  Pairs far outside
# this are not necessarily wrong, but they are worth a second look before the
# result is believed.
SIGMA_LO, SIGMA_HI = 0.003, 0.042

# Files copied from W_I so the merged directory is a usable instruct model.
# Weight files and their index are deliberately excluded -- we write our own.
_CONFIG_SUFFIXES = (".json", ".txt", ".model", ".jinja", ".py", ".md")
_WEIGHT_RE = re.compile(r"\.(safetensors|bin|pt|pth|h5|msgpack)$")


# --------------------------------------------------------------------------- #
# checkpoint access
# --------------------------------------------------------------------------- #
def resolve(path_or_repo: str) -> str:
    """Return a local directory for a local path or a HuggingFace repo id."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    if "/" not in path_or_repo:
        raise FileNotFoundError(f"Not a directory and not a repo id: {path_or_repo}")

    from huggingface_hub import snapshot_download

    cache_root = os.environ.get("SHADOW_RL_MODEL_DIR", os.path.expanduser("~/models"))
    local_dir = os.path.join(cache_root, path_or_repo.replace("/", "__"))
    print(f"[hf] snapshot_download {path_or_repo} -> {local_dir}", flush=True)
    return snapshot_download(
        repo_id=path_or_repo,
        local_dir=local_dir,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "*.json",
            "*.txt",
            "*.model",
            "*.jinja",
        ],
    )


class Checkpoint:
    """Lazy, read-only view over a safetensors checkpoint directory.

    Tensors are fetched one at a time straight off disk; nothing is cached, so
    iterating a whole model costs one tensor of memory rather than one model.
    """

    def __init__(self, path_or_repo: str, name: str):
        self.name = name
        self.path = resolve(path_or_repo)
        self.key_to_file: Dict[str, str] = {}
        self._handles: Dict[str, object] = {}

        index_path = os.path.join(self.path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as fh:
                weight_map = json.load(fh)["weight_map"]
            self.shards = sorted(set(weight_map.values()))
            self.key_to_file = dict(weight_map)
        else:
            shards = sorted(
                f
                for f in os.listdir(self.path)
                if f.endswith(".safetensors")
            )
            if not shards:
                raise FileNotFoundError(f"No .safetensors found in {self.path}")
            self.shards = shards
            for shard in shards:
                with safe_open(os.path.join(self.path, shard), framework="pt") as fh:
                    for key in fh.keys():
                        self.key_to_file[key] = shard

    def keys(self) -> set:
        return set(self.key_to_file)

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def get(self, key: str) -> torch.Tensor:
        return self._handle(self.key_to_file[key]).get_tensor(key)

    def shape(self, key: str) -> Tuple[int, ...]:
        return tuple(self._handle(self.key_to_file[key]).get_slice(key).get_shape())

    def dtype(self, key: str) -> torch.dtype:
        return self.get(key).dtype

    def close(self) -> None:
        self._handles.clear()


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _fmt_keys(keys: Iterable[str], limit: int = 10) -> str:
    keys = sorted(keys)
    head = "\n".join(f"      {k}" for k in keys[:limit])
    if len(keys) > limit:
        head += f"\n      ... and {len(keys) - limit} more"
    return head


def validate(ckpts: List[Checkpoint], ignore: Optional[str]) -> List[str]:
    """Assert key sets and tensor shapes agree; return the merge key list.

    Raises SystemExit with a readable diff rather than a bare assertion, since
    a mismatch here usually means the wrong checkpoint was passed on the CLI.
    """
    ignore_re = re.compile(ignore) if ignore else None
    key_sets = {
        c.name: {k for k in c.keys() if not (ignore_re and ignore_re.search(k))}
        for c in ckpts
    }

    reference_name, reference = next(iter(key_sets.items()))
    problems = []
    for name, keys in key_sets.items():
        if keys == reference:
            continue
        missing, extra = reference - keys, keys - reference
        detail = [f"  {name} vs {reference_name}:"]
        if missing:
            detail.append(f"    missing {len(missing)} key(s):\n{_fmt_keys(missing)}")
        if extra:
            detail.append(f"    extra {len(extra)} key(s):\n{_fmt_keys(extra)}")
        problems.append("\n".join(detail))

    if problems:
        sys.exit(
            "[fail] checkpoint key sets do not match:\n"
            + "\n".join(problems)
            + "\n\nIf the difference is benign (e.g. a tied lm_head saved in one "
            "export but not another), re-run with --ignore-keys '<regex>'."
        )

    keys = sorted(reference)
    shape_problems = []
    for key in keys:
        shapes = {c.name: c.shape(key) for c in ckpts}
        if len(set(shapes.values())) > 1:
            shape_problems.append(f"    {key}: " + ", ".join(f"{n}={s}" for n, s in shapes.items()))
    if shape_problems:
        sys.exit("[fail] tensor shapes do not match:\n" + "\n".join(shape_problems))

    print(f"[ok] {len(keys)} keys, shapes agree across {len(ckpts)} checkpoints")
    return keys


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
class Accum:
    """fp64 accumulators for the sanity statistics."""

    def __init__(self) -> None:
        self.abs_diff = 0.0     # sum |W_B - W_I|
        self.abs_base = 0.0     # sum |W_B|
        self.abs_inst = 0.0     # sum |W_I|
        self.sq_delta_b = 0.0   # ||RL(W_B) - W_B||^2
        self.sq_base = 0.0      # ||W_B||^2

    @property
    def sigma(self) -> float:
        denom = self.abs_base + self.abs_inst
        return self.abs_diff / denom if denom else float("nan")

    @property
    def rel_delta_base(self) -> float:
        return (self.sq_delta_b ** 0.5) / (self.sq_base ** 0.5) if self.sq_base else float("nan")


def relative_delta(original: Checkpoint, tuned: Checkpoint, keys: List[str]) -> float:
    """||tuned - original|| / ||original||, Frobenius over the whole model."""
    sq_delta = sq_orig = 0.0
    for key in keys:
        w0 = original.get(key)
        if not w0.is_floating_point():
            continue
        w0 = w0.to(torch.float32)
        w1 = tuned.get(key).to(torch.float32)
        sq_delta += float((w1 - w0).double().pow(2).sum())
        sq_orig += float(w0.double().pow(2).sum())
    return (sq_delta ** 0.5) / (sq_orig ** 0.5) if sq_orig else float("nan")


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge(
    base: Checkpoint,
    instruct: Checkpoint,
    rl_base: Checkpoint,
    out_dir: str,
    keys: List[str],
    scale: float = 1.0,
) -> Accum:
    """Stream W_I + scale * (RL(W_B) - W_B) into *out_dir*, shard by shard.

    Output sharding mirrors ``W_I`` exactly, which keeps the index file and any
    weight tying consistent with the backbone we are grafting onto.
    """
    os.makedirs(out_dir, exist_ok=True)
    acc = Accum()

    # Group the merge keys by the shard they live in on the instruct side, so we
    # write one output file per instruct shard and free it before the next.
    by_shard: Dict[str, List[str]] = {}
    for key in keys:
        by_shard.setdefault(instruct.key_to_file[key], []).append(key)

    total = len(keys)
    done = 0
    for shard in sorted(by_shard):
        tensors: Dict[str, torch.Tensor] = {}
        for key in by_shard[shard]:
            w_i = instruct.get(key)

            if not w_i.is_floating_point():
                # Integer buffers carry no learned update; take them from W_I.
                tensors[key] = w_i.clone()
                done += 1
                continue

            out_dtype = w_i.dtype
            w_i32 = w_i.to(torch.float32)
            w_b32 = base.get(key).to(torch.float32)
            w_rl32 = rl_base.get(key).to(torch.float32)

            delta = w_rl32 - w_b32
            tensors[key] = (w_i32 + scale * delta).to(out_dtype)

            acc.abs_diff += float((w_b32 - w_i32).double().abs().sum())
            acc.abs_base += float(w_b32.double().abs().sum())
            acc.abs_inst += float(w_i32.double().abs().sum())
            acc.sq_delta_b += float(delta.double().pow(2).sum())
            acc.sq_base += float(w_b32.double().pow(2).sum())
            done += 1

        save_file(tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
        print(f"[write] {shard}  ({done}/{total} tensors)", flush=True)
        del tensors

    return acc


def copy_config(instruct: Checkpoint, out_dir: str, shards: List[str]) -> None:
    """Copy tokenizer/config from W_I and write a matching weight index."""
    for filename in sorted(os.listdir(instruct.path)):
        src = os.path.join(instruct.path, filename)
        if not os.path.isfile(src) or _WEIGHT_RE.search(filename):
            continue
        if filename.endswith(".safetensors.index.json"):
            continue
        if filename.endswith(_CONFIG_SUFFIXES):
            shutil.copy2(src, os.path.join(out_dir, filename))

    src_index = os.path.join(instruct.path, "model.safetensors.index.json")
    if os.path.exists(src_index):
        shutil.copy2(src_index, os.path.join(out_dir, "model.safetensors.index.json"))
    elif len(shards) > 1:
        raise RuntimeError("multi-shard output but the instruct model had no index")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Graft an RL update learned on the base model onto the instruct model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base", required=True, help="W_B, e.g. Qwen/Qwen2.5-3B")
    ap.add_argument("--instruct", required=True, help="W_I, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--rl-base", required=True, help="RL(W_B), the base-side release")
    ap.add_argument("--out", required=True, help="output directory for W_shadow")
    ap.add_argument(
        "--rl-instruct",
        default=None,
        help="RL(W_I); optional, only used to report the instruct-side delta magnitude",
    )
    ap.add_argument("--scale", type=float, default=1.0, help="scale on Delta_K")
    ap.add_argument(
        "--ignore-keys",
        default=None,
        help="regex of parameter names to exclude from the key-set check and the merge",
    )
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="report sigma and delta magnitudes without writing a merged model",
    )
    args = ap.parse_args()

    base = Checkpoint(args.base, "base")
    instruct = Checkpoint(args.instruct, "instruct")
    rl_base = Checkpoint(args.rl_base, "rl_base")
    ckpts = [base, instruct, rl_base]

    rl_instruct = None
    if args.rl_instruct:
        rl_instruct = Checkpoint(args.rl_instruct, "rl_instruct")
        ckpts.append(rl_instruct)

    keys = validate(ckpts, args.ignore_keys)

    if args.stats_only:
        acc = Accum()
        for key in keys:
            w_b = base.get(key)
            if not w_b.is_floating_point():
                continue
            w_b = w_b.to(torch.float32)
            w_i = instruct.get(key).to(torch.float32)
            delta = rl_base.get(key).to(torch.float32) - w_b
            acc.abs_diff += float((w_b - w_i).double().abs().sum())
            acc.abs_base += float(w_b.double().abs().sum())
            acc.abs_inst += float(w_i.double().abs().sum())
            acc.sq_delta_b += float(delta.double().pow(2).sum())
            acc.sq_base += float(w_b.double().pow(2).sum())
    else:
        acc = merge(base, instruct, rl_base, args.out, keys, scale=args.scale)
        copy_config(instruct, args.out, instruct.shards)
        print(f"[ok] wrote merged model to {args.out}")

    print("\n===== sanity check =====")
    print(f"sigma (base vs instruct)      : {acc.sigma:.5f}   [paper range {SIGMA_LO}-{SIGMA_HI}]")
    if not (SIGMA_LO <= acc.sigma <= SIGMA_HI):
        print("  ** WARNING: sigma is outside the paper's working range **")
    print(f"||RL(W_B)-W_B|| / ||W_B||     : {acc.rel_delta_base:.5f}")
    if rl_instruct is not None:
        rel_i = relative_delta(instruct, rl_instruct, keys)
        print(f"||RL(W_I)-W_I|| / ||W_I||     : {rel_i:.5f}")
    print("========================")

    stats = {
        "base": args.base,
        "instruct": args.instruct,
        "rl_base": args.rl_base,
        "rl_instruct": args.rl_instruct,
        "scale": args.scale,
        "sigma": acc.sigma,
        "rel_delta_base": acc.rel_delta_base,
        "sigma_in_range": bool(SIGMA_LO <= acc.sigma <= SIGMA_HI),
    }
    if rl_instruct is not None:
        stats["rel_delta_instruct"] = rel_i
    if not args.stats_only:
        with open(os.path.join(args.out, "shadow_merge_stats.json"), "w") as fh:
            json.dump(stats, fh, indent=2)

    for c in ckpts:
        c.close()


if __name__ == "__main__":
    main()
