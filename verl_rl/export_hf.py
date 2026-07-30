#!/usr/bin/env python3
"""Turn a verl checkpoint into a HuggingFace directory the merge can read.

verl saves the actor as FSDP shards -- ``global_step_N/actor/model_world_size_8_rank_*.pt``
plus a ``huggingface/`` subdirectory holding only the config and tokenizer. Nothing
downstream here can read that: shadow_rl/merge.py streams safetensors, OpenCompass
loads a model directory, and both need whole tensors.

verl ships the consolidation tool; which module it lives in has moved between
releases, so this tries the known entry points in order rather than pinning one.
It then checks the result is actually loadable, because a merger that exits 0
having written only a config is a failure that otherwise surfaces an hour later
in the eval.

    python verl_rl/export_hf.py --role base
    python verl_rl/export_hf.py --role instruct --step 40
    python verl_rl/export_hf.py --ckpt-dir X --out Y      # explicit paths
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys


STEP_RE = re.compile(r"global_step_(\d+)$")

# In release order, newest first. `--help` is not enough to tell whether an entry
# point exists (some print help then fail on import), so each is tried for real.
MERGER_COMMANDS = [
    ["-m", "verl.model_merger", "merge"],
    ["-m", "verl.model_merger"],
    ["-m", "verl.scripts.model_merger", "merge"],
    ["-m", "verl.scripts.model_merger"],
]


def find_latest_step(ckpt_dir: str) -> str:
    """Newest global_step_N under ckpt_dir, by step number not mtime."""
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"no checkpoint directory at {ckpt_dir}")
    steps = []
    for name in os.listdir(ckpt_dir):
        m = STEP_RE.match(name)
        if m and os.path.isdir(os.path.join(ckpt_dir, name)):
            steps.append((int(m.group(1)), name))
    if not steps:
        sys.exit(f"no global_step_* directories under {ckpt_dir}. "
                 "Did training save anything? (trainer.save_freq)")
    steps.sort()
    return steps[-1][1]


def actor_dir(ckpt_dir: str, step: str) -> str:
    path = os.path.join(ckpt_dir, step, "actor")
    if not os.path.isdir(path):
        sys.exit(f"{path} does not exist -- expected verl's actor checkpoint there")
    return path


def run_merger(local_dir: str, target_dir: str) -> bool:
    """Consolidate the FSDP shards. True if some entry point worked."""
    for cmd in MERGER_COMMANDS:
        argv = [sys.executable, *cmd,
                "--backend", "fsdp",
                "--local_dir", local_dir,
                "--target_dir", target_dir]
        print(f"  trying: {' '.join(argv[1:])}")
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode == 0:
            return True
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        # A module that does not exist in this verl is expected; anything else
        # is worth showing, because it is probably the real error.
        if tail and not any("No module named" in line for line in tail[-3:]):
            print("    failed:")
            for line in tail[-8:]:
                print(f"      {line}")
    return False


def copy_aux_files(src_hf_dir: str, out_dir: str) -> None:
    """Config and tokenizer from verl's own huggingface/ copy.

    The merger writes weights; some versions do not bring the tokenizer along,
    and a model directory without one loads in transformers and fails in vLLM.
    """
    if not os.path.isdir(src_hf_dir):
        return
    for name in sorted(os.listdir(src_hf_dir)):
        if re.search(r"\.(safetensors|bin|pt|pth)$", name):
            continue  # never overwrite the consolidated weights
        src, dst = os.path.join(src_hf_dir, name), os.path.join(out_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  copied {name}")


def verify(out_dir: str) -> None:
    """A directory that is not a loadable model is not a successful export."""
    if not os.path.isdir(out_dir):
        sys.exit(f"export produced nothing at {out_dir}")
    files = os.listdir(out_dir)
    weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
    if not weights:
        sys.exit(f"{out_dir} has no weight files -- the merger did not consolidate "
                 f"the shards. Contents: {sorted(files)}")
    if "config.json" not in files:
        sys.exit(f"{out_dir} has no config.json")
    if not any(f.startswith("tokenizer") or f == "vocab.json" for f in files):
        print("  warn: no tokenizer files; vLLM will not serve this directory. "
              "Copy them from the source model.")
    total_gb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in weights) / 1e9
    print(f"  {len(weights)} weight file(s), {total_gb:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["base", "instruct"],
                    help="read the paths for this role out of config.sh")
    ap.add_argument("--ckpt-dir", help="verl checkpoint root (overrides --role)")
    ap.add_argument("--out", help="output directory (overrides --role)")
    ap.add_argument("--step", default="latest",
                    help="global step to export, or 'latest' (default)")
    ap.add_argument("--force", action="store_true",
                    help="re-export even if the output already looks complete")
    args = ap.parse_args()

    if args.ckpt_dir and args.out:
        ckpt_dir, out_dir = args.ckpt_dir, args.out
    elif args.role:
        # Ask config.sh rather than duplicating its path layout here.
        here = os.path.dirname(os.path.abspath(__file__))
        query = (f'source {os.path.join(here, "config.sh")} >/dev/null 2>&1; '
                 f'echo "$CKPT_DIR/{args.role}"; echo "$EXPORT_DIR/{args.role}"')
        out = subprocess.run(["bash", "-c", query], capture_output=True, text=True)
        lines = [line for line in out.stdout.strip().splitlines() if line]
        if len(lines) != 2:
            sys.exit(f"could not read paths from config.sh: {out.stderr.strip()}")
        ckpt_dir, out_dir = lines
    else:
        sys.exit("need --role, or both --ckpt-dir and --out")

    step = args.step if args.step != "latest" else find_latest_step(ckpt_dir)
    if not step.startswith("global_step_"):
        step = f"global_step_{step}"
    src = actor_dir(ckpt_dir, step)

    print(f"exporting {src}")
    print(f"       -> {out_dir}")

    if os.path.isdir(out_dir) and not args.force:
        existing = [f for f in os.listdir(out_dir) if f.endswith((".safetensors", ".bin"))]
        if existing:
            print("  already exported (use --force to redo)")
            verify(out_dir)
            return 0

    os.makedirs(out_dir, exist_ok=True)
    if not run_merger(src, out_dir):
        sys.exit(
            "no verl checkpoint merger worked.\n"
            "  Check `python -c 'import verl; print(verl.__version__)'` and look for\n"
            "  the merger in your install:\n"
            "    python -c \"import verl,os;print(os.path.dirname(verl.__file__))\"\n"
            "  then run it by hand with --backend fsdp --local_dir/--target_dir."
        )

    copy_aux_files(os.path.join(ckpt_dir, step, "actor", "huggingface"), out_dir)
    verify(out_dir)

    with open(os.path.join(out_dir, "verl_export.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_checkpoint": src, "step": step}, fh, indent=2)
    print(f"  exported step {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
