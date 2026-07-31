#!/usr/bin/env python3
"""Say why the eval jobs failed, from the logs they already wrote.

After a multi-GPU run falls over there is one log per (pair, role, dataset) in
shadow_rl/logs/, each thousands of lines of vLLM chatter. Reading them one at a
time to discover they all say the same thing is a waste of an evening.

    python shadow_rl/diagnose.py                    # group every failing log
    python shadow_rl/diagnose.py --full             # print each cause in full
    python shadow_rl/diagnose.py --logs some/dir

Exit status is 1 if anything failed, so it can gate a script.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval_parallel import extract_error, failure_signature  # noqa: E402

# Causes worth naming, because the fix is not obvious from the exception alone.
KNOWN = [
    ("No module named 'vllm'",
     "vllm is not installed in the interpreter running the workers.\n"
     "       pip install vllm     (and check you are in the right venv)"),
    ("out of memory",
     "The GPU ran out of memory. With one worker per GPU the usual causes are a\n"
     "       stale process still holding the card (nvidia-smi) or too high a\n"
     "       --gpu-memory-utilization. Try --gpu-memory-utilization 0.7."),
    ("Connection refused",
     "Nothing is listening on the retriever port. The SearchR1-* pairs need it:\n"
     "       ./shadow_rl/launch_bm25_retriever.sh, or run_all.sh --auto-retriever"),
    ("Max retries exceeded",
     "The retriever stopped answering mid-run -- often it died under the load of\n"
     "       N workers querying at once. Check its own log; consider fewer --jobs."),
    ("EngineDeadError",
     "The vLLM engine process died. Look further up the log for the real error;\n"
     "       it is usually a CUDA/driver mismatch or an out-of-memory kill."),
    ("no kernels image is available",
     "torch has no kernels for this GPU -- a cu12 wheel on a Blackwell card.\n"
     "       Reinstall from the cu130 index; python shadow_rl/check_env.py --full"),
    ("Could not import module",
     "A torch companion (torchvision/torchaudio/torchcodec) was built against a\n"
     "       different CUDA than torch. python shadow_rl/check_env.py names which."),
    ("log is empty",
     "The worker was killed before writing anything, which almost always means\n"
     "       the host ran out of RAM. Lower --jobs, or raise --stagger so the\n"
     "       workers do not all load weights at the same moment."),
    ("CUDA_VISIBLE_DEVICES",
     "The worker could not see the GPU it was pinned to. Check that --jobs x --tp\n"
     "       does not exceed the number of visible GPUs."),
]


def advice(signature: str, body: str) -> str:
    haystack = f"{signature}\n{body}"
    for needle, text in KNOWN:
        if needle.lower() in haystack.lower():
            return text
    return ""


def looks_failed(lines: list) -> bool:
    """A finished job ends with a written row; a failed one ends in an error."""
    if not lines:
        return True
    joined = " ".join(lines).lower()
    return any(k in joined for k in
               ("traceback", "error", "killed", "out of memory", "log is empty"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "logs"))
    ap.add_argument("--full", action="store_true",
                    help="print every failing log's error, not one per cause")
    ap.add_argument("--lines", type=int, default=12)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs, "*.log")))
    if not paths:
        print(f"no logs under {args.logs}")
        return 0

    grouped: dict = {}
    clean = 0
    for path in paths:
        lines = extract_error(path, max_lines=args.lines)
        if not looks_failed(lines):
            clean += 1
            continue
        grouped.setdefault(failure_signature(lines), []).append((path, lines))

    print(f"{len(paths)} log(s) under {args.logs}: "
          f"{clean} with no error, {len(paths) - clean} failing, "
          f"{len(grouped)} distinct cause(s)\n")
    if not grouped:
        return 0

    for sig, entries in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        print("=" * 72)
        print(f"{len(entries)} log(s):  {sig}")
        print("=" * 72)
        show = entries if args.full else entries[:1]
        for path, lines in show:
            print(f"  {os.path.basename(path)}")
            for line in lines:
                print(f"    | {line}")
            print()
        if not args.full and len(entries) > 1:
            print(f"  ... and {len(entries) - 1} more with the same cause:")
            for path, _ in entries[1:6]:
                print(f"      {os.path.basename(path)}")
            if len(entries) > 6:
                print(f"      (+{len(entries) - 6} more)")
            print()
        hint = advice(sig, "\n".join(l for _, ls in show for l in ls))
        if hint:
            print(f"  fix:  {hint}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
