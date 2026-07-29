#!/usr/bin/env python3
"""Validate the environment before a long run, and say exactly how to fix it.

The failure this exists to catch: pip installs torch from the CUDA-specific
PyTorch index but torchvision from default PyPI, which may be built against a
different CUDA major version. Nothing complains at install time. transformers
imports torchvision deep inside `image_utils`, so the first symptom is an
unrelated-looking `Could not import module 'Qwen2ForCausalLM'` -- after the merge
has already run.

    python shadow_rl/check_env.py                     # core checks
    python shadow_rl/check_env.py --full              # also vllm + Search-R1
    python shadow_rl/check_env.py --search-r1-root ~/Search-R1 --full
"""

from __future__ import annotations

import argparse
import os
import sys

RESET, RED, GREEN, YELLOW = "\033[0m", "\033[1;31m", "\033[1;32m", "\033[1;33m"

problems: list = []       # (what failed, how to fix)


def ok(msg):
    print(f"  {GREEN}ok{RESET}    {msg}")


def bad(msg, fix):
    print(f"  {RED}FAIL{RESET}  {msg}")
    problems.append((msg, fix))


def warn(msg):
    print(f"  {YELLOW}warn{RESET}  {msg}")


def torch_index(torch_cuda: str) -> str:
    """PyTorch wheel index matching an installed torch CUDA version."""
    return f"https://download.pytorch.org/whl/cu{torch_cuda.replace('.', '')}"


def check_torch():
    try:
        import torch
    except Exception as exc:
        bad(f"torch does not import: {exc}",
            "pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return None
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        ok(f"CUDA available, {torch.cuda.device_count()} device(s): "
           f"{torch.cuda.get_device_name(0)}")
    else:
        warn("no CUDA device visible (fine for merge/similarity, not for eval)")
    return torch


def check_torchvision(torch_mod):
    """The actual bug: torchvision built against a different CUDA major."""
    t_cuda = getattr(torch_mod, "version", None) and torch_mod.version.cuda
    try:
        import torchvision
    except ModuleNotFoundError:
        # Not installed at all is fine: transformers degrades gracefully without
        # it. Only a *present but mismatched* torchvision breaks model loading.
        warn("torchvision not installed (fine; transformers works without it)")
        return
    except Exception as exc:
        # Importing raised -- the classic case is the CUDA major-version guard
        # in torchvision.extension, which is exactly what breaks transformers.
        bad(f"torchvision is installed but fails to import: {str(exc)[:160]}",
            f"pip install --force-reinstall torchvision --index-url {torch_index(t_cuda)}")
        return
    tv_cuda = getattr(torchvision, "version", None) and getattr(
        torchvision.version, "cuda", None)
    if t_cuda and tv_cuda and t_cuda.split(".")[0] != tv_cuda.split(".")[0]:
        bad(f"torch CUDA {t_cuda} vs torchvision CUDA {tv_cuda} (major mismatch)",
            f"pip install --force-reinstall torchvision --index-url {torch_index(t_cuda)}")
    else:
        ok(f"torchvision {torchvision.__version__} (CUDA {tv_cuda}) matches torch")


def check_transformers(torch_mod):
    """Import the real model class -- the lazy loader hides breakage until then."""
    try:
        import transformers
    except Exception as exc:
        bad(f"transformers does not import: {exc}", "pip install -U transformers")
        return
    ok(f"transformers {transformers.__version__}")

    t_cuda = getattr(torch_mod, "version", None) and torch_mod.version.cuda if torch_mod else None
    fix = (f"pip install --force-reinstall torchvision --index-url {torch_index(t_cuda)}"
           if t_cuda else "reinstall torchvision to match your torch CUDA build")
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: F401
        ok("Qwen2ForCausalLM resolves")
    except Exception as exc:
        bad(f"Qwen2ForCausalLM will not load: {type(exc).__name__}: {str(exc)[:160]}", fix)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        ok("AutoModelForCausalLM / AutoTokenizer import")
    except Exception as exc:
        bad(f"transformers auto classes broken: {exc}", fix)


def check_simple(name, pip_name=None, label=None):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{label or name} {ver}")
        return True
    except Exception as exc:
        bad(f"{label or name} does not import: {str(exc)[:120]}",
            f"pip install {pip_name or name}")
        return False


def check_search_r1(root):
    path = os.path.join(root, "verl", "utils", "reward_score", "qa_em.py")
    if os.path.exists(path):
        ok(f"Search-R1 EM scorer at {path}")
    else:
        bad(f"Search-R1 not found at {root}",
            f"git clone https://github.com/PeterGriffinJin/Search-R1.git {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also check vllm and Search-R1")
    ap.add_argument("--search-r1-root", default=os.environ.get("SEARCH_R1_ROOT",
                                                              os.path.expanduser("~/Search-R1")))
    args = ap.parse_args()

    print("environment check")
    torch_mod = check_torch()
    if torch_mod:
        check_torchvision(torch_mod)
    check_transformers(torch_mod)
    for name, pip_name in (("safetensors", "safetensors"),
                           ("huggingface_hub", "huggingface_hub")):
        check_simple(name, pip_name)

    if args.full:
        check_simple("datasets", "datasets")
        check_simple("vllm", "vllm")
        check_search_r1(args.search_r1_root)

    print()
    if not problems:
        print(f"{GREEN}environment looks good{RESET}")
        return 0

    print(f"{RED}{len(problems)} problem(s) found.{RESET} Suggested fixes, in order:\n")
    seen = set()
    for _, fix in problems:
        if fix not in seen:
            seen.add(fix)
            print(f"  {fix}")
    print("\nThen re-run: python shadow_rl/check_env.py --full")
    return 1


if __name__ == "__main__":
    sys.exit(main())
