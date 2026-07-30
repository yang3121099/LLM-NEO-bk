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
import importlib.util
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


def cuda_major(value) -> str | None:
    """Major CUDA version from whatever a package reports.

    torch reports a string ("12.8"); torchvision builds have been seen reporting
    an int, either packed ("128" meaning 12.8) or bare ("12"). Normalising here
    rather than assuming a string -- an AttributeError in the checker is worse
    than no checker at all.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value.split(".")[0]
    if isinstance(value, (int, float)):
        n = int(value)
        # Two int encodings are in the wild:
        #   CUDA runtime  major*1000 + minor*10   12080 -> 12.8,  13000 -> 13.0
        #   wheel tag     major*10   + minor      128   -> 12.8,  130   -> 13.0
        if n >= 1000:
            return str(n // 1000)
        if n >= 100:
            return str(n // 10)
        return str(n)
    return str(value).split(".")[0]


def torch_index(torch_cuda) -> str:
    """PyTorch wheel index matching an installed torch CUDA version."""
    if torch_cuda is None:
        return "https://download.pytorch.org/whl/cu128"
    tag = str(torch_cuda).replace(".", "")
    return f"https://download.pytorch.org/whl/cu{tag}"


# torch's C++ extensions each embed the CUDA version they were built against and
# check it at import. Any one of them can be the mismatched package, and the
# error surfaces from whichever transformers happens to import first.
SIBLINGS = ("torchvision", "torchaudio", "torchcodec")

# vllm imports torchvision unconditionally during kernel warmup
# (model_executor/warmup -> models/minimax_m3 -> torchvision.transforms), so with
# vllm installed torchvision is required, not optional. transformers, by
# contrast, degrades gracefully when any of these are missing.
REQUIRED_BY_VLLM = {"torchvision"}


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def blame_sibling(message: str):
    """Return the sibling package a CUDA-mismatch message names, if any.

    The message says e.g. "PyTorch and TorchAudio were compiled with different
    CUDA versions" -- pointing the fix at the wrong package wastes a reinstall.
    """
    low = message.lower()
    for name in SIBLINGS:
        if name in low:
            return name
    return None


def check_torch():
    try:
        import torch
    except Exception as exc:
        bad(f"torch does not import: {exc}",
            "pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return None
    ok(f"torch {torch.__version__}  (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        warn("no CUDA device visible (fine for merge/similarity, not for eval)")
        return torch

    ok(f"CUDA available, {torch.cuda.device_count()} device(s): "
       f"{torch.cuda.get_device_name(0)}")

    # The decisive check is not the CUDA version but whether this torch build
    # actually carries kernels for the installed GPU. Blackwell (B200/B300) is
    # sm_100; a cu12 wheel has no sm_100 and every kernel launch fails at
    # runtime with a confusing error rather than at import.
    try:
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        archs = torch.cuda.get_arch_list()
    except Exception as exc:
        warn(f"could not read the device capability: {exc}")
        return torch

    if sm in archs:
        ok(f"{sm} is in this torch build ({', '.join(archs[-4:])})")
    else:
        # sm_100 kernels also run on sm_103 etc. via the same major family, but
        # only if the build shipped them; PTX JIT from a lower arch is not
        # reliable for these, so treat a miss as fatal.
        family = [a for a in archs if a.startswith(f"sm_{cap[0]}")]
        detail = f"build has {', '.join(archs)}"
        if family:
            warn(f"{sm} not in the arch list but {', '.join(family)} is — "
                 f"same family, likely fine ({detail})")
        else:
            bad(f"this torch has no kernels for {sm} "
                f"({torch.cuda.get_device_name(0)}); {detail}",
                "pip install --force-reinstall torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu130")
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
    tv_cuda = getattr(getattr(torchvision, "version", None), "cuda", None)
    t_major, tv_major = cuda_major(t_cuda), cuda_major(tv_cuda)
    if t_major and tv_major and t_major != tv_major:
        bad(f"torch CUDA {t_cuda} vs torchvision CUDA {tv_cuda} (major mismatch)",
            f"pip install --force-reinstall torchvision --index-url {torch_index(t_cuda)}")
    elif t_major and tv_major:
        ok(f"torchvision {torchvision.__version__} (CUDA {tv_cuda}) matches torch")
    else:
        # It imported cleanly, which is the property that actually matters.
        warn(f"torchvision {getattr(torchvision, '__version__', '?')} imports, "
             f"but CUDA version is unreported (torch={t_cuda!r}, torchvision={tv_cuda!r})")


def check_siblings(torch_mod):
    """Import each torch companion package and compare its CUDA major."""
    t_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    t_major = cuda_major(t_cuda)
    fix = f"pip install --force-reinstall {{pkg}} --index-url {torch_index(t_cuda)}"

    vllm_present = have("vllm")

    for name in SIBLINGS:
        try:
            mod = __import__(name)
        except ModuleNotFoundError:
            if vllm_present and name in REQUIRED_BY_VLLM:
                bad(f"{name} is not installed, but vllm requires it "
                    f"(kernel warmup imports {name}.transforms)",
                    f"pip install {name} --index-url {torch_index(t_cuda)}")
            else:
                warn(f"{name} not installed (fine; nothing here needs it)")
            continue
        except Exception as exc:
            culprit = blame_sibling(str(exc)) or name
            bad(f"{name} is installed but fails to import: {str(exc)[:140]}",
                fix.format(pkg=culprit))
            continue
        m_cuda = getattr(getattr(mod, "version", None), "cuda", None)
        m_major = cuda_major(m_cuda)
        if t_major and m_major and t_major != m_major:
            bad(f"torch CUDA {t_cuda} vs {name} CUDA {m_cuda} (major mismatch)",
                fix.format(pkg=name))
        elif m_major:
            ok(f"{name} {getattr(mod, '__version__', '?')} (CUDA {m_cuda}) matches torch")
        else:
            ok(f"{name} {getattr(mod, '__version__', '?')} imports")


def check_transformers(torch_mod):
    """Import the real model class -- the lazy loader hides breakage until then."""
    try:
        import transformers
    except Exception as exc:
        bad(f"transformers does not import: {exc}", "pip install -U transformers")
        return
    ok(f"transformers {transformers.__version__}")

    t_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None) if torch_mod else None

    def fix_for(message: str) -> str:
        pkg = blame_sibling(message) or "torchvision"
        return f"pip install --force-reinstall {pkg} --index-url {torch_index(t_cuda)}"

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: F401
        ok("Qwen2ForCausalLM resolves")
    except Exception as exc:
        bad(f"Qwen2ForCausalLM will not load: {type(exc).__name__}: {str(exc)[:160]}",
            fix_for(str(exc)))

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        ok("AutoModelForCausalLM / AutoTokenizer import")
    except Exception as exc:
        bad(f"transformers auto classes broken: {exc}", fix_for(str(exc)))


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

    def guarded(label, fn, *a):
        """Run one check; a bug in the check must not abort the whole run."""
        try:
            return fn(*a)
        except Exception as exc:
            warn(f"{label} check itself errored ({type(exc).__name__}: {exc}); skipping it")
            return None

    print("environment check")
    torch_mod = guarded("torch", check_torch)
    if torch_mod:
        guarded("torch companions", check_siblings, torch_mod)
    guarded("transformers", check_transformers, torch_mod)
    for name, pip_name in (("safetensors", "safetensors"),
                           ("huggingface_hub", "huggingface_hub")):
        guarded(name, check_simple, name, pip_name)

    if args.full:
        guarded("datasets", check_simple, "datasets", "datasets")
        guarded("vllm", check_simple, "vllm", "vllm")
        guarded("Search-R1", check_search_r1, args.search_r1_root)

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
