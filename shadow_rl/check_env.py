#!/usr/bin/env python3
"""Validate the environment before a long run, and say exactly how to fix it.

Two failures this exists to catch.

1. pip installs torch from the CUDA-specific PyTorch index but torchvision from
   default PyPI, which may be built against a different CUDA major version.
   Nothing complains at install time. transformers imports torchvision deep
   inside `image_utils`, so the first symptom is an unrelated-looking
   `Could not import module 'Qwen2ForCausalLM'` -- after the merge has already
   run.

2. torch imports, reports the GPU by name, and then dies on the first matmul
   with `no kernel image is available for execution on the device`, because the
   wheel carries no cubin for this compute capability. This is what the move
   from H100 (sm_90) to B300 (sm_103) hits: every torch below 2.9 predates
   sm_103, and even a Blackwell wheel built arch-conditionally for a B200
   (sm_100a) has no kernels a B300 can run.

    python shadow_rl/check_env.py                     # core checks
    python shadow_rl/check_env.py --arch-only         # just torch vs this GPU
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


# ---------------------------------------------------------------------------- #
# compute capability
# ---------------------------------------------------------------------------- #
# Which wheel index and which minimum torch a GPU generation needs. This mirrors
# scripts/gpu_profile.sh -- the bash side is what the setup scripts read, this
# side keeps check_env.py runnable on its own. tests/env/test_gpu_profile.py
# asserts the two tables agree.
STACK_BY_SM = {
    103: ("https://download.pytorch.org/whl/cu130", "2.9.0"),   # B300 / GB300
    100: ("https://download.pytorch.org/whl/cu128", "2.7.0"),   # B200 / GB200
    120: ("https://download.pytorch.org/whl/cu128", "2.7.0"),   # RTX Blackwell
}
LEGACY_STACK = ("https://download.pytorch.org/whl/cu126", "2.6.0")  # Hopper and older


def stack_for_sm(sm: int):
    """(wheel index, minimum torch) for a device of compute capability `sm`."""
    return STACK_BY_SM.get(sm, LEGACY_STACK)


def parse_arch_tag(tag: str):
    """'sm_100a' -> (100, 'a'), 'compute_90' -> (90, ''), junk -> None.

    torch.cuda.get_arch_list() returns a mix of both spellings: cubins as
    sm_NNN[a|f], embedded PTX as compute_NNN.
    """
    tag = str(tag).strip()
    for prefix in ("sm_", "compute_"):
        if tag.startswith(prefix):
            rest = tag[len(prefix):]
            suffix = ""
            if rest and rest[-1] in ("a", "f"):
                rest, suffix = rest[:-1], rest[-1]
            if rest.isdigit():
                return int(rest), suffix
    return None


def arch_support(arch_list, sm: int):
    """How a torch built for `arch_list` can run on a device of capability `sm`.

    Returns "cubin" (native code for this arch), "compat" (code for an earlier
    minor of the same major -- binary compatible, guaranteed forwards only),
    "ptx" (has to JIT on first use, slow but works), or None (will raise
    "no kernel image is available for execution on the device").

    The rule that bites in the H100 -> B300 move is the arch-conditional one: a
    cubin built as sm_100a runs on sm_100 and nothing else, so a wheel that
    works on a B200 can still have nothing for a B300. Plain sm_100 code does
    run on sm_103, by CUDA's minor-version binary compatibility.
    """
    best = None
    rank = {"cubin": 3, "compat": 2, "ptx": 1}
    for tag in arch_list or ():
        parsed = parse_arch_tag(tag)
        if parsed is None:
            continue
        arch, suffix = parsed
        is_ptx = str(tag).startswith("compute_")
        if arch == sm:
            found = "ptx" if is_ptx else "cubin"
        elif suffix == "a":
            # Arch-conditional: exact match only, no forward compatibility.
            continue
        elif arch // 10 == sm // 10 and arch < sm:
            # Same major, lower minor: cubins are forwards compatible, and PTX
            # JITs. ('f' family-conditional code behaves like plain code here.)
            found = "ptx" if is_ptx else "compat"
        else:
            continue
        if best is None or rank[found] > rank[best]:
            best = found
    return best


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
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        ok(f"CUDA available, {torch.cuda.device_count()} device(s): "
           f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    else:
        warn("no CUDA device visible (fine for merge/similarity, not for eval)")
    return torch


def check_device_arch(torch_mod):
    """Does this torch actually carry kernels for the GPU in this machine?"""
    if not torch_mod.cuda.is_available():
        return
    major, minor = torch_mod.cuda.get_device_capability(0)
    sm = major * 10 + minor
    try:
        arch_list = torch_mod.cuda.get_arch_list()
    except Exception as exc:  # very old torch, or a CPU-only build
        warn(f"torch cannot report its arch list ({exc}); skipping the arch check")
        return

    index, min_torch = stack_for_sm(sm)
    fix = (f"pip install --force-reinstall 'torch>={min_torch}' torchvision "
           f"--index-url {index}")
    support = arch_support(arch_list, sm)

    if support == "cubin":
        ok(f"torch has sm_{sm} kernels")
    elif support == "compat":
        # Runs, but nothing arch-conditional (the FP8/NVFP4 CUTLASS paths) is
        # there, so it is worth saying out loud rather than passing silently.
        warn(f"torch has no sm_{sm} kernels but does have compatible ones "
             f"({' '.join(arch_list)}); it will run, without the sm_{sm}a "
             f"FP8/NVFP4 paths")
    elif support == "ptx":
        warn(f"torch has only PTX for sm_{sm}; every kernel JITs on first use "
             f"(minutes of stall, then fine)")
    else:
        bad(f"torch {torch_mod.__version__} was built for [{' '.join(arch_list)}] "
            f"and this GPU is sm_{sm}: kernels will fail with 'no kernel image "
            f"is available for execution on the device'", fix)


def check_device_kernels(torch_mod):
    """Run the two kernels every stage here needs, on the GPU, for real.

    The arch list is a claim; this is the test. A wheel can list an arch and
    still fail on a specific kernel, and the error only appears at the first
    launch -- historically 40 minutes into a run, after the merge.
    """
    if not torch_mod.cuda.is_available():
        return
    major, minor = torch_mod.cuda.get_device_capability(0)
    sm = major * 10 + minor
    index, min_torch = stack_for_sm(sm)
    fix = (f"pip install --force-reinstall 'torch>={min_torch}' torchvision "
           f"--index-url {index}")

    try:
        a = torch_mod.randn(64, 64, dtype=torch_mod.bfloat16, device="cuda")
        (a @ a).sum().item()
        ok("bf16 matmul runs on the GPU")
    except Exception as exc:
        bad(f"bf16 matmul fails on this GPU: {str(exc)[:160]}", fix)
        return  # nothing else is going to work either

    try:
        q = torch_mod.randn(1, 4, 8, 64, dtype=torch_mod.bfloat16, device="cuda")
        torch_mod.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch_mod.cuda.synchronize()
        ok("scaled_dot_product_attention runs on the GPU")
    except Exception as exc:
        bad(f"sdpa fails on this GPU: {str(exc)[:160]}", fix)


def check_flash_attn(torch_mod):
    """FlashAttention-2 is optional, and on Blackwell it is often not usable.

    The training scripts pass `--flash_attn`; picking fa2 when the installed
    wheel has no kernels for this arch fails at the first forward pass. flash-attn
    added sm_100 kernels in 2.8, and a wheel built for sm_100a has none for a
    B300 -- so the only honest test is a call.
    """
    if not have("flash_attn"):
        # sdpa is the fallback and is always present; on Blackwell it lands on
        # the cuDNN attention backend, which is fast.
        warn("flash_attn not installed; use --flash_attn sdpa (the default here)")
        return
    try:
        import flash_attn
    except Exception as exc:
        bad(f"flash_attn is installed but fails to import: {str(exc)[:140]}",
            "pip uninstall -y flash-attn   # then train with --flash_attn sdpa")
        return
    ver = getattr(flash_attn, "__version__", "?")
    if not torch_mod.cuda.is_available():
        warn(f"flash_attn {ver} installed; cannot test it without a GPU")
        return

    major, minor = torch_mod.cuda.get_device_capability(0)
    sm = major * 10 + minor
    try:
        from flash_attn import flash_attn_func
        q = torch_mod.randn(1, 8, 2, 64, dtype=torch_mod.bfloat16, device="cuda")
        flash_attn_func(q, q, q, causal=True)
        torch_mod.cuda.synchronize()
        ok(f"flash_attn {ver} runs on sm_{sm} (--flash_attn fa2 is safe)")
    except Exception as exc:
        # Not fatal: sdpa covers it. But it must not be reported as fine, because
        # the generated training scripts choose between the two.
        warn(f"flash_attn {ver} does not run on sm_{sm} ({str(exc)[:110]}); "
             f"train with --flash_attn sdpa")


def check_engines(torch_mod):
    """vllm / lmdeploy carry their own CUDA kernels, with their own arch lists.

    Evaluation goes through lmdeploy TurboMind by default (see the generated
    OpenCompass configs), and TurboMind ships prebuilt kernels: on a compute
    capability its wheel predates, it fails at engine start, not at import.
    """
    if not torch_mod or not torch_mod.cuda.is_available():
        return
    major, minor = torch_mod.cuda.get_device_capability(0)
    sm = major * 10 + minor
    if sm < 100:
        return  # Hopper and older: both engines have shipped kernels for years

    for name, hint in (
        ("vllm", "pip install -U 'vllm>=0.11.0'  # first releases with sm_103 kernels"),
        ("lmdeploy", "pip install -U lmdeploy  # or set the OpenCompass model type "
                     "to the vllm backend"),
    ):
        if not have(name):
            continue
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
        except Exception as exc:
            bad(f"{name} is installed but fails to import: {str(exc)[:140]}", hint)
            continue
        warn(f"{name} {ver} on sm_{sm}: prebuilt kernels for Blackwell Ultra are "
             f"recent; if the engine dies at start-up with 'no kernel image', "
             f"upgrade it ({hint.split('#')[0].strip()})")


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


def report() -> int:
    """Print the collected fixes, and return the exit status."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also check vllm and Search-R1")
    ap.add_argument("--arch-only", action="store_true",
                    help="only check torch against this GPU's compute capability "
                         "(what the setup scripts use to decide on a reinstall)")
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

    if args.arch_only:
        if torch_mod:
            guarded("device arch", check_device_arch, torch_mod)
            guarded("gpu kernels", check_device_kernels, torch_mod)
        return report()

    if torch_mod:
        guarded("device arch", check_device_arch, torch_mod)
        guarded("gpu kernels", check_device_kernels, torch_mod)
        guarded("torch companions", check_siblings, torch_mod)
        guarded("flash-attn", check_flash_attn, torch_mod)
    guarded("transformers", check_transformers, torch_mod)
    for name, pip_name in (("safetensors", "safetensors"),
                           ("huggingface_hub", "huggingface_hub")):
        guarded(name, check_simple, name, pip_name)
    guarded("inference engines", check_engines, torch_mod)

    if args.full:
        guarded("datasets", check_simple, "datasets", "datasets")
        guarded("vllm", check_simple, "vllm", "vllm")
        guarded("Search-R1", check_search_r1, args.search_r1_root)

    return report()


if __name__ == "__main__":
    sys.exit(main())
