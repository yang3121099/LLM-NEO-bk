#!/usr/bin/env python3
"""Tests for the environment checker.

The checker runs during preflight, so a bug in it blocks the whole pipeline --
which is exactly what happened when torchvision reported its CUDA version as an
int and the checker assumed a string. These pin the version parsing and the
guarantee that no individual check can abort the run.

Run:  python shadow_rl/tests/test_check_env.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main():
    import check_env as ce

    print("\n[1] cuda_major() accepts whatever a package reports")
    cases = [
        ("12.8", "12", "torch-style string"),
        ("13.0", "13", "string, other major"),
        (128, "12", "wheel-tag int"),
        (12080, "12", "CUDA runtime int (12.8) -- was misread as 1208"),
        (13000, "13", "CUDA runtime int (13.0)"),
        (11080, "11", "CUDA runtime int (11.8)"),
        (130, "13", "packed int, other major"),
        (12, "12", "bare int major"),
        (12.8, "12", "float"),
        ("12", "12", "string without a dot"),
        (None, None, "None"),
        ("", None, "empty string"),
    ]
    for value, want, label in cases:
        got = ce.cuda_major(value)
        check(f"cuda_major({value!r}) -> {want!r}  [{label}]", got == want, f"got {got!r}")

    print("\n[2] mismatch detection uses the parsed major")
    check("12.8 vs 130 is a mismatch",
          ce.cuda_major("12.8") != ce.cuda_major(130))
    check("12.8 vs 128 is not a mismatch",
          ce.cuda_major("12.8") == ce.cuda_major(128))
    check("12.8 vs 12080 is not a mismatch (the false positive)",
          ce.cuda_major("12.8") == ce.cuda_major(12080))
    check("12.8 vs 13000 is a mismatch",
          ce.cuda_major("12.8") != ce.cuda_major(13000))
    check("12.8 vs 12.6 is not a major mismatch",
          ce.cuda_major("12.8") == ce.cuda_major("12.6"))

    print("\n[3] torch_index() builds the right wheel index")
    check("12.8 -> cu128", ce.torch_index("12.8").endswith("/cu128"))
    check("13.0 -> cu130", ce.torch_index("13.0").endswith("/cu130"))
    check("None falls back", ce.torch_index(None).startswith("https://"))

    print("\n[4] a broken torchvision is reported, not raised")
    stub = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub, "torchvision"))
    with open(os.path.join(stub, "torchvision", "__init__.py"), "w") as fh:
        fh.write('raise RuntimeError("Detected that PyTorch and torchvision were '
                 'compiled with different CUDA major versions. PyTorch has CUDA '
                 'Version=12.8 and torchvision has CUDA Version=13.0.")\n')
    env = dict(os.environ, PYTHONPATH=stub)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    check("exits non-zero", proc.returncode != 0)
    check("no traceback", "Traceback" not in out)
    check("names torchvision", "torchvision" in out)
    check("suggests the matching index", "download.pytorch.org/whl/cu" in out)

    print("\n[5] a torchvision reporting an int does not crash the checker")
    stub2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub2, "torchvision"))
    with open(os.path.join(stub2, "torchvision", "__init__.py"), "w") as fh:
        fh.write("__version__ = '0.20.0'\n"
                 "import types as _t\n"
                 "version = _t.SimpleNamespace(cuda=128)\n")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=stub2))
    out = proc.stdout + proc.stderr
    check("no AttributeError", "AttributeError" not in out)
    check("no traceback", "Traceback" not in out)

    print("\n[6] the fix names the package the error actually blames")
    check("TorchAudio message -> torchaudio",
          ce.blame_sibling("PyTorch and TorchAudio were compiled with different "
                           "CUDA versions. PyTorch has CUDA version 12.8 whereas "
                           "TorchAudio has CUDA version 13.0.") == "torchaudio")
    check("torchvision message -> torchvision",
          ce.blame_sibling("PyTorch and torchvision were compiled with different "
                           "CUDA major versions.") == "torchvision")
    check("unrelated message -> None", ce.blame_sibling("some other error") is None)

    stub4 = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub4, "torchaudio"))
    with open(os.path.join(stub4, "torchaudio", "__init__.py"), "w") as fh:
        fh.write('raise RuntimeError("Detected that PyTorch and TorchAudio were '
                 'compiled with different CUDA versions. PyTorch has CUDA version '
                 '12.8 whereas TorchAudio has CUDA version 13.0.")\n')
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=stub4))
    out = proc.stdout + proc.stderr
    check("broken torchaudio is detected", "torchaudio" in out)
    check("fix targets torchaudio, not torchvision",
          "--force-reinstall torchaudio" in out)
    check("no traceback", "Traceback" not in out)

    print("\n[7] torchvision is required when vllm is installed")
    # vllm's kernel warmup imports torchvision.transforms unconditionally, so
    # "absent is fine" holds for transformers but not for vllm.
    stub_vllm = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub_vllm, "vllm"))
    with open(os.path.join(stub_vllm, "vllm", "__init__.py"), "w") as fh:
        fh.write("__version__ = '0.26.0'\n")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=stub_vllm))
    out = proc.stdout + proc.stderr
    if ce.have("torchvision"):
        check("torchvision present -> no complaint", "vllm requires it" not in out)
    else:
        check("missing torchvision + vllm is fatal", "vllm requires it" in out)
        check("suggests installing torchvision", "pip install torchvision" in out)
        check("exits non-zero", proc.returncode != 0)

    # Without vllm, an absent torchvision is only a warning.
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True)
    out_novllm = proc.stdout + proc.stderr
    if not ce.have("torchvision") and not ce.have("vllm"):
        check("missing torchvision without vllm is only a warning",
              "vllm requires it" not in out_novllm)

    print("\n[8] a dependency that raises on import does not abort the run")
    stub3 = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub3, "safetensors"))
    with open(os.path.join(stub3, "safetensors", "__init__.py"), "w") as fh:
        fh.write("raise ImportError('deliberately broken')\n")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "check_env.py")],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=stub3))
    out = proc.stdout + proc.stderr
    check("broken dependency reported without traceback", "Traceback" not in out)
    check("still exits non-zero", proc.returncode != 0)

    print("\n[9] compute capability: which kernels can run on which device")
    # The H100 -> B300 move: torch 2.6.0's arch list stops at sm_90, so every
    # kernel launch on an sm_103 device fails. And a Blackwell wheel built
    # arch-conditionally for a B200 (sm_100a) is no help either -- `a` code is
    # arch-exact, which is the part that surprises people.
    hopper_wheel = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    arch_cases = [
        (["sm_103"], 103, "cubin", "native kernels"),
        (["sm_103a"], 103, "cubin", "arch-conditional, right arch"),
        (["sm_100a"], 103, None, "arch-conditional for a B200 -- does NOT run on B300"),
        (["sm_100"], 103, "compat", "plain sm_100 is minor-version compatible"),
        (["sm_103"], 100, None, "no backwards compatibility to an earlier minor"),
        (["compute_100"], 103, "ptx", "PTX only -- JITs, works"),
        (["compute_100", "sm_103"], 103, "cubin", "a cubin beats PTX"),
        (["sm_120"], 103, None, "different family"),
        (hopper_wheel, 103, None, "a Hopper-era torch on a B300"),
        (hopper_wheel, 90, "cubin", "the same torch on the H100 it was built for"),
        ([], 103, None, "empty arch list"),
        (["not_an_arch", "sm_", "compute_x"], 103, None, "junk tags"),
    ]
    for arch_list, sm, want, label in arch_cases:
        got = ce.arch_support(arch_list, sm)
        check(f"{arch_list} on sm_{sm} -> {want!r}  [{label}]", got == want, f"got {got!r}")

    check("B300 is told to install a torch that has sm_103",
          ce.stack_for_sm(103) == ("https://download.pytorch.org/whl/cu130", "2.9.0"))
    check("Hopper keeps the stack its results were produced with",
          ce.stack_for_sm(90) == ("https://download.pytorch.org/whl/cu126", "2.6.0"))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
