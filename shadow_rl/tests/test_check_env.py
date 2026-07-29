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
        (128, "12", "packed int (the case that crashed)"),
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

    print("\n[6] a dependency that raises on import does not abort the run")
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

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
