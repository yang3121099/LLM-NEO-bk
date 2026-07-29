#!/usr/bin/env python3
"""Tests for run_all.sh's reuse logic.

Exercises the bash helpers directly against real directory layouts: a complete
merge, an interrupted one, a stale smoke marker. The case worth pinning is the
interrupted merge -- a directory with a config.json but a missing shard must be
redone, not silently reused.

Run:  python shadow_rl/tests/test_resume.py
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
RUN_ALL = os.path.join(ROOT, "run_all.sh")

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def extract_helpers():
    """Pull just the helper function definitions out of run_all.sh.

    Cut at the pair-resolution marker: everything above it is definitions,
    everything below starts doing work (and would fail without a real argv).
    """
    src = open(RUN_ALL).read()
    marker = "# ---- resolve the pair selection"
    assert marker in src, "run_all.sh layout changed; update this test"
    prefix = src.split(marker)[0]
    # Drop the argument parsing loop, which consumes "$@" we do not have.
    start = prefix.index("has_stage()")
    return prefix[start:]


HELPERS = None


def call_helper(fn, arg):
    """Source run_all.sh's helpers in isolation and invoke one."""
    global HELPERS
    if HELPERS is None:
        HELPERS = extract_helpers()
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(HELPERS)
        path = fh.name
    try:
        out = subprocess.run(
            ["bash", "-c", f'cd "{REPO}" && source "{path}" && '
                           f'{fn} "{arg}" && echo YES || echo NO'],
            capture_output=True, text=True,
        )
        tail = out.stdout.strip().splitlines()
        assert tail and tail[-1] in ("YES", "NO"), (
            f"helper {fn} produced no verdict.\nstdout: {out.stdout}\nstderr: {out.stderr}")
        return tail[-1] == "YES"
    finally:
        os.unlink(path)


def make_merged(path, shards=("model-00001-of-00002.safetensors",
                             "model-00002-of-00002.safetensors"),
                write_all=True, stats=True):
    os.makedirs(path, exist_ok=True)
    weight_map = {f"w{i}": s for i, s in enumerate(shards)}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": weight_map}, fh)
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump({"model_type": "qwen2"}, fh)
    for s in (shards if write_all else shards[:1]):
        open(os.path.join(path, s), "wb").write(b"\0")
    if stats:
        with open(os.path.join(path, "shadow_merge_stats.json"), "w") as fh:
            json.dump({"sigma": 0.01}, fh)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("\n[1] merged_complete()")

        good = os.path.join(tmp, "good")
        make_merged(good)
        check("complete merge is reused", call_helper("merged_complete", good))

        partial = os.path.join(tmp, "partial")
        make_merged(partial, write_all=False)
        check("interrupted merge (missing shard) is redone",
              not call_helper("merged_complete", partial))

        nostats = os.path.join(tmp, "nostats")
        make_merged(nostats, stats=False)
        check("merge without the final stats file is redone",
              not call_helper("merged_complete", nostats))

        check("absent directory is redone",
              not call_helper("merged_complete", os.path.join(tmp, "nope")))

        # A config.json alone used to be enough -- that was the bug.
        cfgonly = os.path.join(tmp, "cfgonly")
        os.makedirs(cfgonly)
        with open(os.path.join(cfgonly, "config.json"), "w") as fh:
            json.dump({"model_type": "qwen2"}, fh)
        check("config.json alone is not treated as a merged model",
              not call_helper("merged_complete", cfgonly))

        print("\n[2] smoke_passed()")
        check("no marker -> smoke test runs", not call_helper("smoke_passed", good))

        open(os.path.join(good, ".smoke_ok"), "w").close()
        check("marker newer than the merge -> skipped", call_helper("smoke_passed", good))

        # Re-merging must invalidate a previous pass.
        os.utime(os.path.join(good, "shadow_merge_stats.json"), None)
        check("marker older than a re-merge -> smoke test runs again",
              not call_helper("smoke_passed", good))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
