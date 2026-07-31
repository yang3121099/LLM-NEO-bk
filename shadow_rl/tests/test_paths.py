#!/usr/bin/env python3
"""paths.py and paths.sh must resolve to the same directory, always.

There are two implementations because bash scripts and python entry points both
need the answer, and they are consulted independently: setup_search_r1.sh clones
to whatever paths.sh says, and check_env.py then looks wherever paths.py says. If
they ever disagree, setup reports success and the very next command reports the
checkout missing -- which is exactly the failure this file exists to prevent.

Run:  python shadow_rl/tests/test_paths.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PATHS_SH = os.path.join(REPO, "shadow_rl", "paths.sh")

sys.path.insert(0, os.path.join(REPO, "shadow_rl"))
import paths as P  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def from_sh(fn, env):
    out = subprocess.run(
        ["bash", "-c", f'source "{PATHS_SH}"; {fn}'],
        capture_output=True, text=True,
        env={**os.environ, **env, "BASH_SOURCE_OVERRIDE": ""},
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def from_py(fn, env):
    """Run paths.py in a subprocess so os.environ edits cannot leak between cases."""
    code = f"import sys; sys.path.insert(0, {os.path.join(REPO, 'shadow_rl')!r}); " \
           f"import paths; print(paths.{fn}())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, **env})
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def both(sh_fn, py_fn, env):
    return from_sh(sh_fn, env), from_py(py_fn, env)


def main():
    tmp = tempfile.mkdtemp(prefix="shadow-paths-")
    fake_home = os.path.join(tmp, "home")
    os.makedirs(os.path.join(fake_home, "Search-R1"))
    os.makedirs(os.path.join(fake_home, "verl"))
    in_tree = os.path.join(REPO, "third_party", "Search-R1")

    print("\n[1] nothing set: the in-tree path, never $HOME")
    # HOME points at a directory with no checkout in it, so the answer must be
    # the working tree regardless.
    env = {"HOME": tmp, "SEARCH_R1_ROOT": "", "VERL_ROOT": ""}
    sh, py = both("shadow_rl_search_r1_root", "search_r1_root", env)
    check("shell and python agree", sh == py, f"{sh} vs {py}")
    check("resolves in-tree", sh == in_tree, sh)
    check("not under $HOME", not sh.startswith(tmp), sh)

    print("\n[2] an explicit override wins")
    env = {"HOME": tmp, "SEARCH_R1_ROOT": "/somewhere/else"}
    sh, py = both("shadow_rl_search_r1_root", "search_r1_root", env)
    check("shell and python agree", sh == py, f"{sh} vs {py}")
    check("override honoured", sh == "/somewhere/else", sh)

    print("\n[3] an existing $HOME checkout is still used")
    # Users who cloned to ~ before this change should not have to re-clone. This
    # only applies when there is no in-tree checkout to prefer.
    if os.path.isdir(in_tree):
        print("     (skipped: an in-tree checkout exists here, which rightly wins)")
    else:
        env = {"HOME": fake_home, "SEARCH_R1_ROOT": ""}
        sh, py = both("shadow_rl_search_r1_root", "search_r1_root", env)
        check("shell and python agree", sh == py, f"{sh} vs {py}")
        check("picks up the $HOME clone",
              sh == os.path.join(fake_home, "Search-R1"), sh)

    print("\n[4] verl resolves by the same rule")
    env = {"HOME": tmp, "VERL_ROOT": ""}
    sh, py = both("shadow_rl_verl_root", "verl_root", env)
    check("shell and python agree", sh == py, f"{sh} vs {py}")
    env = {"HOME": tmp, "VERL_ROOT": "/opt/verl"}
    sh, py = both("shadow_rl_verl_root", "verl_root", env)
    check("override honoured", sh == py == "/opt/verl", f"{sh} vs {py}")

    print("\n[5] no default anywhere points into $HOME")
    # The whole point of the change: a box where $HOME is /root and unwritable
    # must still get a usable path out of every script.
    env = {"HOME": "/nonexistent-home", "SEARCH_R1_ROOT": "", "VERL_ROOT": ""}
    for sh_fn, py_fn in (("shadow_rl_search_r1_root", "search_r1_root"),
                         ("shadow_rl_verl_root", "verl_root")):
        sh, py = both(sh_fn, py_fn, env)
        check(f"{py_fn} avoids $HOME", sh == py and sh.startswith(REPO), sh)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
