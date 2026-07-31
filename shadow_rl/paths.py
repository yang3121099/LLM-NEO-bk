#!/usr/bin/env python3
"""Shared default locations for the python entry points.

shadow_rl/paths.sh is the shell twin; the two must agree, and
shadow_rl/tests/test_paths.py checks that they do.

Everything defaults into the working tree rather than $HOME. On a container or
a shared box $HOME is often /root -- small, or not writable at all -- and a
Search-R1 checkout landing there fails in ways that surface much later as
something unrelated. The working tree is the one directory we know is writable,
because the repo is already in it.

Anything already cloned under $HOME still works: an existing checkout there is
picked up rather than ignored, so upgrading does not orphan one.

    python shadow_rl/paths.py        # print the resolved paths
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(env_var: str, name: str) -> str:
    """Explicit override, then in-tree, then $HOME, then in-tree as the target."""
    override = os.environ.get(env_var)
    if override:
        return override
    in_tree = os.path.join(REPO_ROOT, "third_party", name)
    if os.path.isdir(in_tree):
        return in_tree
    home = os.path.expanduser(os.path.join("~", name))
    if os.path.isdir(home):
        return home
    return in_tree


def search_r1_root() -> str:
    return _resolve("SEARCH_R1_ROOT", "Search-R1")


def verl_root() -> str:
    return _resolve("VERL_ROOT", "verl")


if __name__ == "__main__":
    print(f"repo root      : {REPO_ROOT}")
    print(f"SEARCH_R1_ROOT : {search_r1_root()}")
    print(f"VERL_ROOT      : {verl_root()}")
