#!/usr/bin/env python3
"""Probe HuggingFace for candidate checkpoint pairs and record the real ones.

Search-R1's v0.3 training scripts cover Qwen 3B/7B/14B and DeepSeek distills;
Llama-3.2-3B and Llama-3.1-8B appear in the v0.1 scripts. A training script is
not a release, though, and grafting needs *both* sides of a pair published --
so the ids in pairs.CANDIDATES are inferred from the naming convention and have
to be checked rather than assumed.

This asks the Hub which of them exist and writes the confirmed pairs to
verified_pairs.json, which pairs.py merges into the manifest. Nothing that 404s
ever reaches a run.

    python shadow_rl/discover.py                 # report only
    python shadow_rl/discover.py --write         # also record what exists
    python shadow_rl/discover.py --write --extra-versions v0.4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs import (  # noqa: E402
    CANDIDATES, HF_PREFIX, ORIGINALS, PAIRS, PAIRS_BY_ID, Pair,
)

VERIFIED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verified_pairs.json")


def exists(repo_id: str, token=None) -> bool:
    """True if the repo is visible to us on the Hub."""
    from huggingface_hub import model_info
    from huggingface_hub.utils import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
    )
    try:
        model_info(repo_id, token=token)
        return True
    except (RepositoryNotFoundError, EntryNotFoundError):
        return False
    except GatedRepoError:
        # It exists; we just have not accepted the licence. Worth knowing.
        print(f"    [gated] {repo_id} — accept the licence on its model page")
        return True
    except Exception as exc:
        print(f"    [?] {repo_id}: {type(exc).__name__}: {str(exc)[:80]}")
        return False


def synthesise(versions: List[str]) -> List[Pair]:
    """Extra candidates for versions not covered by the hardcoded list."""
    out = []
    for version in versions:
        for size in ("3b", "7b", "14b", "llama3.2-3b", "llama3.1-8b"):
            slug = size if size.startswith("llama") else f"qwen2.5-{size}"
            for algo in ("grpo", "ppo"):
                out.append(Pair(
                    version, size, algo, True,
                    f"SearchR1-nq_hotpotqa_train-{slug}-em-{algo}-{version}",
                    f"SearchR1-nq_hotpotqa_train-{slug}-it-em-{algo}-{version}",
                ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help=f"record confirmed pairs to {os.path.basename(VERIFIED)}")
    ap.add_argument("--extra-versions", default="",
                    help="comma list of extra version tags to probe, e.g. v0.4")
    ap.add_argument("--also-originals", action="store_true",
                    help="also check the base/instruct originals are reachable")
    ap.add_argument("--token", default=None, help="HF token for gated repos")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("[fail] huggingface_hub not installed. pip install huggingface_hub")

    candidates = list(CANDIDATES)
    extra = [v.strip() for v in args.extra_versions.split(",") if v.strip()]
    if extra:
        seen = {p.pair_id for p in candidates} | set(PAIRS_BY_ID)
        for p in synthesise(extra):
            if p.pair_id not in seen:
                candidates.append(p)
                seen.add(p.pair_id)

    print(f"probing {len(candidates)} candidate pair(s) on HuggingFace\n")
    confirmed, partial, absent = [], [], []

    for pair in candidates:
        if pair.pair_id in PAIRS_BY_ID:
            continue                       # already in the manifest
        b, i = pair.repo("rl_on_base"), pair.repo("rl_on_instruct")
        print(f"  {pair.pair_id}")
        has_b, has_i = exists(b, args.token), exists(i, args.token)
        if has_b and has_i:
            print(f"    FOUND both sides")
            confirmed.append(pair)
        elif has_b or has_i:
            # Useless for grafting: the delta needs the base side and the
            # comparison needs the instruct side.
            which = "base-side only" if has_b else "instruct-side only"
            print(f"    incomplete — {which}, cannot form a pair")
            partial.append((pair, which))
        else:
            print(f"    not found")
            absent.append(pair)

    if args.also_originals:
        print("\noriginals:")
        for key, spec in ORIGINALS.items():
            ok_b, ok_i = exists(spec["base"], args.token), exists(spec["instruct"], args.token)
            state = "ok" if (ok_b and ok_i) else "MISSING"
            print(f"  {key:<14} {state}   {spec['base']} / {spec['instruct']}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print(f"confirmed  : {len(confirmed)}")
    for p in confirmed:
        print(f"  + {p.pair_id}   ({HF_PREFIX}{p.rl_base})")
    if partial:
        print(f"incomplete : {len(partial)}")
        for p, which in partial:
            print(f"  ~ {p.pair_id}   ({which})")
    print(f"absent     : {len(absent)}")

    if not confirmed:
        print("\nNothing new to add. The manifest already covers what is released.")
        return

    if not args.write:
        print(f"\nRe-run with --write to add these {len(confirmed)} pair(s) to the manifest.")
        return

    payload = {"pairs": [
        {"version": p.version, "size": p.size, "algo": p.algo,
         "with_search": p.with_search, "rl_base": p.rl_base,
         "rl_instruct": p.rl_instruct}
        for p in confirmed
    ]}
    # Merge with anything already recorded, keyed by pair_id.
    if os.path.exists(VERIFIED):
        with open(VERIFIED) as fh:
            old = json.load(fh).get("pairs", [])
        by_id = {f"{r['algo']}-{'search' if r['with_search'] else 'nosearch'}"
                 f"-{r['size']}-{r['version']}": r for r in old}
        for r in payload["pairs"]:
            by_id[f"{r['algo']}-{'search' if r['with_search'] else 'nosearch'}"
                  f"-{r['size']}-{r['version']}"] = r
        payload["pairs"] = list(by_id.values())

    with open(VERIFIED, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[ok] wrote {VERIFIED} ({len(payload['pairs'])} pair(s))")
    print("They are now part of the manifest:  python shadow_rl/pairs.py")


if __name__ == "__main__":
    main()
