#!/usr/bin/env python3
"""Tests for the pair manifest, the --pairs selector, and candidate discovery.

The selector decides what a run actually evaluates, so a silent mistake here
wastes GPU hours on the wrong checkpoints. The discovery path matters for a
different reason: candidate ids are *inferred* from Search-R1's naming
convention, and an unverified id must never reach a run.

Run:  python shadow_rl/tests/test_pairs.py
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def select(spec, env=None):
    """Run run_all.sh's resolver the way the script does, returning pair ids."""
    code = open(os.path.join(ROOT, "run_all.sh")).read()
    start = code.index('PAIR_LIST_RAW=$(python3 - "$PAIRS_SEL" <<\'PY\'')
    body = code[code.index("\n", start) + 1:code.index("\nPY\n", start)]
    proc = subprocess.run([sys.executable, "-c", body, spec],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, **(env or {})})
    return proc.returncode, proc.stdout.split(), proc.stderr


def main():
    import pairs as P

    print("\n[1] the shipped manifest is internally consistent")
    check("15 released pairs", len(P.PAIRS) == 15, f"{len(P.PAIRS)}")
    check("pair ids unique", len({p.pair_id for p in P.PAIRS}) == len(P.PAIRS))
    check("every size resolves to originals",
          all(p.size in P.ORIGINALS for p in P.PAIRS))
    check("every candidate size resolves to originals",
          all(p.size in P.ORIGINALS for p in P.CANDIDATES),
          str(sorted({p.size for p in P.CANDIDATES} - set(P.ORIGINALS))))
    check("candidates do not duplicate released pairs",
          not ({p.pair_id for p in P.CANDIDATES} & {p.pair_id for p in P.PAIRS}))
    check("base and instruct differ for every size",
          all(v["base"] != v["instruct"] for v in P.ORIGINALS.values()))

    print("\n[2] group selectors")
    for spec, expect in [
        ("v0.3", {"grpo-search-3b-v0.3", "grpo-search-7b-v0.3", "ppo-search-3b-v0.3",
                  "grpo-search-14b-v0.3"}),
        ("nosearch", {"ppo-nosearch-3b-v0.2", "ppo-nosearch-7b-v0.2"}),
    ]:
        rc, got, err = select(spec)
        check(f"--pairs {spec}", rc == 0 and set(got) == expect,
              f"got {sorted(got)}")

    rc, got, _ = select("v0.3,grpo")
    check("groups intersect, not union",
          rc == 0 and set(got) == {"grpo-search-3b-v0.3", "grpo-search-7b-v0.3",
                                   "grpo-search-14b-v0.3"},
          f"got {sorted(got)}")

    rc, got, _ = select("v0.3,ppo")
    check("v0.3,ppo narrows to one", rc == 0 and got == ["ppo-search-3b-v0.3"], f"{got}")

    rc, got, _ = select("latest")
    rc2, got2, _ = select("v0.3")
    check("latest aliases v0.3", got == got2 and rc == rc2 == 0)

    rc, got, _ = select("qwen")
    n_llama = sum(1 for p in P.PAIRS if p.size.startswith("llama"))
    check("qwen excludes the llama pairs",
          rc == 0 and len(got) == len(P.PAIRS) - n_llama, f"{len(got)} of {len(P.PAIRS)}")

    rc, got, _ = select("llama")
    check("llama selects the llama pairs", rc == 0 and len(got) == n_llama, f"{got}")

    print("\n[2b] exclusion syntax")
    rc, got, _ = select("all,-v0.2")
    check("all,-v0.2 drops every v0.2 pair",
          rc == 0 and not any(g.endswith("v0.2") for g in got), f"{sorted(got)}")
    check("all,-v0.2 keeps the rest",
          len(got) == len([p for p in P.PAIRS if p.version != "v0.2"]), f"{len(got)}")
    rc, got, _ = select("demo")
    check("demo is exactly one pair", rc == 0 and got == ["grpo-search-3b-v0.3"], f"{got}")
    rc, got, err = select("-v0.2")
    check("a bare exclusion is rejected", rc != 0, err.strip()[:60])
    rc, got, err = select("all,-nosuchgroup")
    check("excluding an unknown group fails", rc != 0 and "unknown group" in err)

    print("\n[3] bad selections are rejected, not silently widened")
    rc, got, err = select("v0.3,ppo-search-3b-v0.3")
    check("mixing a group with an id fails", rc != 0 and "do not mix" in err, err.strip()[:70])
    rc, got, err = select("no-such-pair")
    check("unknown id fails", rc != 0 and "unknown pair" in err)
    rc, got, err = select("v0.9")
    check("unknown group treated as an id and fails", rc != 0)

    print("\n[4] verified_pairs.json is merged into the manifest")
    verified = os.path.join(ROOT, "verified_pairs.json")
    backup = verified + ".bak" if os.path.exists(verified) else None
    if backup:
        os.rename(verified, backup)
    try:
        with open(verified, "w") as fh:
            json.dump({"pairs": [{
                "version": "v0.3", "size": "llama3.2-3b", "algo": "grpo",
                "with_search": True,
                "rl_base": "SearchR1-nq_hotpotqa_train-llama3.2-3b-em-grpo-v0.3",
                "rl_instruct": "SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-grpo-v0.3",
            }]}, fh)
        rc, got, _ = select("llama")
        check("a verified pair becomes selectable",
              rc == 0 and "grpo-search-llama3.2-3b-v0.3" in got, f"{got}")
        rc, got, _ = select("v0.3")
        check("and joins its version group",
              "grpo-search-llama3.2-3b-v0.3" in got, f"{sorted(got)}")

        # Its originals must point at Llama, not Qwen.
        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,'shadow_rl');"
             "from pairs import PAIRS_BY_ID as B;"
             "p=B['grpo-search-llama3.2-3b-v0.3'];print(p.base);print(p.instruct)"],
            capture_output=True, text=True, cwd=REPO)
        check("verified pair resolves Llama originals",
              "meta-llama/Llama-3.2-3B" in proc.stdout
              and "Llama-3.2-3B-Instruct" in proc.stdout, proc.stdout.strip()[:80])

        print("\n[5] a corrupt verified file is ignored, not fatal")
        with open(verified, "w") as fh:
            fh.write("{ this is not json")
        rc, got, err = select("v0.3")
        n_v03 = len([p for p in P.PAIRS if p.version == "v0.3"])
        check("manifest still loads", rc == 0 and len(got) == n_v03, f"rc={rc} got={got}")
        check("warns about the bad file", "ignoring" in err or "warn" in err.lower(),
              err.strip()[:80])
    finally:
        if os.path.exists(verified):
            os.unlink(verified)
        if backup:
            os.rename(backup, verified)

    print("\n[6] discover.py synthesises ids matching the release convention")
    import discover as D
    made = D.synthesise(["v0.4"])
    ids = {p.rl_base for p in made}
    check("qwen naming", "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.4" in ids)
    check("llama naming", "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-ppo-v0.4" in ids)
    check("instruct side uses the -it- infix",
          all("-it-em-" in p.rl_instruct for p in made))
    check("base side has no -it-", all("-it-em-" not in p.rl_base for p in made))
    check("every synthesised size is known",
          all(p.size in P.ORIGINALS for p in made))

    proc = subprocess.run([sys.executable, os.path.join(ROOT, "discover.py"), "--help"],
                          capture_output=True, text=True)
    check("discover.py --help works", proc.returncode == 0)

    print("\n[7] v0.1 repo names carry no version suffix")
    # The first release predates the convention: `...-em-grpo`, not
    # `...-em-grpo-v0.1`. Getting this wrong means every v0.1 probe 404s.
    check("v0.1 base has no suffix",
          D.repo_name("7b", "grpo", "v0.1", instruct=False)
          == "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo")
    check("v0.1 instruct has no suffix",
          D.repo_name("7b", "grpo", "v0.1", instruct=True)
          == "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo")
    check("v0.2 is suffixed",
          D.repo_name("7b", "ppo", "v0.2", instruct=False).endswith("-em-ppo-v0.2"))
    check("v0.3 is suffixed",
          D.repo_name("3b", "grpo", "v0.3", instruct=False).endswith("-em-grpo-v0.3"))
    check("llama slug omits the qwen prefix",
          D.repo_name("llama3.1-8b", "grpo", "v0.1", instruct=False)
          == "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-grpo")

    # The three v0.1 pairs we were originally given must round-trip exactly.
    for pid, size, algo in [("grpo-search-3b-v0.1", "3b", "grpo"),
                            ("ppo-search-3b-v0.1", "3b", "ppo"),
                            ("ppo-search-7b-v0.1", "7b", "ppo")]:
        want = P.PAIRS_BY_ID[pid]
        check(f"{pid} matches the generated name",
              D.repo_name(size, algo, "v0.1", False) == want.rl_base
              and D.repo_name(size, algo, "v0.1", True) == want.rl_instruct,
              want.rl_base)

    check("--sweep covers every known version",
          {p.version for p in D.synthesise(["v0.1", "v0.2", "v0.3"])}
          == {"v0.1", "v0.2", "v0.3"})

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
