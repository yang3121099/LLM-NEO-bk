#!/usr/bin/env python3
"""Tests for the parameter-level similarity statistics.

Builds four tiny checkpoints with *known* relationships between them and checks
that the reported sigma, cosines and relative deltas come back with the values
those relationships imply.

Run:  python shadow_rl/tests/test_similarity.py
"""

import csv
import math
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_merge import ALL_KEYS, FLOAT_KEYS, random_tensors, write_ckpt  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def close(a, b, tol=2e-3):
    return a == a and abs(a - b) <= tol


def main():
    import pairs as pairs_mod
    import similarity as sim

    print("\n[1] classify() splits parameter names")
    check("layer + module", sim.classify("model.layers.12.self_attn.q_proj.weight") == (12, "q_proj"))
    check("embedding has no layer", sim.classify("model.embed_tokens.weight") == (-1, "embed_tokens"))
    check("final norm", sim.classify("model.norm.weight") == (-1, "norm"))
    check("unknown -> other", sim.classify("some.random.thing") == (-1, "other"))

    print("\n[2] Stats folds tensors exactly")
    g = torch.Generator().manual_seed(7)
    a, b = torch.randn(64, generator=g), torch.randn(64, generator=g)
    ra, rb = a + 0.01 * torch.randn(64, generator=g), b + 0.01 * torch.randn(64, generator=g)
    whole = sim.tensor_stats(a, b, ra, rb)
    halves = sim.Stats()
    for sl in (slice(0, 32), slice(32, 64)):
        halves.add(sim.tensor_stats(a[sl], b[sl], ra[sl], rb[sl]))
    wr, hr = whole.row(), halves.row()
    check("sigma is split-invariant", close(wr["sigma"], hr["sigma"], 1e-9))
    check("cosine is split-invariant",
          close(wr["cos_delta_base_instruct"], hr["cos_delta_base_instruct"], 1e-9))

    print("\n[3] known-relationship checkpoints")
    with tempfile.TemporaryDirectory() as tmp:
        base_t = random_tensors(0)

        # W_I = W_B + eps  -> cos(W_B, W_I) very close to 1, sigma small.
        eps = torch.Generator().manual_seed(11)
        inst_t = {k: (v + torch.randn(v.shape, generator=eps) * 0.002
                      if v.is_floating_point() else v.clone())
                  for k, v in base_t.items()}

        # Both RL runs apply the SAME update -> cos(Delta_B, Delta_I) == 1.
        dg = torch.Generator().manual_seed(12)
        delta = {k: (torch.randn(v.shape, generator=dg) * 0.001
                     if v.is_floating_point() else torch.zeros_like(v))
                 for k, v in base_t.items()}
        rlb_t = {k: v + delta[k] for k, v in base_t.items()}
        rli_t = {k: v + delta[k] for k, v in inst_t.items()}

        paths = {}
        for name, tensors, dtype in (("base", base_t, torch.float32),
                                     ("instruct", inst_t, torch.float32),
                                     ("rl_base", rlb_t, torch.float32),
                                     ("rl_instruct", rli_t, torch.float32)):
            paths[name] = os.path.join(tmp, name)
            write_ckpt(paths[name], tensors, dtype)

        # Point the manifest at the local dirs.
        pairs_mod.ORIGINALS["3b"] = {"base": paths["base"], "instruct": paths["instruct"]}
        pairs_mod.HF_PREFIX = ""
        pair = pairs_mod.PAIRS_BY_ID["ppo-nosearch-3b-v0.2"]
        object.__setattr__(pair, "rl_base", paths["rl_base"])
        object.__setattr__(pair, "rl_instruct", paths["rl_instruct"])
        sim.SIGMA_LO, sim.SIGMA_HI = 0.0, 1.0

        params_csv = os.path.join(tmp, "params.csv")
        summary_csv = os.path.join(tmp, "summary.csv")
        with open(params_csv, "w", newline="") as pf, open(summary_csv, "w", newline="") as sf:
            pw = csv.DictWriter(pf, fieldnames=sim.PARAM_FIELDS)
            sw = csv.DictWriter(sf, fieldnames=sim.SUMMARY_FIELDS)
            pw.writeheader()
            sw.writeheader()
            total = sim.run_pair(pair, pw, sw, None, "cpu")

        check("identical RL updates -> cos(dB,dI) == 1",
              close(total["cos_delta_base_instruct"], 1.0, 1e-3),
              f"{total['cos_delta_base_instruct']:.5f}")
        check("near-identical backbones -> cos(W_B,W_I) ~ 1",
              close(total["cos_base_instruct"], 1.0, 1e-2),
              f"{total['cos_base_instruct']:.5f}")
        check("sigma small for near-identical backbones",
              0.0 < total["sigma"] < 0.1, f"{total['sigma']:.5f}")
        check("both relative deltas positive",
              total["rel_delta_base"] > 0 and total["rel_delta_instruct"] > 0)

        with open(params_csv) as fh:
            prows = list(csv.DictReader(fh))
        with open(summary_csv) as fh:
            srows = list(csv.DictReader(fh))

        check("one row per float parameter", len(prows) == len(FLOAT_KEYS),
              f"{len(prows)} rows vs {len(FLOAT_KEYS)} float params")
        check("integer buffer excluded",
              all("int_thing" not in r["param"] for r in prows))
        check("module column populated",
              {r["module"] for r in prows} >= {"q_proj", "down_proj", "norm"})
        check("layer column populated for layer params",
              {r["layer"] for r in prows if r["module"] == "q_proj"} == {"0", "1"})
        check("summary has ALL row", any(r["group"] == "ALL" for r in srows))
        check("summary has per-module rows",
              any(r["group"] == "module:q_proj" for r in srows))
        check("summary has per-layer rows",
              any(r["group"] == "layer:000" for r in srows))

        # Group totals must equal the whole-model total, not an average of ratios.
        all_row = next(r for r in srows if r["group"] == "ALL")
        check("ALL numel == sum of parameter numels",
              int(all_row["numel"]) == sum(int(r["numel"]) for r in prows))

        print("\n[4] orthogonal RL updates are detected")
        # Delta_B and Delta_I drawn independently -> cosine near 0.
        og = torch.Generator().manual_seed(21)
        rli_orth = {k: (v + torch.randn(v.shape, generator=og) * 0.001
                        if v.is_floating_point() else v.clone())
                    for k, v in inst_t.items()}
        p_orth = os.path.join(tmp, "rl_instruct_orth")
        write_ckpt(p_orth, rli_orth, torch.float32)
        object.__setattr__(pair, "rl_instruct", p_orth)

        with open(os.path.join(tmp, "p2.csv"), "w", newline="") as pf, \
             open(os.path.join(tmp, "s2.csv"), "w", newline="") as sf:
            pw = csv.DictWriter(pf, fieldnames=sim.PARAM_FIELDS)
            sw = csv.DictWriter(sf, fieldnames=sim.SUMMARY_FIELDS)
            pw.writeheader()
            sw.writeheader()
            t2 = sim.run_pair(pair, pw, sw, None, "cpu")

        check("independent RL updates -> cos(dB,dI) ~ 0",
              abs(t2["cos_delta_base_instruct"]) < 0.1,
              f"{t2['cos_delta_base_instruct']:.5f}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
