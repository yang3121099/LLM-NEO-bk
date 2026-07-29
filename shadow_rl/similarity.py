#!/usr/bin/env python3
"""Parameter-level similarity statistics for a matched RL checkpoint pair.

Runs before any merging. Answers three questions per parameter tensor:

1. **How far apart are the backbones?**  `sigma = sum|W_B - W_I| / (sum|W_B| + sum|W_I|)`
   and `cos(W_B, W_I)`. This is the Shadow-FT applicability condition; the paper's
   working range is sigma in 0.003-0.042.

2. **How big is each RL update?**  `||RL(W)-W|| / ||W||` for both sides. If the
   base-side update is much larger than the instruct-side one, direct RL had more
   room to work on the base model -- which is the premise of grafting.

3. **Do the two RL runs learn the same thing?**  `cos(Delta_B, Delta_I)` where
   `Delta_B = RL(W_B)-W_B` and `Delta_I = RL(W_I)-W_I`. This is the statistic that
   most directly predicts whether grafting can work: a high cosine means the update
   learned on the base model points the same way as the one learned on the instruct
   model, so transplanting it is well-posed. A cosine near zero means the two runs
   found unrelated solutions and grafting is a gamble.

Everything streams one tensor at a time -- four checkpoints open, one tensor resident.

    python shadow_rl/similarity.py --pair ppo-nosearch-3b-v0.2
    python shadow_rl/similarity.py --all
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge import SIGMA_HI, SIGMA_LO, Checkpoint, validate  # noqa: E402
from pairs import PAIRS, PAIRS_BY_ID, Pair  # noqa: E402

PARAM_FIELDS = [
    "pair_id", "param", "layer", "module", "numel",
    "sigma", "cos_base_instruct",
    "rel_delta_base", "rel_delta_instruct", "cos_delta_base_instruct",
]
SUMMARY_FIELDS = [
    "pair_id", "group", "n_tensors", "numel",
    "sigma", "cos_base_instruct",
    "rel_delta_base", "rel_delta_instruct", "cos_delta_base_instruct",
]

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_MODULE_RE = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|embed_tokens|lm_head|input_layernorm|post_attention_layernorm|norm)"
)


def classify(name: str):
    """Split a parameter name into (layer index, module kind)."""
    layer = _LAYER_RE.search(name)
    module = _MODULE_RE.search(name)
    return (int(layer.group(1)) if layer else -1,
            module.group(1) if module else "other")


class Stats:
    """Streaming accumulators; every quantity is a ratio of sums, so tensors
    can be folded in one at a time and the group totals stay exact."""

    __slots__ = ("n", "numel", "abs_diff", "abs_b", "abs_i",
                 "dot_bi", "sq_b", "sq_i",
                 "sq_db", "sq_di", "dot_dbdi")

    def __init__(self):
        self.n = self.numel = 0
        self.abs_diff = self.abs_b = self.abs_i = 0.0
        self.dot_bi = self.sq_b = self.sq_i = 0.0
        self.sq_db = self.sq_di = self.dot_dbdi = 0.0

    def add(self, other: "Stats") -> None:
        for f in self.__slots__:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @staticmethod
    def _ratio(num, den):
        return num / den if den > 0 else float("nan")

    @staticmethod
    def _cos(dot, sq_a, sq_b):
        den = (sq_a ** 0.5) * (sq_b ** 0.5)
        return dot / den if den > 0 else float("nan")

    def row(self) -> Dict[str, float]:
        return {
            "numel": self.numel,
            "sigma": self._ratio(self.abs_diff, self.abs_b + self.abs_i),
            "cos_base_instruct": self._cos(self.dot_bi, self.sq_b, self.sq_i),
            "rel_delta_base": self._ratio(self.sq_db ** 0.5, self.sq_b ** 0.5),
            "rel_delta_instruct": self._ratio(self.sq_di ** 0.5, self.sq_i ** 0.5),
            "cos_delta_base_instruct": self._cos(self.dot_dbdi, self.sq_db, self.sq_di),
        }


def tensor_stats(w_b, w_i, w_rb, w_ri) -> Stats:
    """All accumulation in float64 -- these are sums over ~10^9 elements."""
    s = Stats()
    b, i = w_b.double(), w_i.double()
    d_b = w_rb.double() - b
    d_i = w_ri.double() - i if w_ri is not None else None

    s.n = 1
    s.numel = b.numel()
    s.abs_diff = float((b - i).abs().sum())
    s.abs_b = float(b.abs().sum())
    s.abs_i = float(i.abs().sum())
    s.dot_bi = float((b * i).sum())
    s.sq_b = float(b.pow(2).sum())
    s.sq_i = float(i.pow(2).sum())
    s.sq_db = float(d_b.pow(2).sum())
    if d_i is not None:
        s.sq_di = float(d_i.pow(2).sum())
        s.dot_dbdi = float((d_b * d_i).sum())
    return s


def analyse(pair: Pair, ignore: Optional[str], device: str):
    """Yield (name, layer, module, Stats) for every float parameter."""
    ckpts = [
        Checkpoint(pair.base, "base"),
        Checkpoint(pair.instruct, "instruct"),
        Checkpoint(pair.repo("rl_on_base"), "rl_base"),
        Checkpoint(pair.repo("rl_on_instruct"), "rl_instruct"),
    ]
    base, instruct, rl_base, rl_instruct = ckpts
    keys = validate(ckpts, ignore)

    try:
        for idx, key in enumerate(keys, 1):
            w_b = base.get(key)
            if not w_b.is_floating_point():
                continue
            # fp32 on the accelerator, then reduce in fp64 on the host.
            w_b = w_b.to(device, torch.float32)
            w_i = instruct.get(key).to(device, torch.float32)
            w_rb = rl_base.get(key).to(device, torch.float32)
            w_ri = rl_instruct.get(key).to(device, torch.float32)

            layer, module = classify(key)
            yield key, layer, module, tensor_stats(w_b, w_i, w_rb, w_ri)

            if idx % 50 == 0:
                print(f"  [{idx}/{len(keys)}] {key}", flush=True)
            del w_b, w_i, w_rb, w_ri
    finally:
        for c in ckpts:
            c.close()


def run_pair(pair: Pair, param_writer, summary_writer, ignore, device) -> Dict[str, float]:
    print(f"\n{'=' * 70}\n[pair] {pair.pair_id}\n{'=' * 70}")
    print(f"  base      : {pair.base}")
    print(f"  instruct  : {pair.instruct}")
    print(f"  RL(base)  : {pair.repo('rl_on_base')}")
    print(f"  RL(instr) : {pair.repo('rl_on_instruct')}")

    by_module: Dict[str, Stats] = defaultdict(Stats)
    by_layer: Dict[int, Stats] = defaultdict(Stats)
    overall = Stats()

    for name, layer, module, st in analyse(pair, ignore, device):
        row = st.row()
        param_writer.writerow({
            "pair_id": pair.pair_id, "param": name,
            "layer": layer if layer >= 0 else "", "module": module,
            **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()},
        })
        by_module[module].add(st)
        if layer >= 0:
            by_layer[layer].add(st)
        overall.add(st)

    def emit(group: str, st: Stats):
        row = st.row()
        summary_writer.writerow({
            "pair_id": pair.pair_id, "group": group, "n_tensors": st.n,
            **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()},
        })
        return row

    for module in sorted(by_module):
        emit(f"module:{module}", by_module[module])
    for layer in sorted(by_layer):
        emit(f"layer:{layer:03d}", by_layer[layer])
    total = emit("ALL", overall)

    # ---- console summary ---------------------------------------------------
    print(f"\n  --- {pair.pair_id} ---")
    print(f"  sigma (W_B vs W_I)            : {total['sigma']:.5f}"
          f"   [paper range {SIGMA_LO}-{SIGMA_HI}]")
    if not (SIGMA_LO <= total["sigma"] <= SIGMA_HI):
        print("    ** WARNING: outside the paper's working range **")
    print(f"  cos(W_B, W_I)                 : {total['cos_base_instruct']:.5f}")
    print(f"  ||RL(W_B)-W_B|| / ||W_B||     : {total['rel_delta_base']:.5f}")
    print(f"  ||RL(W_I)-W_I|| / ||W_I||     : {total['rel_delta_instruct']:.5f}")
    print(f"  cos(Delta_B, Delta_I)         : {total['cos_delta_base_instruct']:.5f}")

    cos_d = total["cos_delta_base_instruct"]
    if cos_d == cos_d:  # not nan
        if cos_d > 0.3:
            verdict = "the two RL runs learned aligned updates -- grafting is well-posed"
        elif cos_d > 0.05:
            verdict = "weak alignment between the two RL updates"
        else:
            verdict = "the two RL runs found largely unrelated updates -- treat the graft as exploratory"
        print(f"    -> {verdict}")

    print("\n  by module:")
    print(f"    {'module':<22}{'sigma':>10}{'relD_base':>12}{'relD_instr':>12}{'cos(dB,dI)':>12}")
    for module in sorted(by_module):
        r = by_module[module].row()
        print(f"    {module:<22}{r['sigma']:>10.5f}{r['rel_delta_base']:>12.5f}"
              f"{r['rel_delta_instruct']:>12.5f}{r['cos_delta_base_instruct']:>12.5f}")

    if by_layer:
        layers = sorted(by_layer)
        print(f"\n  by layer (first/middle/last):")
        for layer in (layers[0], layers[len(layers) // 2], layers[-1]):
            r = by_layer[layer].row()
            print(f"    layer {layer:<16}{r['sigma']:>10.5f}{r['rel_delta_base']:>12.5f}"
                  f"{r['rel_delta_instruct']:>12.5f}{r['cos_delta_base_instruct']:>12.5f}")

    return {"pair_id": pair.pair_id, **total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair", choices=sorted(PAIRS_BY_ID))
    g.add_argument("--all", action="store_true", help="every pair in the manifest")
    ap.add_argument("--params-out", default="shadow_rl/similarity_params.csv")
    ap.add_argument("--summary-out", default="shadow_rl/similarity_summary.csv")
    ap.add_argument("--ignore-keys", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--append", action="store_true", help="add to existing CSVs")
    args = ap.parse_args()

    targets: List[Pair] = PAIRS if args.all else [PAIRS_BY_ID[args.pair]]
    print(f"[info] {len(targets)} pair(s), device={args.device}")

    for path in (args.params_out, args.summary_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(args.params_out))

    totals = []
    with open(args.params_out, mode, newline="") as pf, \
         open(args.summary_out, mode, newline="") as sf:
        pw = csv.DictWriter(pf, fieldnames=PARAM_FIELDS)
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        if write_header:
            pw.writeheader()
            sw.writeheader()

        for pair in targets:
            try:
                totals.append(run_pair(pair, pw, sw, args.ignore_keys, args.device))
            except SystemExit as exc:
                print(f"[fail] {pair.pair_id}: {exc}", file=sys.stderr)
            except Exception as exc:                      # keep going across pairs
                print(f"[fail] {pair.pair_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            pf.flush()
            sf.flush()

    if totals:
        print(f"\n{'=' * 92}\nALL PAIRS\n{'=' * 92}")
        print(f"{'pair':<26}{'sigma':>9}{'cos(B,I)':>10}{'relD_base':>11}"
              f"{'relD_instr':>12}{'cos(dB,dI)':>12}{'':>4}")
        for t in totals:
            flag = "" if SIGMA_LO <= t["sigma"] <= SIGMA_HI else "  <- sigma"
            print(f"{t['pair_id']:<26}{t['sigma']:>9.5f}{t['cos_base_instruct']:>10.5f}"
                  f"{t['rel_delta_base']:>11.5f}{t['rel_delta_instruct']:>12.5f}"
                  f"{t['cos_delta_base_instruct']:>12.5f}{flag}")

    print(f"\n[ok] per-parameter -> {args.params_out}")
    print(f"[ok] per-group     -> {args.summary_out}")


if __name__ == "__main__":
    main()
