#!/usr/bin/env python3
"""Terminal report for the math track, mirroring shadow_rl/report.py.

    python shadow_rl/verl_math/report_math.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import Style  # noqa: E402

ROLES = ["base_baseline", "instruct_baseline", "rl_on_instruct", "rl_on_base", "shadow"]
SHORT = {
    "base_baseline": ("W_B", "untuned base"),
    "instruct_baseline": ("W_I", "untuned instruct"),
    "rl_on_instruct": ("RL(W_I)", "GRPO on instruct"),
    "rl_on_base": ("RL(W_B)", "GRPO on base"),
    "shadow": ("W_shadow", "ours"),
}
LABEL = {"aime24": "AIME24", "aime25": "AIME25", "amc23": "AMC23"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="shadow_rl/verl_math/math_results.csv")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    st = Style(not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    if not os.path.exists(args.results):
        print(f"\n  no results yet at {args.results}\n")
        return

    data = defaultdict(dict)
    ks, ns = set(), {}
    with open(args.results, newline="") as fh:
        for row in csv.DictReader(fh):
            data[row["model_role"]][row["benchmark"]] = float(row["accuracy"])
            ks.add(int(row["k"]))
            ns[row["benchmark"]] = int(row["n_problems"])

    benches = [b for b in LABEL if any(b in v for v in data.values())]
    if not benches:
        print("\n  no benchmarks scored yet\n")
        return

    k_desc = f"avg@{ks.pop()}" if len(ks) == 1 else f"avg@{min(ks)}-{max(ks)}"
    print()
    print(st.bold("  Shadow-FT on self-trained RL — Qwen3-4B, GRPO on DAPO-Math"))
    print(st.dim(f"  W_shadow = W_I + (RL(W_B) − W_B)   ·   {k_desc}   ·   "
                 + " ".join(f"{LABEL[b]} n={ns[b]}" for b in benches)))
    print(st.cyan("━" * 88))

    header = st.pad("  model", 34)
    for b in benches:
        header += st.pad(LABEL[b], 11, right=False)
    header += st.pad("Mean", 12, right=False)
    print(st.dim(header))
    print(st.dim("  " + "─" * 86))

    means = {}
    for role in ROLES:
        per = data.get(role, {})
        if not per:
            continue
        present = [b for b in benches if b in per]
        means[role] = sum(per[b] for b in present) / len(present)
        sym, desc = SHORT[role]
        name = f"  {sym:<9} {desc}"
        line = st.pad(st.green(name) if role == "shadow" else name, 34)
        for b in benches:
            line += st.pad(f"{per[b]:.3f}" if b in per else st.dim("  -  "), 11, right=False)
        line += st.pad(f"{means[role]:.3f}", 12, right=False)
        print(line)

    print()
    if "shadow" not in means:
        print(st.dim("  W_shadow not evaluated yet"))
    else:
        for competitor, label in (("rl_on_instruct", "RL(W_I)"), ("rl_on_base", "RL(W_B)")):
            if competitor not in means:
                continue
            d = means["shadow"] - means[competitor]
            mark = st.green("WIN ") if d > 0 else (st.red("LOSS") if d < 0 else st.yellow("TIE "))
            print(f"  {mark}  shadow − {label:<8} = "
                  + (st.green if d > 0 else st.red)(f"{d:+.4f}")
                  + st.dim(f"    ({means['shadow']:.4f} vs {means[competitor]:.4f})"))
    print()
    print(st.dim("  AIME24/25 are 30 problems each and AMC23 is 40, so even avg@k is"))
    print(st.dim("  noisy — a margin under ~0.05 here is not a result. Raise --k."))
    print()


if __name__ == "__main__":
    main()
