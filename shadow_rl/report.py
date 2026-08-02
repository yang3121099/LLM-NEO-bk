#!/usr/bin/env python3
"""Render results.csv as a comparison table in the terminal.

Companion to aggregate.py, which writes FINDINGS.md. This one is for reading at
a glance while a run is in progress: one block per pair, the five models as
rows, datasets as columns, with the two margins that matter called out.

    python shadow_rl/report.py                       # every pair with results
    python shadow_rl/report.py --pair ppo-nosearch-3b-v0.2
    python shadow_rl/report.py --datasets hotpotqa,musique,bamboogle,simpleqa
    python shadow_rl/report.py --no-color            # for piping to a file
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import avg, is_complete, load, se_avg, se_diff  # noqa: E402
from pairs import (  # noqa: E402
    DATASETS, IN_DOMAIN, MODEL_ROLES, PAIRS, reference_avg, reference_for,
)

# Short labels; the long ones do not fit next to seven dataset columns.
SHORT = {
    "base_baseline": ("W_B", "untuned base"),
    "instruct_baseline": ("W_I", "untuned instruct"),
    "rl_on_instruct": ("RL(W_I)", "direct RL on instruct"),
    "rl_on_base": ("RL(W_B)", "direct RL on base"),
    "shadow": ("W_shadow", "ours"),
}
ABBREV = {
    "nq": "NQ", "triviaqa": "TriviaQA", "popqa": "PopQA", "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki", "musique": "Musique", "bamboogle": "Bamboogle",
    "gpqa_diamond": "GPQA-D", "simpleqa": "SimpleQA",
}


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t):   return self._w("1", t)
    def dim(self, t):    return self._w("2", t)
    def green(self, t):  return self._w("1;32", t)
    def red(self, t):    return self._w("1;31", t)
    def yellow(self, t): return self._w("1;33", t)
    def cyan(self, t):   return self._w("1;36", t)

    def width(self, text: str) -> int:
        """Printable width, ignoring ANSI escapes."""
        if not self.on:
            return len(text)
        out, i = 0, 0
        while i < len(text):
            if text[i] == "\033":
                while i < len(text) and text[i] != "m":
                    i += 1
                i += 1
            else:
                out += 1
                i += 1
        return out

    def pad(self, text: str, n: int, right=True) -> str:
        gap = " " * max(0, n - self.width(text))
        return (text + gap) if right else (gap + text)


def fmt_em(x, st: Style, best=False):
    if x is None:
        return st.dim("  -  ")
    s = f"{x:.3f}"
    return st.bold(s) if best else s


def render_pair(pair, roles_data, nmap, counts, st: Style, out):
    present = [d for d in DATASETS if any(d in roles_data.get(r, {}) for r in MODEL_ROLES)]
    if not present:
        return

    tag = "with search" if pair.with_search else "no search"
    n_q = sorted({n for d in present for n in counts.get(d, set())})
    n_desc = f"n={n_q[0]}/dataset" if len(n_q) == 1 else f"n={min(n_q)}-{max(n_q)}/dataset"

    out.append("")
    out.append(st.cyan("━" * 96))
    cov = f"{len(present)}/{len(DATASETS)} datasets"
    out.append(st.bold(f"  {pair.pair_id}") +
               st.dim(f"    {pair.algo.upper()} · {pair.size.upper()} · {tag} · {pair.version}"
                      f"    {cov} · {n_desc}"))
    out.append(st.cyan("━" * 96))

    label_w = 34
    col_w = 10

    header = st.pad("  model", label_w)
    for d in present:
        name = ABBREV.get(d, d)
        if d in IN_DOMAIN:
            name += "*"
        header += st.pad(name, col_w, right=False)
    header += st.pad("Avg", col_w + 2, right=False) + "  ±se"
    out.append(st.dim(header))
    out.append(st.dim("  " + "─" * 94))

    # Best per column, so the winner is visible without reading every number.
    best = {}
    for d in present + ["__avg__"]:
        vals = []
        for r in MODEL_ROLES:
            per = roles_data.get(r, {})
            v = avg(per) if d == "__avg__" else per.get(d)
            if v is not None:
                vals.append(v)
        best[d] = max(vals) if vals else None

    shadow_avg = avg(roles_data.get("shadow", {}))

    for role in MODEL_ROLES:
        per = roles_data.get(role, {})
        if not per:
            continue
        sym, desc = SHORT[role]
        is_shadow = role == "shadow"
        name = f"  {sym:<9} {desc}"
        line = st.pad(st.green(name) if is_shadow else name, label_w)
        for d in present:
            v = per.get(d)
            cell = fmt_em(v, st, best=v is not None and best[d] is not None and v >= best[d])
            line += st.pad(cell, col_w, right=False)
        a = avg(per)
        cell = fmt_em(a, st, best=a is not None and best["__avg__"] is not None and a >= best["__avg__"])
        # Only flag a role that is missing data the other roles have -- the
        # pair-level coverage is already stated in the header.
        if a is not None and len(per) < len(present):
            cell += st.yellow("!")
        line += st.pad(cell, col_w + 2, right=False)
        sd = se_avg(per, nmap.get(role, {}))
        if sd is not None:
            line += st.dim(f"  ±{sd:.3f}")
        out.append(line)

    # ---- the two margins the experiment is about --------------------------- #
    out.append("")
    if shadow_avg is None:
        out.append(st.dim("  W_shadow not evaluated for this pair yet"))
    else:
        for competitor, label in (("rl_on_instruct", "RL(W_I)"), ("rl_on_base", "RL(W_B)")):
            other = avg(roles_data.get(competitor, {}))
            if other is None:
                continue
            d = shadow_avg - other
            sd = se_diff(roles_data.get("shadow", {}), nmap.get("shadow", {}),
                         roles_data.get(competitor, {}), nmap.get(competitor, {}))
            mark = st.green("WIN ") if d > 0 else (st.red("LOSS") if d < 0 else st.yellow("TIE "))
            line = (f"  {mark}  shadow − {label:<8} = "
                    + (st.green if d > 0 else st.red)(f"{d:+.4f}")
                    + st.dim(f"    ({shadow_avg:.4f} vs {other:.4f})"))
            if sd is not None:
                # 2 SE ~ 95%. The bound is conservative: both models saw the same
                # questions, so the paired error is smaller than this.
                if abs(d) >= 2 * sd:
                    line += st.green(f"    solid (|d| > 2×se={2 * sd:.4f})")
                elif abs(d) >= sd:
                    line += st.yellow(f"    weak (se={sd:.4f}, needs more questions)")
                else:
                    line += st.red(f"    WITHIN NOISE (se={sd:.4f}) — not a result yet")
            out.append(line)

    # ---- reproduction against the published numbers ------------------------ #
    checks = []
    for role in ("rl_on_base", "rl_on_instruct"):
        per = roles_data.get(role, {})
        ours = avg(per)
        ref = reference_avg(pair, role, subset=set(per))
        if ours is None or ref is None:
            continue
        d = ours - ref
        flag = st.green("ok") if abs(d) <= 0.05 else st.yellow("off")
        checks.append(f"{SHORT[role][0]} {ours:.3f} vs {ref:.3f} published ({d:+.3f}) {flag}")
    if checks:
        out.append("")
        out.append(st.dim("  harness check:  ") + st.dim(" | ").join(checks))


def show_progress(results, args, st: Style) -> None:
    """Completion grid, readable while a run is still going.

    results.csv is appended one dataset at a time and flushed, so this reflects
    live state -- a cell is filled the moment that (pair, role, dataset) is
    scored, without waiting for the pair to finish.
    """
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    if args.pairs:
        want = [p for p in PAIRS if p.pair_id in
                {x.strip() for x in args.pairs.split(",")}]
    else:
        want = [p for p in PAIRS if p.pair_id in results]
    if not want:
        print("\n  " + st.yellow("no results yet — nothing has been scored") + "\n")
        return

    # Datasets actually being evaluated: inferred from what has appeared so far.
    seen = {d for pid in results for r in results[pid] for d in results[pid][r]}
    cols = [d for d in DATASETS if d in seen] or DATASETS

    done = total = 0
    print("")
    print(st.bold("  progress") + st.dim("   ✓ scored · · pending"))
    print(st.cyan("━" * 96))
    header = st.pad("  pair / model", 40)
    for d in cols:
        header += st.pad(ABBREV.get(d, d)[:9], 10, right=False)
    print(st.dim(header))

    for pair in want:
        rd = results.get(pair.pair_id, {})
        print(st.dim("  " + "─" * 94))
        print(st.bold(f"  {pair.pair_id}"))
        for role in roles:
            per = rd.get(role, {})
            sym, _ = SHORT.get(role, (role, ""))
            line = st.pad(f"    {sym}", 40)
            for d in cols:
                total += 1
                if d in per:
                    done += 1
                    line += st.pad(st.green("✓"), 10, right=False)
                else:
                    line += st.pad(st.dim("·"), 10, right=False)
            print(line)

    pct = 100.0 * done / total if total else 0.0
    bar_w = 40
    filled = int(bar_w * done / total) if total else 0
    bar = st.green("█" * filled) + st.dim("░" * (bar_w - filled))
    print("")
    print(f"  {bar}  {done}/{total} cells ({pct:.0f}%)")
    print("")
    print(st.dim("  results are written per dataset, so this is live. "
                 "Ctrl-C is safe: re-running skips what is already here."))
    print("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="shadow_rl/results.csv")
    ap.add_argument("--pair", default=None, help="only this pair")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--progress", action="store_true",
                    help="completion grid: which (role, dataset) cells are done")
    ap.add_argument("--datasets", default=None,
                    help="only these datasets in the avg (comma list, e.g. hotpotqa,bamboogle)")
    ap.add_argument("--roles", default=",".join(MODEL_ROLES),
                    help="roles the run was launched with, for the progress denominator")
    ap.add_argument("--pairs", default=None,
                    help="pair ids the run was launched with (default: those with results)")
    args = ap.parse_args()

    st = Style(not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    results, counts, nmap = load(args.results)

    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",") if d.strip()}
        bad = keep - set(DATASETS)
        if bad:
            sys.exit(f"unknown dataset(s): {', '.join(sorted(bad))}")
        for pid in results:
            for role in results[pid]:
                results[pid][role] = {d: v for d, v in results[pid][role].items() if d in keep}
        for pid in nmap:
            for role in nmap[pid]:
                nmap[pid][role] = {d: v for d, v in nmap[pid][role].items() if d in keep}
        counts = {d: v for d, v in counts.items() if d in keep}

    if args.progress:
        show_progress(results, args, st)
        return

    out = []
    out.append("")
    out.append(st.bold("  Shadow-FT weight grafting on matched RL checkpoints"))
    out.append(st.dim("  W_shadow = W_I + (RL(W_B) − W_B)   ·   metric: Exact Match   ·   "
                      "* = in-domain   ·   ! = incomplete row"))

    shown = 0
    for pair in PAIRS:
        if args.pair and pair.pair_id != args.pair:
            continue
        if pair.pair_id not in results:
            continue
        render_pair(pair, results[pair.pair_id], nmap.get(pair.pair_id, {}),
                    counts, st, out)
        shown += 1

    if shown > 1:
        out.append("")
        out.append(st.cyan("━" * 96))
        out.append(st.bold("  overview") + st.dim("   Avg EM across the datasets each pair ran"))
        out.append(st.cyan("━" * 96))
        head = st.pad("  pair", 26)
        for sym, _ in (SHORT[r] for r in MODEL_ROLES):
            head += st.pad(sym, 11, right=False)
        head += st.pad("vs RL(W_I)", 13, right=False) + st.pad("vs RL(W_B)", 13, right=False)
        out.append(st.dim(head))
        out.append(st.dim("  " + "─" * 94))
        wins = losses = 0
        for pair in PAIRS:
            if pair.pair_id not in results or (args.pair and pair.pair_id != args.pair):
                continue
            rd = results[pair.pair_id]
            line = st.pad(f"  {pair.pair_id}", 26)
            a = {r: avg(rd.get(r, {})) for r in MODEL_ROLES}
            for r in MODEL_ROLES:
                line += st.pad(fmt_em(a[r], st, best=(r == "shadow")), 11, right=False)
            for competitor in ("rl_on_instruct", "rl_on_base"):
                if a["shadow"] is None or a[competitor] is None:
                    line += st.pad(st.dim("  -  "), 13, right=False)
                    continue
                d = a["shadow"] - a[competitor]
                sd = se_diff(rd.get("shadow", {}), nmap.get(pair.pair_id, {}).get("shadow", {}),
                             rd.get(competitor, {}),
                             nmap.get(pair.pair_id, {}).get(competitor, {}))
                if competitor == "rl_on_instruct":
                    wins += d > 0
                    losses += d < 0
                txt = f"{d:+.4f}"
                if sd is not None and abs(d) < sd:
                    txt += "?"          # inside the noise floor
                line += st.pad((st.green if d > 0 else st.red)(txt), 13, right=False)
            out.append(line)
        out.append("")
        verdict = f"  shadow beats RL(W_I) on {wins}/{wins + losses} pair(s)"
        out.append(st.green(verdict) if wins > losses else st.yellow(verdict))
        out.append(st.dim("  ? marks a margin smaller than its own standard error"))

    if not shown:
        out.append("")
        out.append(st.yellow("  no results yet — run ./shadow_rl/run_all.sh --fast --yes"))
    else:
        total_q = sum(max(counts[d]) for d in counts) if counts else 0
        if total_q and total_q < 40000:
            out.append("")
            out.append(st.yellow("  note: ") + st.dim(
                f"subsampled run ({total_q} questions/role vs ~51,700 full). "
                "Margins below the sampling error are not real."))
        out.append("")
        out.append(st.dim(f"  full tables: shadow_rl/FINDINGS.md   ·   raw: {args.results}"))
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
