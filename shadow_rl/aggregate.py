#!/usr/bin/env python3
"""Turn results.csv into FINDINGS.md.

Reports, for every pair with results, the four Avg numbers and the two margins
that the experiment is actually about:

    shadow - rl_on_instruct     does grafting beat direct RL on the instruct model?
    shadow - rl_on_base         does grafting beat direct RL on the base model?

Also diffs `rl_on_base` and `rl_on_instruct` against the published numbers, since
a merged-model result means nothing if the harness cannot reproduce those first.

    python shadow_rl/aggregate.py --results shadow_rl/results.csv --out shadow_rl/FINDINGS.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs import (  # noqa: E402
    DATASETS, IN_DOMAIN, MODEL_ROLES, PAIRS, PAIRS_BY_ID, reference_avg, reference_for,
)

ROLE_LABEL = {
    "base_baseline": "W_B (untuned base)",
    "instruct_baseline": "W_I (untuned instruct)",
    "rl_on_instruct": "RL(W_I) (direct RL on instruct)",
    "rl_on_base": "RL(W_B) (direct RL on base)",
    "shadow": "W_shadow (ours)",
}

# Reproduction is called good when every dataset lands within this of the paper.
REPRO_TOL = 0.05


def fmt(x: Optional[float], places: int = 3, sign: bool = False) -> str:
    if x is None:
        return "--"
    return f"{x:+.{places}f}" if sign else f"{x:.{places}f}"


def load(path: str):
    if not os.path.exists(path):
        sys.exit(f"[fail] {path} not found. Run shadow_rl/evaluate.py first.")
    # results[pair_id][role][dataset] = em
    results: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    # counts[dataset] = set of question counts seen, to detect sampled runs
    counts: Dict[str, set] = defaultdict(set)
    # nmap[pair][role][dataset] = questions scored, needed for standard errors
    nmap: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            with_search = str(row["with_search"]).lower() == "true"
            tag = "search" if with_search else "nosearch"
            pair_id = f"{row['algo']}-{tag}-{row['size']}-{row['version']}"
            results[pair_id][row["model_role"]][row["dataset"]] = float(row["em"])
            if row.get("n_questions"):
                n = int(row["n_questions"])
                counts[row["dataset"]].add(n)
                nmap[pair_id][row["model_role"]][row["dataset"]] = n
    return results, counts, nmap


def se_avg(per_dataset: Dict[str, float], per_n: Dict[str, int]) -> Optional[float]:
    """Standard error of the dataset-averaged EM.

    EM on one dataset is a mean of Bernoulli trials, so its variance is
    p(1-p)/n; the equally-weighted average over k datasets has variance
    sum(var_i)/k^2. Sampling error only -- it says nothing about how well the
    subsample represents the full test set beyond its size.
    """
    present = [d for d in DATASETS if d in per_dataset and per_n.get(d)]
    if not present:
        return None
    var = sum(per_dataset[d] * (1 - per_dataset[d]) / per_n[d] for d in present)
    return (var ** 0.5) / len(present)


def se_diff(a_em, a_n, b_em, b_n) -> Optional[float]:
    """Conservative standard error on the difference of two averages.

    Both models are scored on the *same* sampled questions, so the paired error
    is smaller than this; results.csv keeps only per-dataset means, not
    per-question outcomes, so the independent-samples bound is what we can
    compute. Treat it as an upper bound on the noise.
    """
    sa, sb = se_avg(a_em, a_n), se_avg(b_em, b_n)
    if sa is None or sb is None:
        return None
    return (sa ** 2 + sb ** 2) ** 0.5


def avg(per_dataset: Dict[str, float]) -> Optional[float]:
    """Average over whichever datasets are present, or None if none are.

    Deliberately not "all seven or nothing": a run may skip datasets on purpose,
    and a five-dataset average is still meaningful as long as every role is
    averaged over the same five. Completeness is surfaced separately via
    is_complete() so partial numbers are marked rather than silently compared.
    """
    present = [d for d in DATASETS if d in per_dataset]
    return sum(per_dataset[d] for d in present) / len(present) if present else None


def is_complete(per_dataset: Dict[str, float]) -> bool:
    return all(d in per_dataset for d in DATASETS)


def mark(per_dataset: Dict[str, float], value: Optional[float]) -> str:
    """Format an average, flagging it when it does not cover all seven sets."""
    if value is None:
        return "--"
    return f"{value:.3f}" if is_complete(per_dataset) else f"{value:.3f}*"


def partial_avg(per_dataset: Dict[str, float]) -> Optional[float]:
    return sum(per_dataset.values()) / len(per_dataset) if per_dataset else None


def render(results, counts, out_path: str) -> None:
    lines = [
        "# Shadow-FT weight grafting on matched RL checkpoint pairs",
        "",
        "`W_shadow = W_I + (RL(W_B) - W_B)` — the RL update learned on the base model,",
        "grafted onto the instruct backbone. No training; weight arithmetic plus evaluation.",
        "",
        "Metric is Exact Match over seven QA sets "
        f"({', '.join(sorted(IN_DOMAIN))} in-domain; the rest out-of-domain),",
        "scored with `verl/utils/reward_score/qa_em.py` from the official Search-R1 harness.",
        "",
    ]

    evaluated = [p for p in PAIRS if p.pair_id in results]
    if not evaluated:
        lines += [
            "## Status: no results yet",
            "",
            "`results.csv` has no rows, so there is nothing to report. The pipeline is in",
            "place and tested; what remains is GPU time. This file is regenerated by",
            "`shadow_rl/aggregate.py` every time `results.csv` grows.",
            "",
            "Order of work (see `shadow_rl/README.md`):",
            "",
            "1. Merge the 3B `R1-*` pair and confirm it loads and generates coherent text.",
            "2. Reproduce `rl_on_base` = 0.229 and `rl_on_instruct` = 0.224 on that pair.",
            "   **Do not proceed until the harness agrees** — a merged-model number is",
            "   uninterpretable if the two released checkpoints do not reproduce.",
            "3. Evaluate `W_shadow`. First real result.",
            "4. Repeat for the 7B `R1-*` pair (reference 0.276 / 0.271).",
            "5. Stand up BM25 retrieval and move to the GRPO pairs, 3B first.",
            "",
            "### Pairs awaiting evaluation",
            "",
            "| pair | algo | size | search | published RL(W_B) | published RL(W_I) |",
            "|---|---|---|---|---|---|",
        ]
        for p in PAIRS:
            lines.append(
                f"| `{p.pair_id}` | {p.algo} | {p.size} | "
                f"{'yes' if p.with_search else 'no'} | "
                f"{fmt(reference_avg(p, 'rl_on_base'))} | "
                f"{fmt(reference_avg(p, 'rl_on_instruct'))} |"
            )
        lines += [
            "",
            "Published averages are from the v0.1 paper; v0.2/v0.3 rows are ballpark only.",
            "",
        ]
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines))
        print(f"[ok] wrote {out_path}  (no results yet)")
        return

    missing = [p.pair_id for p in PAIRS if p.pair_id not in results]

    # ------------------------------------------------------------------ verdict
    # Lead with the answer. The margin against RL(W_I) is the whole experiment:
    # RL(W_B) is what the update was learned on, so beating it is expected;
    # beating direct RL on the instruct model is the claim.
    wins, losses, pending = [], [], []
    for pair in evaluated:
        roles = results[pair.pair_id]
        a_s, a_i = avg(roles.get("shadow", {})), avg(roles.get("rl_on_instruct", {}))
        if a_s is None or a_i is None:
            pending.append(pair.pair_id)
        elif a_s > a_i:
            wins.append((pair.pair_id, a_s - a_i))
        else:
            losses.append((pair.pair_id, a_s - a_i))

    lines += ["## Verdict", ""]
    if wins or losses:
        n = len(wins) + len(losses)
        lines.append(f"`W_shadow` beats `RL(W_I)` on **{len(wins)} of {n}** evaluated pair(s).")
        lines.append("")
        for pid, d in sorted(wins, key=lambda kv: -kv[1]):
            lines.append(f"- ✅ `{pid}` — **{fmt(d, sign=True)}** over direct RL on instruct")
        for pid, d in sorted(losses, key=lambda kv: kv[1]):
            lines.append(f"- ❌ `{pid}` — {fmt(d, sign=True)} against direct RL on instruct")
        lines.append("")
    if pending:
        lines += [f"Incomplete (missing `shadow` or `rl_on_instruct`): "
                  + ", ".join(f"`{p}`" for p in pending), ""]
    lines += [
        "Beating `RL(W_B)` is the weaker claim — that is the run the update was",
        "learned from. Beating `RL(W_I)` is the result the experiment is after.",
        "",
    ]

    # ---------------------------------------------------------------- summary
    lines += [
        "## Summary",
        "",
        "| pair | algo | size | search | W_B | W_I | RL(W_I) | RL(W_B) | **W_shadow** "
        "| shadow − RL(W_I) | shadow − RL(W_B) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pair in evaluated:
        roles = results[pair.pair_id]
        a = {r: avg(roles.get(r, {})) for r in MODEL_ROLES}
        d_inst = (a["shadow"] - a["rl_on_instruct"]
                  if a["shadow"] is not None and a["rl_on_instruct"] is not None else None)
        d_base = (a["shadow"] - a["rl_on_base"]
                  if a["shadow"] is not None and a["rl_on_base"] is not None else None)
        lines.append(
            f"| `{pair.pair_id}` | {pair.algo} | {pair.size} | "
            f"{'yes' if pair.with_search else 'no'} | "
            f"{mark(roles.get('base_baseline', {}), a['base_baseline'])} | "
            f"{mark(roles.get('instruct_baseline', {}), a['instruct_baseline'])} | "
            f"{mark(roles.get('rl_on_instruct', {}), a['rl_on_instruct'])} | "
            f"{mark(roles.get('rl_on_base', {}), a['rl_on_base'])} | "
            f"**{mark(roles.get('shadow', {}), a['shadow'])}** | "
            f"{fmt(d_inst, sign=True)} | {fmt(d_base, sign=True)} |"
        )
    lines += ["", "Avg is over the datasets evaluated; `*` marks fewer than all seven, "
              "`--` means the role has no results yet.", ""]

    # ------------------------------------------------------- harness validation
    lines += [
        "## Harness validation",
        "",
        "The merged-model number is only meaningful if the harness reproduces the published",
        "numbers for the two released checkpoints. Δ is ours minus published; the v0.1 paper",
        "is the source, so v0.2/v0.3 rows are expected to agree only in the ballpark.",
        "",
        "| pair | role | ours | published | Δ | worst dataset Δ |",
        "|---|---|---|---|---|---|",
    ]
    for pair in evaluated:
        for role in ("rl_on_base", "rl_on_instruct"):
            got = results[pair.pair_id].get(role, {})
            ours = avg(got)
            ref = reference_avg(pair, role, subset=set(got))
            if ours is None or ref is None:
                continue
            ref_per = reference_for(pair, role)
            worst_ds, worst_d = max(
                ((d, got[d] - ref_per[d]) for d in DATASETS if d in got and d in ref_per),
                key=lambda kv: abs(kv[1]),
                default=(None, None),
            )
            worst = f"{worst_ds} {fmt(worst_d, sign=True)}" if worst_ds else "--"
            lines.append(
                f"| `{pair.pair_id}` | {role} | {fmt(ours)} | {fmt(ref)} | "
                f"{fmt(ours - ref, sign=True)} | {worst} |"
            )
    lines += [
        "",
        f"A pair is safe to draw conclusions from when every dataset is within ~{REPRO_TOL:.2f}",
        "of the published number. The `instruct_baseline` row has no comparable published",
        "number: the paper's *direct inference* baseline uses a plain QA prompt, while we",
        "prompt every role with the same Search-R1 template so the four are comparable.",
        "",
    ]

    # ------------------------------------------------------------- per pair
    lines += ["## Per pair, per dataset", ""]
    for pair in evaluated:
        roles = results[pair.pair_id]
        lines += [
            f"### `{pair.pair_id}`",
            "",
            f"- {pair.algo.upper()}, {pair.size}, "
            f"{'with search' if pair.with_search else 'no search'}, {pair.version}",
            f"- base-side: `PeterJinGo/{pair.rl_base}`",
            f"- instruct-side: `PeterJinGo/{pair.rl_instruct}`",
            "",
            "| model | " + " | ".join(DATASETS) + " | **Avg** |",
            "|---" * (len(DATASETS) + 2) + "|",
        ]
        for role in MODEL_ROLES:
            per = roles.get(role, {})
            if not per:
                continue
            cells = " | ".join(fmt(per.get(d)) for d in DATASETS)
            a = avg(per)
            shown = mark(per, a)
            label = ROLE_LABEL[role]
            if role == "shadow":
                label = f"**{label}**"
                shown = f"**{shown}**"
            lines.append(f"| {label} | {cells} | {shown} |")

        a_shadow = avg(roles.get("shadow", {}))
        a_inst = avg(roles.get("rl_on_instruct", {}))
        a_base = avg(roles.get("rl_on_base", {}))
        lines.append("")
        if a_shadow is None:
            lines += ["> `W_shadow` not evaluated yet for this pair.", ""]
        else:
            verdict = []
            if a_inst is not None:
                d = a_shadow - a_inst
                verdict.append(
                    f"- `shadow − rl_on_instruct` = **{fmt(d, sign=True)}** "
                    f"({'beats' if d > 0 else 'does not beat'} direct RL on instruct)"
                )
            if a_base is not None:
                d = a_shadow - a_base
                verdict.append(
                    f"- `shadow − rl_on_base` = **{fmt(d, sign=True)}** "
                    f"({'beats' if d > 0 else 'does not beat'} direct RL on base)"
                )
            lines += verdict + [""]

        stats_path = os.path.join(os.path.dirname(out_path), "merged", pair.pair_id,
                                  "shadow_merge_stats.json")
        if os.path.exists(stats_path):
            import json
            s = json.load(open(stats_path))
            flag = "" if s.get("sigma_in_range") else "  **outside the paper's 0.003–0.042 range**"
            lines += [
                f"- sigma (W_B vs W_I) = {s['sigma']:.5f}{flag}",
                f"- ||RL(W_B) − W_B|| / ||W_B|| = {s['rel_delta_base']:.5f}",
            ]
            if "rel_delta_instruct" in s:
                lines.append(f"- ||RL(W_I) − W_I|| / ||W_I|| = {s['rel_delta_instruct']:.5f}")
            lines.append("")

    if missing:
        lines += [
            "## Not yet evaluated",
            "",
            *(f"- `{m}`" for m in missing),
            "",
        ]

    lines += ["## Notes", ""]
    if counts:
        inconsistent = {d: sorted(v) for d, v in counts.items() if len(v) > 1}
        total = sum(max(v) for v in counts.values())
        detail = ", ".join(f"{d} {max(counts[d])}" for d in DATASETS if d in counts)
        lines.append(f"- Questions evaluated per role: {detail} (total {total}).")
        if total < 40000:
            lines.append(
                "  This is a **subsample** of the full test sets (~51,700 questions). EM on a"
                "  subsample is noisier than the published numbers and the two are not"
                "  strictly comparable; margins smaller than the sampling error should not be"
                "  read as real.")
        if inconsistent:
            lines.append(
                "- **WARNING: roles were evaluated on different numbers of questions** — "
                + "; ".join(f"{d}: {v}" for d, v in inconsistent.items())
                + ". The comparison between roles is not valid until these match.")
    lines += [
        "- `*` on an Avg marks an average over fewer than all seven datasets. Published",
        "  averages in the validation table are restricted to the same subset, so the Δ",
        "  stays a like-for-like comparison.",
        "- Retrieval for the `SearchR1-*` pairs uses `PeterJinGo/wiki-18-bm25-index` for speed.",
        "  The paper uses a dense E5 index, so absolute numbers on the search pairs sit below",
        "  the published ones; the four roles share the index, so the comparison stays fair.",
        "- Decoding is greedy, so EM is reproducible across runs.",
        "",
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[ok] wrote {out_path}  ({len(evaluated)}/{len(PAIRS)} pairs with results)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="shadow_rl/results.csv")
    ap.add_argument("--out", default="shadow_rl/FINDINGS.md")
    args = ap.parse_args()
    results, counts, _ = load(args.results)
    render(results, counts, args.out)


if __name__ == "__main__":
    main()
