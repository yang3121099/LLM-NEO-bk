"""Static post-training vector-field audit over a Base->SFT->DPO->RLVR lineage.

This is a *pure static weight analysis*: it never loads downstream data, never
trains, never evaluates, and never computes a loss or gradient.  It streams the
four checkpoints' shared floating tensors, folds their stage deltas into float64
accumulators per group, then materialises the metric CSVs + a markdown summary.

Usage::

    python analyze_lineage.py --config configs/llama31_8b_lineage.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

# Allow running both as ``python src/analyze_lineage.py`` and as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accumulators import AccumulatorStore, CHECKPOINTS  # noqa: E402
from grouping import ALL_GRANULARITIES, assign_groups  # noqa: E402
from hf_checkpoint_io import (  # noqa: E402
    CheckpointReader,
    common_floating_names,
    is_tied_name,
)
import metrics as M  # noqa: E402

REQUIRED_STAGES = ["base", "sft", "dpo", "rlvr"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_readers(cfg: dict) -> Dict[str, CheckpointReader]:
    lineage = {item["name"]: item["path"] for item in cfg["lineage"]}
    missing = [s for s in REQUIRED_STAGES if s not in lineage]
    if missing:
        raise ValueError(f"config lineage is missing stages: {missing}")
    print("Indexing checkpoints (no full-model load)...")
    readers = {}
    for stage in REQUIRED_STAGES:
        print(f"  [{stage}] {lineage[stage]}")
        readers[stage] = CheckpointReader(stage, lineage[stage])
    return readers


def stream_accumulate(
    cfg: dict,
    readers: Dict[str, CheckpointReader],
    names: List[str],
) -> AccumulatorStore:
    granularities = [g for g in cfg.get("granularities", ALL_GRANULARITIES)
                     if g in ALL_GRANULARITIES]
    depth_bins = cfg.get("depth_bins", {})
    eps = float(cfg.get("eps", 1e-12))
    topk_cfg = cfg.get("topk", {"enabled": False})

    use_cuda, torch_dtype = _resolve_device(cfg)
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        import torch
        print(f"GPU path enabled: device=cuda, reduce dtype={torch_dtype}")

    store = AccumulatorStore()
    n_total = len(names)
    t0 = time.time()
    for i, name in enumerate(names):
        tied = is_tied_name(name)
        groups = assign_groups(name, granularities, depth_bins)

        if use_cuda:
            # GPU path: per-tensor reductions on-device, float64 accumulation on host.
            w = {c: readers[c].load_flat_torch(name, device, torch_dtype)
                 for c in CHECKPOINTS}
            vecs = {
                "sft": w["sft"] - w["base"],
                "dpo": w["dpo"] - w["sft"],
                "rlvr": w["rlvr"] - w["dpo"],
                "post": w["rlvr"] - w["base"],
            }
            for gran, group in groups:
                acc = store.get(gran, group)
                acc.update_torch(w, vecs, eps=eps, topk_cfg=topk_cfg)
                if tied:
                    acc.has_tied = True
        else:
            # CPU path: flat float64 numpy vectors.
            w = {c: readers[c].load_flat_f64(name) for c in CHECKPOINTS}
            vecs = {
                "sft": w["sft"] - w["base"],
                "dpo": w["dpo"] - w["sft"],
                "rlvr": w["rlvr"] - w["dpo"],
                "post": w["rlvr"] - w["base"],
            }
            for gran, group in groups:
                acc = store.get(gran, group)
                acc.update(w, vecs, eps=eps, topk_cfg=topk_cfg)
                if tied:
                    acc.has_tied = True

        # Release immediately.
        del w, vecs

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  processed {i + 1}/{n_total} tensors ({rate:.1f}/s)", flush=True)
    return store


def _resolve_device(cfg: dict):
    """Return ``(use_cuda, torch_dtype)`` honouring the config + hardware.

    Falls back to CPU (with a warning) when ``device: cuda`` is requested but no
    GPU is visible.  ``reduce_dtype`` controls the on-device reduction precision
    (default float32 — much faster on consumer GPUs than float64).
    """
    device = str(cfg.get("device", "cpu")).lower()
    if not device.startswith("cuda"):
        return False, None
    import torch

    if not torch.cuda.is_available():
        print("  [warn] device: cuda requested but no GPU visible — using CPU.")
        return False, None
    reduce_dtype = str(cfg.get("reduce_dtype", "float32")).lower()
    torch_dtype = torch.float64 if reduce_dtype in ("float64", "f64", "double") \
        else torch.float32
    return True, torch_dtype


def write_summary_md(
    path: str,
    cfg: dict,
    tables: Dict[str, List[dict]],
    metrics: dict,
    notes: Dict[str, str],
    tied_groups: List[str],
) -> None:
    def global_row(rows, **filt):
        for r in rows:
            if r.get("granularity") == "global" and all(r.get(k) == v for k, v in filt.items()):
                yield r

    lines: List[str] = []
    lines.append("# Static Post-Training Vector-Field Audit\n")
    lines.append("Lineage: **Base → SFT → DPO → RLVR**\n")
    lines.append(
        "> **Caveat:** this is a *static weight-space* analysis only. It reads "
        "checkpoint parameters and computes geometric relationships between stage "
        "deltas. It does **not** train, evaluate, or measure any downstream task "
        "performance. Nothing here is a performance claim.\n"
    )

    # 1. Global pairwise sigma.
    lines.append("## 1. Global pairwise σ (relative gap ratio)\n")
    lines.append("| pair | sigma |")
    lines.append("|------|-------|")
    for r in global_row(tables["pairwise_sigma"]):
        lines.append(f"| {r['pair']} | {r['sigma']:.6f} |")
    lines.append("")

    # 2. Global stage norms.
    lines.append("## 2. Global stage norms\n")
    lines.append("| stage | norm | rel_l2 | stage_fraction |")
    lines.append("|-------|------|--------|----------------|")
    for r in global_row(tables["stage_norm_budget"]):
        lines.append(
            f"| {r['stage']} | {r['norm']:.4f} | {r['rel_l2']:.6f} | {r['stage_fraction']:.4f} |"
        )
    lines.append("")

    # 3. Global cosine matrix.
    lines.append("## 3. Global stage-delta cosine\n")
    lines.append("| pair | cosine |")
    lines.append("|------|--------|")
    for r in global_row(tables["pairwise_cosine"]):
        lines.append(f"| {r['pair']} | {r['cosine']:.4f} |")
    lines.append("")

    # 4. Global straightness / cancellation / curvature.
    gm = metrics.get(("global", "global:all"), {})
    lines.append("## 4. Global path geometry\n")
    if gm:
        lines.append(f"- **straightness** (L_chord / L_path): {gm['straightness']:.4f}")
        lines.append(f"- **cancellation** (1 - straightness): {gm['cancellation']:.4f}")
        lines.append(f"- **curvature** (φ_sft→dpo + φ_dpo→rlvr): {gm['curvature']:.4f} rad")
        lines.append(f"- **effective_rank** of stage subspace: {gm['effective_rank']:.4f}")
        lines.append(f"- **pc1_energy**: {gm['pc1_energy']:.4f}")
    lines.append("")

    # 5. Top 10 groups by curvature.
    lines.append("## 5. Top 10 groups by curvature\n")
    curv = sorted(
        ({"granularity": g, "group": grp, "curvature": m["curvature"]}
         for (g, grp), m in metrics.items()),
        key=lambda r: r["curvature"], reverse=True,
    )[:10]
    lines.append("| granularity | group | curvature (rad) |")
    lines.append("|-------------|-------|-----------------|")
    for r in curv:
        lines.append(f"| {r['granularity']} | {r['group']} | {r['curvature']:.4f} |")
    lines.append("")

    # 6. Top 10 groups by residual-to-chord (rlvr stage).
    lines.append("## 6. Top 10 groups by residual-to-chord (RLVR stage)\n")
    rtc = sorted(
        (r for r in tables["chord_projection"] if r["stage"] == "rlvr"),
        key=lambda r: r["residual_to_chord"], reverse=True,
    )[:10]
    lines.append("| granularity | group | residual_to_chord |")
    lines.append("|-------------|-------|-------------------|")
    for r in rtc:
        lines.append(f"| {r['granularity']} | {r['group']} | {r['residual_to_chord']:.4f} |")
    lines.append("")

    # 7. Top 10 rollback candidates.
    lines.append("## 7. Top 10 static rollback candidates\n")
    lines.append("| granularity | group | stage | score |")
    lines.append("|-------------|-------|-------|-------|")
    for r in tables["top_rollback_candidates"][:10]:
        lines.append(
            f"| {r['granularity']} | {r['group']} | {r['stage']} | {r['score']:.3f} |"
        )
    lines.append("")

    # 8. Tied-weight note + skipped tensors.
    lines.append("## 8. Notes\n")
    if tied_groups:
        lines.append(
            "- Tied embedding / lm_head tensors were folded into these groups "
            "(no special merging applied): " + ", ".join(sorted(set(tied_groups))) + "."
        )
    skipped = {}
    for n, reason in notes.items():
        skipped[reason] = skipped.get(reason, 0) + 1
    if skipped:
        lines.append("- Skipped tensors by reason:")
        for reason, c in sorted(skipped.items()):
            lines.append(f"  - {reason}: {c}")
    lines.append("")
    lines.append(
        "_Interpretation caveat (repeated): curved / residual / late-stage-specific "
        "deltas flagged here are candidates for further study; they are not, on their "
        "own, evidence of degraded downstream behaviour._\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Static vector-field audit (weights only).")
    ap.add_argument("--config", required=True, help="Path to lineage YAML config.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N shared tensors (debug).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("output_dir", "outputs/static_vector_field")
    os.makedirs(out_dir, exist_ok=True)

    readers = build_readers(cfg)
    names, notes = common_floating_names(
        list(readers.values()),
        ignore_patterns=cfg.get("ignore_regex", []),
        include_non_floating=cfg.get("include_non_floating", False),
    )
    print(f"Shared floating tensors: {len(names)} (skipped {len(notes)})")
    if args.limit:
        names = names[: args.limit]
        print(f"  [debug] limited to {len(names)} tensors")

    store = stream_accumulate(cfg, readers, names)
    print(f"Built {len(store)} group accumulators.")

    eps = float(cfg.get("eps", 1e-12))
    tables, metrics = M.build_all_tables(store, eps)

    # Track which groups received tied weights for the summary note.
    tied_groups = [grp for (gran, grp), acc in store.items() if acc.has_tied]

    # Write CSVs.
    csv_dir = out_dir
    schemas = {
        "pairwise_sigma": ["granularity", "group", "pair", "sigma", "numel"],
        "subspace_rank": ["granularity", "group", "lambda1", "lambda2", "lambda3",
                           "pc1_energy", "effective_rank"],
        "rollback_scores": ["granularity", "group", "stage", "score",
                            "residual_to_chord", "orth_residual_ratio", "curvature",
                            "cancellation", "stage_fraction", "chord_align"],
        "top_rollback_candidates": ["granularity", "group", "stage", "score",
                                    "residual_to_chord", "orth_residual_ratio",
                                    "curvature", "cancellation", "stage_fraction",
                                    "chord_align"],
    }
    for stem, rows in tables.items():
        path = os.path.join(csv_dir, f"{stem}.csv")
        M.write_csv(path, rows, schemas.get(stem))
        print(f"  wrote {path} ({len(rows)} rows)")

    write_summary_md(
        os.path.join(out_dir, "summary.md"), cfg, tables, metrics, notes, tied_groups
    )
    print(f"  wrote {os.path.join(out_dir, 'summary.md')}")

    # Run metadata for reproducibility.
    meta = {
        "n_shared_tensors": len(names),
        "n_skipped": len(notes),
        "n_groups": len(store),
        "tied_groups": sorted(set(tied_groups)),
        "config": cfg,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("Done.")


if __name__ == "__main__":
    main()
