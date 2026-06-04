"""Build a *static* rollback candidate ("plasticized shadow position").

Given the rollback scores produced by ``analyze_lineage.py``, this tool pulls the
RLVR endpoint partially back toward the straight Base->RLVR chord, *only* inside
the highest-scoring groups, by removing a fraction (``rho``) of each selected
late-stage delta's component that is orthogonal to the chord.

This is still purely static: no data, no training, no evaluation.  The output is
a weight delta (and optionally full tensors) that a downstream experiment can
later graft and study.  By default only the modified tensors' deltas are saved,
so memory stays bounded to the selected subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grouping import assign_groups  # noqa: E402
from hf_checkpoint_io import CheckpointReader, common_floating_names  # noqa: E402

from safetensors.torch import save_file  # noqa: E402


def _read_rollback_csv(path: str) -> List[dict]:
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_groups(
    rollback_rows: List[dict],
    granularity: str,
    stages: List[str],
    top_fraction: float,
) -> Dict[str, Set[str]]:
    """Return ``{stage: set(group)}`` of the top-scoring groups per stage."""
    selected: Dict[str, Set[str]] = {s: set() for s in stages}
    for stage in stages:
        recs = [
            r for r in rollback_rows
            if r["granularity"] == granularity and r["stage"] == stage
        ]
        recs = [r for r in recs if _isfinite(r.get("score"))]
        recs.sort(key=lambda r: float(r["score"]), reverse=True)
        k = max(1, int(round(len(recs) * top_fraction))) if recs else 0
        for r in recs[:k]:
            selected[stage].add(r["group"])
    return selected


def _isfinite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _residual(v: np.ndarray, chord: np.ndarray, eps: float) -> np.ndarray:
    """Component of ``v`` orthogonal to ``chord`` (per-tensor)."""
    cc = float(np.dot(chord, chord))
    if cc <= eps:
        return v.copy()
    coef = float(np.dot(v, chord)) / cc
    return v - coef * chord


def main() -> None:
    ap = argparse.ArgumentParser(description="Build static rollback candidate (weights only).")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rb = cfg.get("rollback", {})
    if not rb.get("enabled", False):
        print("rollback.enabled is false — nothing to build.")
        return

    out_dir = cfg.get("output_dir", "outputs/static_vector_field")
    shadow_dir = os.path.join(out_dir, "static_shadow")
    os.makedirs(shadow_dir, exist_ok=True)

    eps = float(cfg.get("eps", 1e-12))
    granularity = rb.get("granularity", "layer_module")
    stages = rb.get("stages", ["dpo", "rlvr"])
    top_fraction = float(rb.get("top_fraction", 0.2))
    rho = float(rb.get("rho_max", 0.5))
    save_full = bool(rb.get("save_full_checkpoint", False))
    save_delta = bool(rb.get("save_delta_only", True))
    depth_bins = cfg.get("depth_bins", {})

    rollback_rows = _read_rollback_csv(os.path.join(out_dir, "rollback_scores.csv"))
    selected = select_groups(rollback_rows, granularity, stages, top_fraction)
    print("Selected groups per stage:")
    for s, groups in selected.items():
        print(f"  {s}: {len(groups)} groups")

    # Build readers.
    lineage = {item["name"]: item["path"] for item in cfg["lineage"]}
    readers = {s: CheckpointReader(s, lineage[s]) for s in ["base", "sft", "dpo", "rlvr"]}
    names, _ = common_floating_names(
        list(readers.values()),
        ignore_patterns=cfg.get("ignore_regex", []),
        include_non_floating=cfg.get("include_non_floating", False),
    )

    delta_tensors: Dict[str, torch.Tensor] = {}
    full_tensors: Dict[str, torch.Tensor] = {}
    n_modified = 0
    manifest: Dict[str, dict] = {}

    for name in names:
        gmap = dict(assign_groups(name, [granularity], depth_bins))
        group = gmap.get(granularity)
        # Which stages touch this tensor's group?
        active = [s for s in stages if group is not None and group in selected[s]]
        if not active:
            continue

        base = readers["base"].load_flat_f64(name)
        sft = readers["sft"].load_flat_f64(name)
        dpo = readers["dpo"].load_flat_f64(name)
        rlvr = readers["rlvr"].load_flat_f64(name)
        chord = rlvr - base

        delta = np.zeros_like(rlvr)
        for stage in active:
            if stage == "rlvr":
                v = rlvr - dpo
            elif stage == "dpo":
                v = dpo - sft
            else:
                continue
            delta -= rho * _residual(v, chord, eps)

        new_rlvr = rlvr + delta
        shape = readers["rlvr"].shape(name)
        n_modified += 1
        manifest[name] = {
            "group": group,
            "stages": active,
            "rho": rho,
            "delta_l2": float(np.linalg.norm(delta)),
            "rel_change": float(np.linalg.norm(delta) / (np.linalg.norm(rlvr) + eps)),
        }
        if save_delta:
            delta_tensors[name] = torch.from_numpy(
                delta.reshape(shape).astype(np.float32)
            )
        if save_full:
            full_tensors[name] = torch.from_numpy(
                new_rlvr.reshape(shape).astype(np.float32)
            )
        del base, sft, dpo, rlvr, chord, delta, new_rlvr

    print(f"Modified {n_modified} tensors.")

    if save_delta and delta_tensors:
        dpath = os.path.join(shadow_dir, "rollback_delta.safetensors")
        save_file(delta_tensors, dpath)
        print(f"  wrote {dpath}")
    if save_full and full_tensors:
        fpath = os.path.join(shadow_dir, "rollback_full.safetensors")
        save_file(full_tensors, fpath)
        print(f"  wrote {fpath}")

    with open(os.path.join(shadow_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "target": rb.get("target", "rlvr"),
                "strategy": rb.get("strategy", "residual_to_chord"),
                "granularity": granularity,
                "stages": stages,
                "rho": rho,
                "top_fraction": top_fraction,
                "n_modified_tensors": n_modified,
                "tensors": manifest,
            },
            f,
            indent=2,
        )
    print(f"  wrote {os.path.join(shadow_dir, 'manifest.json')}")
    print("Note: this is a static weight-space candidate only — not evaluated.")


if __name__ == "__main__":
    main()
