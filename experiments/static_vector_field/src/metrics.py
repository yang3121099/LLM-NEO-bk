"""Transform float64 accumulators into the audit's metric tables.

Every function here is pure (numpy + math) and operates on
:class:`~accumulators.Accumulator` objects, so the whole metric layer is unit
testable without any model checkpoints.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from accumulators import (
    Accumulator,
    AccumulatorStore,
    SIGMA_PAIRS,
    STAGE_PAIRS,
    STAGE_PREV,
)

# Stages that represent a real "step" along the path (post is the chord).
STEP_STAGES = ("sft", "dpo", "rlvr")
# Late stages eligible for static rollback.
LATE_STAGES = ("dpo", "rlvr")

EPS = 1e-12


def _norm(acc: Accumulator, stage: str) -> float:
    return math.sqrt(max(acc.norm_sq[stage], 0.0))


def _cos(acc: Accumulator, a: str, b: str, eps: float) -> float:
    if (a, b) in acc.dot:
        dot = acc.dot[(a, b)]
    elif (b, a) in acc.dot:
        dot = acc.dot[(b, a)]
    elif a == b:
        dot = acc.norm_sq[a]
    else:
        raise KeyError((a, b))
    na, nb = _norm(acc, a), _norm(acc, b)
    return dot / (na * nb + eps)


def _dot(acc: Accumulator, a: str, b: str) -> float:
    if (a, b) in acc.dot:
        return acc.dot[(a, b)]
    if (b, a) in acc.dot:
        return acc.dot[(b, a)]
    if a == b:
        return acc.norm_sq[a]
    raise KeyError((a, b))


# --------------------------------------------------------------------------- #
# Per-group metric bundle
# --------------------------------------------------------------------------- #
def compute_group_metrics(acc: Accumulator, eps: float = EPS) -> Dict[str, float]:
    """Return a flat dict of every scalar metric for one group."""
    m: Dict[str, float] = {"numel": acc.numel, "tensor_count": acc.tensor_count}

    # Norms.
    norms = {s: _norm(acc, s) for s in ("sft", "dpo", "rlvr", "post")}
    for s, v in norms.items():
        m[f"norm_{s}"] = v

    # Path geometry.
    l_path = norms["sft"] + norms["dpo"] + norms["rlvr"]
    l_chord = norms["post"]
    straightness = l_chord / (l_path + eps)
    m["l_path"] = l_path
    m["l_chord"] = l_chord
    m["straightness"] = straightness
    m["cancellation"] = 1.0 - straightness

    phi_sd = math.acos(min(1.0, max(-1.0, _cos(acc, "sft", "dpo", eps))))
    phi_dr = math.acos(min(1.0, max(-1.0, _cos(acc, "dpo", "rlvr", eps))))
    m["phi_sft_dpo"] = phi_sd
    m["phi_dpo_rlvr"] = phi_dr
    m["curvature"] = phi_sd + phi_dr

    # Chord projection per step stage.
    chord_sq = acc.norm_sq["post"]
    for s in STEP_STAGES:
        d = _dot(acc, s, "post")
        m[f"progress_{s}"] = d / (chord_sq + eps)
        align = d / (norms[s] * norms["post"] + eps)
        m[f"chord_align_{s}"] = align
        m[f"residual_to_chord_{s}"] = math.sqrt(max(0.0, 1.0 - align * align))

    # Stage norm budget (fraction of total step energy).
    energy = acc.norm_sq["sft"] + acc.norm_sq["dpo"] + acc.norm_sq["rlvr"]
    for s in STEP_STAGES:
        m[f"stage_fraction_{s}"] = acc.norm_sq[s] / (energy + eps)
        prev_norm = math.sqrt(max(acc.theta_sq[STAGE_PREV[s]], 0.0))
        m[f"rel_l2_{s}"] = norms[s] / (prev_norm + eps)
        m[f"l1_{s}"] = acc.l1[s]
        m[f"linf_{s}"] = acc.linf[s]

    # Orthogonal residual energies.
    m["orth_residual_ratio_dpo"] = _dpo_residual_ratio(acc, eps)
    m["orth_residual_ratio_rlvr"] = _rlvr_residual_ratio(acc, eps)

    # Subspace rank.
    lam, pc1, erank = _subspace(acc, eps)
    m["lambda1"], m["lambda2"], m["lambda3"] = lam
    m["pc1_energy"] = pc1
    m["effective_rank"] = erank

    return m


def _dpo_residual_ratio(acc: Accumulator, eps: float) -> float:
    """||v_dpo - P_sft v_dpo||^2 / ||v_dpo||^2."""
    ndpo2 = acc.norm_sq["dpo"]
    nsft2 = acc.norm_sq["sft"]
    d = _dot(acc, "sft", "dpo")
    res = ndpo2 - (d * d) / (nsft2 + eps)
    return max(0.0, res) / (ndpo2 + eps)


def _rlvr_residual_ratio(acc: Accumulator, eps: float) -> float:
    """||v_rlvr - P_span(sft,dpo) v_rlvr||^2 / ||v_rlvr||^2 via 2x2 solve."""
    nr2 = acc.norm_sq["rlvr"]
    g = np.array(
        [
            [acc.norm_sq["sft"], _dot(acc, "sft", "dpo")],
            [_dot(acc, "sft", "dpo"), acc.norm_sq["dpo"]],
        ],
        dtype=np.float64,
    )
    b = np.array([_dot(acc, "sft", "rlvr"), _dot(acc, "dpo", "rlvr")], dtype=np.float64)
    ridge = 1e-12 * (np.trace(g) + 1.0)
    try:
        coef = np.linalg.solve(g + ridge * np.eye(2), b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(g, b, rcond=None)[0]
    explained = float(b @ coef)
    res = nr2 - explained
    return max(0.0, res) / (nr2 + eps)


def _subspace(acc: Accumulator, eps: float) -> Tuple[Tuple[float, float, float], float, float]:
    g = acc.gram()
    eig = np.linalg.eigvalsh(g)
    eig = np.clip(eig, 0.0, None)
    eig = np.sort(eig)[::-1]
    total = eig.sum()
    if total <= 0:
        return (0.0, 0.0, 0.0), 0.0, 0.0
    energy = eig / (total + eps)
    pc1 = float(energy[0])
    erank = float(math.exp(-np.sum(energy * np.log(energy + eps))))
    return (float(eig[0]), float(eig[1]), float(eig[2])), pc1, erank


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _all_metrics(store: AccumulatorStore, eps: float) -> Dict[Tuple[str, str], Dict[str, float]]:
    return {key: compute_group_metrics(acc, eps) for key, acc in store.items()}


def pairwise_sigma_rows(store: AccumulatorStore, eps: float) -> List[dict]:
    pair_to_stage = {
        ("base", "sft"): "sft",
        ("sft", "dpo"): "dpo",
        ("dpo", "rlvr"): "rlvr",
        ("base", "rlvr"): "post",
    }
    rows = []
    for (gran, group), acc in store.items():
        for (a, b) in SIGMA_PAIRS:
            stage = pair_to_stage[(a, b)]
            num = acc.l1[stage]
            den = acc.abs_sum[a] + acc.abs_sum[b] + eps
            rows.append(
                {
                    "granularity": gran,
                    "group": group,
                    "pair": f"{a}_{b}",
                    "sigma": num / den,
                    "numel": acc.numel,
                }
            )
    return rows


def stage_norm_budget_rows(metrics: Dict[Tuple[str, str], Dict[str, float]]) -> List[dict]:
    rows = []
    for (gran, group), m in metrics.items():
        for s in STEP_STAGES:
            rows.append(
                {
                    "granularity": gran,
                    "group": group,
                    "stage": s,
                    "norm": m[f"norm_{s}"],
                    "rel_l2": m[f"rel_l2_{s}"],
                    "stage_fraction": m[f"stage_fraction_{s}"],
                    "l1": m[f"l1_{s}"],
                    "linf": m[f"linf_{s}"],
                    "numel": m["numel"],
                }
            )
    return rows


def pairwise_cosine_rows(store: AccumulatorStore, eps: float) -> List[dict]:
    rows = []
    for (gran, group), acc in store.items():
        for (a, b) in STAGE_PAIRS:
            rows.append(
                {
                    "granularity": gran,
                    "group": group,
                    "pair": f"{a}_{b}",
                    "cosine": _cos(acc, a, b, eps),
                }
            )
    return rows


def path_geometry_rows(metrics: Dict[Tuple[str, str], Dict[str, float]]) -> List[dict]:
    rows = []
    for (gran, group), m in metrics.items():
        rows.append(
            {
                "granularity": gran,
                "group": group,
                "l_path": m["l_path"],
                "l_chord": m["l_chord"],
                "straightness": m["straightness"],
                "cancellation": m["cancellation"],
                "phi_sft_dpo": m["phi_sft_dpo"],
                "phi_dpo_rlvr": m["phi_dpo_rlvr"],
                "curvature": m["curvature"],
            }
        )
    return rows


def chord_projection_rows(metrics: Dict[Tuple[str, str], Dict[str, float]]) -> List[dict]:
    rows = []
    for (gran, group), m in metrics.items():
        for s in STEP_STAGES:
            rows.append(
                {
                    "granularity": gran,
                    "group": group,
                    "stage": s,
                    "progress": m[f"progress_{s}"],
                    "chord_align": m[f"chord_align_{s}"],
                    "residual_to_chord": m[f"residual_to_chord_{s}"],
                }
            )
    return rows


def subspace_rank_rows(metrics: Dict[Tuple[str, str], Dict[str, float]]) -> List[dict]:
    rows = []
    for (gran, group), m in metrics.items():
        rows.append(
            {
                "granularity": gran,
                "group": group,
                "lambda1": m["lambda1"],
                "lambda2": m["lambda2"],
                "lambda3": m["lambda3"],
                "pc1_energy": m["pc1_energy"],
                "effective_rank": m["effective_rank"],
            }
        )
    return rows


def orth_residual_rows(metrics: Dict[Tuple[str, str], Dict[str, float]]) -> List[dict]:
    rows = []
    for (gran, group), m in metrics.items():
        rows.append(
            {
                "granularity": gran,
                "group": group,
                "dpo_residual_ratio": m["orth_residual_ratio_dpo"],
                "rlvr_residual_ratio": m["orth_residual_ratio_rlvr"],
            }
        )
    return rows


def topk_overlap_rows(store: AccumulatorStore, eps: float) -> List[dict]:
    rows = []
    for (gran, group), acc in store.items():
        for (a, b) in STAGE_PAIRS:
            total = acc.topk_total[(a, b)]
            if total <= 0:
                continue
            rows.append(
                {
                    "granularity": gran,
                    "group": group,
                    "pair": f"{a}_{b}",
                    "topk_overlap_approx": acc.topk_inter[(a, b)] / (total + eps),
                    "approximate": True,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Rollback scoring (z-scored within each granularity)
# --------------------------------------------------------------------------- #
def _zscore(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0] * len(values)
    mean = float(finite.mean())
    std = float(finite.std())
    if std == 0 or not math.isfinite(std):
        return [0.0 if math.isfinite(v) else 0.0 for v in values]
    out = []
    for v in values:
        out.append((v - mean) / std if math.isfinite(v) else 0.0)
    return out


def rollback_score_rows(
    metrics: Dict[Tuple[str, str], Dict[str, float]]
) -> List[dict]:
    """Build rollback_scores rows, z-scored within each granularity."""
    # Collect raw per (granularity, group, stage) records.
    raw: List[dict] = []
    for (gran, group), m in metrics.items():
        for stage in LATE_STAGES:
            raw.append(
                {
                    "granularity": gran,
                    "group": group,
                    "stage": stage,
                    "residual_to_chord": m[f"residual_to_chord_{stage}"],
                    "orth_residual_ratio": m[f"orth_residual_ratio_{stage}"],
                    "curvature": m["curvature"],
                    "cancellation": m["cancellation"],
                    "stage_fraction": m[f"stage_fraction_{stage}"],
                    "chord_align": m[f"chord_align_{stage}"],
                }
            )

    # z-score each component within its granularity.
    components_pos = ["residual_to_chord", "orth_residual_ratio", "curvature",
                      "cancellation", "stage_fraction"]
    components_neg = ["chord_align"]

    by_gran: Dict[str, List[dict]] = {}
    for r in raw:
        by_gran.setdefault(r["granularity"], []).append(r)

    rows: List[dict] = []
    for gran, recs in by_gran.items():
        zcols: Dict[str, List[float]] = {}
        for comp in components_pos + components_neg:
            zcols[comp] = _zscore([r[comp] for r in recs])
        for i, r in enumerate(recs):
            score = 0.0
            for comp in components_pos:
                score += zcols[comp][i]
            for comp in components_neg:
                score -= zcols[comp][i]
            out = dict(r)
            out["score"] = score
            rows.append(out)
    return rows


def top_rollback_candidates(rollback_rows: List[dict], top_n: int = 50) -> List[dict]:
    ordered = sorted(
        rollback_rows,
        key=lambda r: (r["score"] if math.isfinite(r["score"]) else -math.inf),
        reverse=True,
    )
    return ordered[:top_n]


# --------------------------------------------------------------------------- #
# CSV writing
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        # Still emit a header-only file when we know the schema.
        with open(path, "w", newline="") as f:
            if fieldnames:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_all_tables(store: AccumulatorStore, eps: float = EPS) -> Dict[str, List[dict]]:
    """Compute every table and return ``{filename_stem: rows}``."""
    metrics = _all_metrics(store, eps)
    rollback = rollback_score_rows(metrics)
    return {
        "pairwise_sigma": pairwise_sigma_rows(store, eps),
        "stage_norm_budget": stage_norm_budget_rows(metrics),
        "pairwise_cosine": pairwise_cosine_rows(store, eps),
        "path_geometry": path_geometry_rows(metrics),
        "chord_projection": chord_projection_rows(metrics),
        "subspace_rank": subspace_rank_rows(metrics),
        "orth_residual": orth_residual_rows(metrics),
        "topk_overlap": topk_overlap_rows(store, eps),
        "rollback_scores": rollback,
        "top_rollback_candidates": top_rollback_candidates(rollback),
    }, metrics
