"""Render figures for the static vector-field audit from the metric CSVs.

Reads the CSVs produced by ``analyze_lineage.py`` and writes PNGs into
``<output_dir>/figures/``.  This module is intentionally read-only with respect
to the audit: it depends only on the CSV tables, not on the checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

STEP_STAGES = ["sft", "dpo", "rlvr"]
COSINE_PAIRS = ["sft_dpo", "sft_rlvr", "dpo_rlvr", "sft_post", "dpo_post", "rlvr_post"]


def read_csv(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _layer_idx(group: str) -> int:
    m = re.search(r"layer:(\d+)", group)
    return int(m.group(1)) if m else -1


def _save(fig, fig_dir: str, name: str) -> None:
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_global_cosine_matrix(rows, fig_dir):
    glob = {r["pair"]: float(r["cosine"]) for r in rows if r["granularity"] == "global"}
    if not glob:
        return
    stages = ["sft", "dpo", "rlvr", "post"]
    mat = np.full((len(stages), len(stages)), np.nan)
    for i, a in enumerate(stages):
        mat[i, i] = 1.0
        for j, b in enumerate(stages):
            key = f"{a}_{b}"
            rkey = f"{b}_{a}"
            if key in glob:
                mat[i, j] = mat[j, i] = glob[key]
            elif rkey in glob:
                mat[i, j] = mat[j, i] = glob[rkey]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    for i in range(len(stages)):
        for j in range(len(stages)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Global stage-delta cosine")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, fig_dir, "global_cosine_matrix.png")


def plot_layer_cosine_heatmap(rows, fig_dir):
    by_layer: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["granularity"] != "layer":
            continue
        li = _layer_idx(r["group"])
        if li >= 0:
            by_layer[li][r["pair"]] = float(r["cosine"])
    if not by_layer:
        return
    layers = sorted(by_layer)
    mat = np.full((len(COSINE_PAIRS), len(layers)), np.nan)
    for j, li in enumerate(layers):
        for i, p in enumerate(COSINE_PAIRS):
            if p in by_layer[li]:
                mat[i, j] = by_layer[li][p]
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.25), 4))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_yticks(range(len(COSINE_PAIRS)))
    ax.set_yticklabels(COSINE_PAIRS, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_title("Per-layer stage-delta cosine")
    plt.colorbar(im, ax=ax, fraction=0.046, label="cosine")
    _save(fig, fig_dir, "layer_cosine_heatmap.png")


def plot_module_norm_budget(rows, fig_dir):
    by_mod: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["granularity"] != "module":
            continue
        by_mod[r["group"].replace("module:", "")][r["stage"]] = float(r["norm"])
    if not by_mod:
        return
    mods = sorted(by_mod)
    x = np.arange(len(mods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, stage in enumerate(STEP_STAGES):
        vals = [by_mod[m].get(stage, 0.0) for m in mods]
        ax.bar(x + (k - 1) * width, vals, width, label=stage)
    ax.set_xticks(x)
    ax.set_xticklabels(mods, rotation=30, ha="right")
    ax.set_ylabel("||stage delta||")
    ax.set_title("Per-module stage norm budget")
    ax.legend()
    _save(fig, fig_dir, "module_norm_budget.png")


def _layer_series(rows, value_key):
    by_layer = {}
    for r in rows:
        if r["granularity"] != "layer":
            continue
        li = _layer_idx(r["group"])
        if li >= 0:
            by_layer[li] = float(r[value_key])
    layers = sorted(by_layer)
    return layers, [by_layer[l] for l in layers]


def plot_layer_line(rows, fig_dir, value_key, title, fname, color):
    layers, vals = _layer_series(rows, value_key)
    if not layers:
        return
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(layers, vals, marker="o", ms=3, color=color)
    ax.set_xlabel("layer")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, fname)


def plot_subspace_pc1(rows, fig_dir):
    layers, vals = _layer_series(rows, "pc1_energy")
    if not layers:
        return
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(layers, vals, marker="o", ms=3, color="purple")
    ax.set_xlabel("layer")
    ax.set_ylabel("pc1_energy")
    ax.set_title("Per-layer subspace PC1 energy (stage deltas)")
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, "subspace_pc1_energy.png")


def plot_residual_heatmap(rows, fig_dir):
    by_layer: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["granularity"] != "layer":
            continue
        li = _layer_idx(r["group"])
        if li >= 0:
            by_layer[li][r["stage"]] = float(r["residual_to_chord"])
    if not by_layer:
        return
    layers = sorted(by_layer)
    mat = np.full((len(STEP_STAGES), len(layers)), np.nan)
    for j, li in enumerate(layers):
        for i, s in enumerate(STEP_STAGES):
            if s in by_layer[li]:
                mat[i, j] = by_layer[li][s]
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.25), 3))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_yticks(range(len(STEP_STAGES)))
    ax.set_yticklabels(STEP_STAGES)
    ax.set_xlabel("layer")
    ax.set_title("Residual-to-chord per layer/stage")
    plt.colorbar(im, ax=ax, fraction=0.046, label="residual_to_chord")
    _save(fig, fig_dir, "residual_to_chord_heatmap.png")


def plot_rollback_heatmap(rows, fig_dir):
    by_layer: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["granularity"] != "layer":
            continue
        li = _layer_idx(r["group"])
        if li >= 0:
            by_layer[li][r["stage"]] = float(r["score"])
    if not by_layer:
        return
    layers = sorted(by_layer)
    stages = ["dpo", "rlvr"]
    mat = np.full((len(stages), len(layers)), np.nan)
    for j, li in enumerate(layers):
        for i, s in enumerate(stages):
            if s in by_layer[li]:
                mat[i, j] = by_layer[li][s]
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.25), 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("layer")
    ax.set_title("Static rollback score per layer/late-stage")
    plt.colorbar(im, ax=ax, fraction=0.046, label="z-score sum")
    _save(fig, fig_dir, "static_rollback_score_heatmap.png")


def main():
    ap = argparse.ArgumentParser(description="Plot static vector-field audit figures.")
    ap.add_argument("--config", help="Lineage YAML (to locate output_dir).")
    ap.add_argument("--output_dir", help="Override output dir holding the CSVs.")
    args = ap.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        if not args.config:
            ap.error("provide --config or --output_dir")
        with open(args.config) as f:
            out_dir = yaml.safe_load(f).get("output_dir", "outputs/static_vector_field")

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cosine = read_csv(os.path.join(out_dir, "pairwise_cosine.csv"))
    norm_budget = read_csv(os.path.join(out_dir, "stage_norm_budget.csv"))
    path_geom = read_csv(os.path.join(out_dir, "path_geometry.csv"))
    subspace = read_csv(os.path.join(out_dir, "subspace_rank.csv"))
    chord = read_csv(os.path.join(out_dir, "chord_projection.csv"))
    rollback = read_csv(os.path.join(out_dir, "rollback_scores.csv"))

    plot_global_cosine_matrix(cosine, fig_dir)
    plot_layer_cosine_heatmap(cosine, fig_dir)
    plot_module_norm_budget(norm_budget, fig_dir)
    plot_layer_line(path_geom, fig_dir, "straightness",
                    "Per-layer straightness", "layer_straightness.png", "teal")
    plot_layer_line(path_geom, fig_dir, "cancellation",
                    "Per-layer cancellation", "layer_cancellation.png", "darkorange")
    plot_layer_line(path_geom, fig_dir, "curvature",
                    "Per-layer curvature (rad)", "layer_curvature.png", "crimson")
    plot_subspace_pc1(subspace, fig_dir)
    plot_residual_heatmap(chord, fig_dir)
    plot_rollback_heatmap(rollback, fig_dir)
    print("Figures done.")


if __name__ == "__main__":
    main()
