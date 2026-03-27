"""
Visualize Cross-Delta Transfer Matrix

Reads OpenCompass evaluation results and produces:
  (1) Absolute score matrix:  A[s,t] = score(merge(source_s → target_t))
  (2) Gain-over-direct-FT:   G[s,t] = A[s,t] - score(direct_FT(target_t))
  (3) σ-vs-gain correlation:  scatter plot of weight similarity vs transfer gain

Usage:
    python3 visualize_transfer_matrix.py \
        --eval_dir opencompass/outputs/cross-delta-exp/ \
        --sigma_file results/.../weight_similarity/pairwise_results.json \
        --output_dir results/.../transfer_matrix_plots
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Default experiment configuration
SOURCES = ["Base", "SFT", "DPO", "RLVR"]
TARGETS = ["Base", "SFT", "DPO", "RLVR", "Instruct", "Tulu3.1", "Llama3-Inst", "R1-Distill"]

# Benchmarks to aggregate (Math-7 minus MATH)
BENCHMARKS = [
    "math-500",
    "minerva_math",
    "gsm8k",
    "gsm8k_0shot",
    "aime2024",
    "svamp",
]


def load_opencompass_results(eval_dir: str) -> dict:
    """
    Load OpenCompass evaluation results.
    Searches for summary CSV/JSON files in the eval_dir.
    Returns: {model_abbr: {benchmark: score}}
    """
    results = {}

    # Try CSV summary files
    for csv_file in Path(eval_dir).rglob("summary/*.csv"):
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("dataset", row.get("model", ""))
                    if not model:
                        continue
                    scores = {}
                    for k, v in row.items():
                        if k in ("dataset", "model"):
                            continue
                        try:
                            scores[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                    if scores:
                        results[model] = scores
        except Exception:
            continue

    # Try JSON summary
    for json_file in Path(eval_dir).rglob("summary/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for model, benchmarks in data.items():
                    if isinstance(benchmarks, dict):
                        results[model] = {k: float(v) for k, v in benchmarks.items()
                                          if isinstance(v, (int, float))}
        except Exception:
            continue

    # Try reading from individual result directories
    for result_json in Path(eval_dir).rglob("results/**/result.json"):
        try:
            with open(result_json) as f:
                data = json.load(f)
            # Extract model name from path
            parts = result_json.parts
            model = None
            for i, p in enumerate(parts):
                if p == "results" and i + 1 < len(parts):
                    model = parts[i + 1]
                    break
            if model and isinstance(data, dict):
                if model not in results:
                    results[model] = {}
                results[model].update({k: float(v) for k, v in data.items()
                                       if isinstance(v, (int, float))})
        except Exception:
            continue

    return results


def load_results_manual(results_file: str) -> dict:
    """Load results from a manually prepared JSON file.

    Expected format:
    {
        "Base2Base": {"math-500": 45.2, "gsm8k": 60.1, ...},
        "Base2SFT": {"math-500": 48.0, ...},
        "DirectFT-Base": {"math-500": 44.0, ...},
        ...
    }
    """
    with open(results_file) as f:
        return json.load(f)


def build_transfer_matrix(results: dict, sources: list, targets: list,
                          benchmarks: list) -> tuple:
    """
    Build the absolute score and gain matrices.

    Returns:
        abs_matrix: np.array of shape (len(sources), len(targets))
        gain_matrix: np.array of shape (len(sources), len(targets))
        baseline_scores: dict {target: avg_score}
    """
    n_src, n_tgt = len(sources), len(targets)
    abs_matrix = np.full((n_src, n_tgt), np.nan)
    baseline_scores = {}

    # Compute baseline scores (direct-FT on each target)
    for ti, tgt in enumerate(targets):
        baseline_key = f"DirectFT-{tgt}"
        if baseline_key in results:
            scores = [results[baseline_key].get(b, np.nan) for b in benchmarks]
            valid = [s for s in scores if not np.isnan(s)]
            baseline_scores[tgt] = np.mean(valid) if valid else np.nan
        else:
            baseline_scores[tgt] = np.nan

    # Build absolute score matrix
    for si, src in enumerate(sources):
        for ti, tgt in enumerate(targets):
            merge_key = f"{src}2{tgt}"
            if merge_key in results:
                scores = [results[merge_key].get(b, np.nan) for b in benchmarks]
                valid = [s for s in scores if not np.isnan(s)]
                abs_matrix[si, ti] = np.mean(valid) if valid else np.nan

    # Build gain matrix
    gain_matrix = np.full((n_src, n_tgt), np.nan)
    for si in range(n_src):
        for ti in range(n_tgt):
            if not np.isnan(abs_matrix[si, ti]) and not np.isnan(baseline_scores.get(targets[ti], np.nan)):
                gain_matrix[si, ti] = abs_matrix[si, ti] - baseline_scores[targets[ti]]

    return abs_matrix, gain_matrix, baseline_scores


def build_per_benchmark_matrices(results: dict, sources: list, targets: list,
                                  benchmarks: list) -> dict:
    """Build separate matrices for each benchmark."""
    matrices = {}
    for bench in benchmarks:
        n_src, n_tgt = len(sources), len(targets)
        mat = np.full((n_src, n_tgt), np.nan)
        for si, src in enumerate(sources):
            for ti, tgt in enumerate(targets):
                merge_key = f"{src}2{tgt}"
                if merge_key in results and bench in results[merge_key]:
                    mat[si, ti] = results[merge_key][bench]
        matrices[bench] = mat
    return matrices


def plot_matrix(matrix: np.ndarray, row_labels: list, col_labels: list,
                title: str, output_path: str, cmap="RdYlGn",
                fmt=".1f", vmin=None, vmax=None,
                annotate_fmt=None, highlight_diagonal=True):
    """Plot a heatmap matrix with annotations."""
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.4), max(6, n_rows * 1.2)))

    if vmin is None:
        vmin = np.nanmin(matrix)
    if vmax is None:
        vmax = np.nanmax(matrix)

    # Handle diverging colormap for gain matrix
    if "Gain" in title or "gain" in title:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        cmap = "RdYlGn"

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
                continue

            text_color = "white" if abs(val - vmin) > 0.7 * (vmax - vmin) or abs(val - vmax) > 0.7 * (vmax - vmin) else "black"

            if annotate_fmt:
                text = annotate_fmt(val)
            else:
                text = f"{val:{fmt}}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight="bold" if highlight_diagonal and row_labels[i] == col_labels[j] else "normal",
                    color=text_color)

    # Highlight diagonal (source == target)
    if highlight_diagonal:
        for i, src in enumerate(row_labels):
            for j, tgt in enumerate(col_labels):
                if src == tgt:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         linewidth=3, edgecolor="blue",
                                         facecolor="none", linestyle="--")
                    ax.add_patch(rect)

    # Draw separator between pipeline and independent targets
    n_pipeline = sum(1 for t in col_labels if t in SOURCES)
    if 0 < n_pipeline < n_cols:
        ax.axvline(x=n_pipeline - 0.5, color="navy", linewidth=2, alpha=0.5)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Target Model", fontsize=12, labelpad=8)
    ax.set_ylabel("Source Delta", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, pad=12)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Score" if "Gain" not in title else "Gain (%)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sigma_vs_gain(sigma_matrix: np.ndarray, gain_matrix: np.ndarray,
                       sigma_labels: list, sources: list, targets: list,
                       output_path: str):
    """Scatter plot: weight similarity σ vs transfer gain."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Map sigma labels to indices
    sigma_idx = {l: i for i, l in enumerate(sigma_labels)}

    xs, ys, labels_list = [], [], []
    for si, src in enumerate(sources):
        for ti, tgt in enumerate(targets):
            if src == tgt:
                continue  # skip diagonal
            gain = gain_matrix[si, ti]
            if np.isnan(gain):
                continue

            # Find sigma between source and target
            s_src = sigma_idx.get(src, sigma_idx.get(f"Llama3.1-8B-{src}", -1))
            s_tgt = sigma_idx.get(tgt, sigma_idx.get(f"Llama3.1-8B-{tgt}", -1))

            if s_src < 0 or s_tgt < 0:
                continue

            sigma = sigma_matrix[s_src, s_tgt]
            if sigma == 0 or np.isnan(sigma):
                continue

            xs.append(sigma)
            ys.append(gain)
            labels_list.append(f"{src}→{tgt}")

    if not xs:
        print("  No data for σ-vs-gain plot")
        return

    ax.scatter(xs, ys, s=80, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Annotate points
    for x, y, label in zip(xs, ys, labels_list):
        ax.annotate(label, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    # Reference line at gain = 0
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Gain = 0")

    # Trend line
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, p(x_line), "b--", alpha=0.4, label=f"Trend (slope={z[0]:.1f})")

        # Correlation
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Weight Similarity σ (source ↔ target)", fontsize=12)
    ax.set_ylabel("Gain over Direct-FT (%)", fontsize=12)
    ax.set_title("Weight Similarity vs Transfer Gain\n(Shadow-FT Boundary Analysis)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_analysis(abs_matrix, gain_matrix, baseline_scores, sources, targets):
    """Print analysis insights."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Sanity check: source == target should ≈ direct-FT
    print("\n[Sanity Check] source == target (should be ≈ 0 gain):")
    for si, src in enumerate(sources):
        for ti, tgt in enumerate(targets):
            if src == tgt:
                gain = gain_matrix[si, ti]
                status = "OK" if abs(gain) < 2.0 else "MISMATCH"
                print(f"  {src}→{tgt}: gain = {gain:+.2f}%  [{status}]")

    # Best source for each target
    print("\n[Best Source] for each target:")
    for ti, tgt in enumerate(targets):
        best_si = np.nanargmax(abs_matrix[:, ti])
        best_src = sources[best_si]
        best_score = abs_matrix[best_si, ti]
        gain = gain_matrix[best_si, ti]
        baseline = baseline_scores.get(tgt, np.nan)
        print(f"  {tgt}: best source = {best_src} "
              f"(score={best_score:.1f}, gain={gain:+.1f}, baseline={baseline:.1f})")

    # Pipeline ordering insight
    pipeline = ["Base", "SFT", "DPO", "RLVR"]
    instruct_ti = targets.index("Instruct") if "Instruct" in targets else -1
    if instruct_ti >= 0:
        print(f"\n[Pipeline → Instruct] transfer scores:")
        for src in pipeline:
            si = sources.index(src)
            score = abs_matrix[si, instruct_ti]
            gain = gain_matrix[si, instruct_ti]
            print(f"  {src}→Instruct: score={score:.1f}, gain={gain:+.1f}")

    # Cross-family transfers
    cross_targets = [t for t in targets if t not in pipeline]
    if cross_targets:
        print(f"\n[Cross-Family Transfers] (average gain per source):")
        for si, src in enumerate(sources):
            gains = []
            for tgt in cross_targets:
                ti = targets.index(tgt)
                g = gain_matrix[si, ti]
                if not np.isnan(g):
                    gains.append(g)
            if gains:
                print(f"  {src}: avg gain = {np.mean(gains):+.2f}% "
                      f"(range: {min(gains):+.1f} to {max(gains):+.1f})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-delta transfer matrix results"
    )
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="OpenCompass evaluation output directory")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Manually prepared results JSON file (alternative to eval_dir)")
    parser.add_argument("--sigma_file", type=str, default=None,
                        help="Pairwise weight similarity results JSON (from weight_similarity_matrix.py)")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Experiment config JSON (auto-generated by run_cross_delta_experiment.sh)")
    parser.add_argument("--output_dir", type=str, default="transfer_matrix_results",
                        help="Output directory for plots")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo plots with synthetic data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sources = SOURCES
    targets = TARGETS

    # Load config if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            cfg = json.load(f)
        sources = cfg.get("sources", sources)
        targets = cfg.get("targets", targets)

    if args.demo:
        print("Generating demo with synthetic data...")
        np.random.seed(42)
        n_src, n_tgt = len(sources), len(targets)

        # Simulate: closer pipeline stages → higher scores
        results = {}
        pipeline_order = {"Base": 0, "SFT": 1, "DPO": 2, "RLVR": 3,
                          "Instruct": 2.5, "Tulu3.1": 3.5,
                          "Llama3-Inst": 4, "R1-Distill": 5}

        for src in sources:
            for tgt in targets:
                key = f"{src}2{tgt}"
                dist = abs(pipeline_order.get(src, 0) - pipeline_order.get(tgt, 0))
                base_score = 55 - dist * 3 + np.random.randn() * 2
                results[key] = {b: max(0, min(100, base_score + np.random.randn() * 5))
                                for b in BENCHMARKS}

        for tgt in targets:
            key = f"DirectFT-{tgt}"
            base_score = 50 + np.random.randn() * 3
            results[key] = {b: max(0, min(100, base_score + np.random.randn() * 5))
                            for b in BENCHMARKS}

    elif args.results_file:
        results = load_results_manual(args.results_file)
    elif args.eval_dir:
        results = load_opencompass_results(args.eval_dir)
    else:
        print("Please provide --eval_dir, --results_file, or --demo")
        return

    print(f"Loaded results for {len(results)} model configurations")

    # Build matrices
    abs_matrix, gain_matrix, baseline_scores = build_transfer_matrix(
        results, sources, targets, BENCHMARKS
    )

    # Print tables
    print("\n" + "=" * 70)
    print("ABSOLUTE SCORE MATRIX (avg across math benchmarks)")
    print("=" * 70)
    header = f"{'Source':>10}" + "".join(f"{t:>12}" for t in targets)
    print(header)
    for si, src in enumerate(sources):
        row = f"{src:>10}"
        for ti in range(len(targets)):
            v = abs_matrix[si, ti]
            row += f"{v:>12.1f}" if not np.isnan(v) else f"{'—':>12}"
        print(row)

    print(f"\n{'Baseline':>10}" + "".join(
        f"{baseline_scores.get(t, float('nan')):>12.1f}" if not np.isnan(baseline_scores.get(t, float('nan')))
        else f"{'—':>12}" for t in targets
    ))

    print("\n" + "=" * 70)
    print("GAIN OVER DIRECT-FT MATRIX")
    print("=" * 70)
    print(header)
    for si, src in enumerate(sources):
        row = f"{src:>10}"
        for ti in range(len(targets)):
            v = gain_matrix[si, ti]
            row += f"{v:>+12.1f}" if not np.isnan(v) else f"{'—':>12}"
        print(row)

    # Print analysis
    print_analysis(abs_matrix, gain_matrix, baseline_scores, sources, targets)

    # Generate plots
    print("\nGenerating visualizations...")

    # (1) Absolute score matrix
    plot_matrix(
        abs_matrix, sources, targets,
        "Absolute Score Matrix: A[s,t] = score(merge(source_s → target_t))\n(avg across math benchmarks)",
        os.path.join(args.output_dir, "absolute_score_matrix.png"),
        cmap="YlGn",
    )

    # (2) Gain matrix
    plot_matrix(
        gain_matrix, sources, targets,
        "Gain over Direct-FT: G[s,t] = A[s,t] - score(DirectFT(target_t))\n(positive = transfer helps)",
        os.path.join(args.output_dir, "gain_over_directft_matrix.png"),
        cmap="RdYlGn",
        annotate_fmt=lambda v: f"{v:+.1f}",
    )

    # (3) Per-benchmark matrices
    if not args.demo and args.results_file:
        per_bench = build_per_benchmark_matrices(results, sources, targets, BENCHMARKS)
        for bench, mat in per_bench.items():
            if not np.all(np.isnan(mat)):
                plot_matrix(
                    mat, sources, targets,
                    f"Transfer Matrix: {bench}",
                    os.path.join(args.output_dir, f"matrix_{bench}.png"),
                    cmap="YlGn",
                )

    # (4) σ vs gain correlation
    if args.sigma_file and os.path.exists(args.sigma_file):
        with open(args.sigma_file) as f:
            sigma_data = json.load(f)
        sigma_matrix = np.array(sigma_data["sigma_matrix"])
        sigma_labels = sigma_data["labels"]
        plot_sigma_vs_gain(
            sigma_matrix, gain_matrix, sigma_labels,
            sources, targets,
            os.path.join(args.output_dir, "sigma_vs_gain.png"),
        )

    # Save results
    output = {
        "sources": sources,
        "targets": targets,
        "benchmarks": BENCHMARKS,
        "absolute_scores": abs_matrix.tolist(),
        "gain_over_direct_ft": gain_matrix.tolist(),
        "baseline_scores": {k: float(v) if not np.isnan(v) else None
                            for k, v in baseline_scores.items()},
    }
    json_path = os.path.join(args.output_dir, "transfer_matrix_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone! All results in {args.output_dir}/")


if __name__ == "__main__":
    main()
