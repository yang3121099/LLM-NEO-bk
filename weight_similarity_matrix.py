"""
Pairwise Weight Similarity Matrix across multiple model checkpoints.

Extended from weight_similarity_analysis.py to compute an N×N similarity matrix
for the Llama-3.1-8B training pipeline and related models.

Models included:
  Pipeline (same base):
    1. meta-llama/Llama-3.1-8B              (Base)
    2. meta-llama/Llama-3.1-8B-Instruct     (Official Instruct)
    3. allenai/Llama-3.1-Tulu-3-8B-SFT      (SFT stage)
    4. allenai/Llama-3.1-Tulu-3-8B-DPO      (DPO stage)
    5. allenai/Llama-3.1-Tulu-3-8B          (RLVR final = Tulu 3)
  Independent models (same architecture, different training):
    6. allenai/Llama-3.1-Tulu-3.1-8B        (Tulu 3.1, GRPO-based)
    7. meta-llama/Llama-3-8B-Instruct       (Llama 3, not 3.1)
    8. deepseek-ai/DeepSeek-R1-Distill-Llama-8B (Distilled from R1)

Usage:
    python weight_similarity_matrix.py --model_dir /path/to/models --output_dir results
    python weight_similarity_matrix.py --models model1:/path/1 model2:/path/2 ...
"""

import argparse
import itertools
import os
import json
import time

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from weight_similarity_analysis import compute_relative_gap_ratio, load_model_state_dict, categorize_param, extract_layer_idx


# Default model configuration
DEFAULT_MODELS = {
    # Training pipeline checkpoints (Llama-3.1-8B based)
    "Llama3.1-8B-Base": "meta-llama/Llama-3.1-8B",
    "Llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Tulu3-8B-SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "Tulu3-8B-DPO": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "Tulu3-8B-RLVR": "allenai/Llama-3.1-Tulu-3-8B",
    # Independent models (same architecture, different training)
    "Tulu3.1-8B": "allenai/Llama-3.1-Tulu-3.1-8B",
    "Llama3-8B-Base": "meta-llama/Llama-3-8B",
    "Llama3-8B-Instruct": "meta-llama/Llama-3-8B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

# Group labels for visual separation in heatmap
MODEL_GROUPS = {
    "Llama-3.1 Pipeline": [
        "Llama3.1-8B-Base",
        "Llama3.1-8B-Instruct",
        "Tulu3-8B-SFT",
        "Tulu3-8B-DPO",
        "Tulu3-8B-RLVR",
    ],
    "Independent": [
        "Tulu3.1-8B",
        "Llama3-8B-Base",
        "Llama3-8B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B",
    ],
}


def compute_pairwise_sigma(sd_a: dict, sd_b: dict, per_component: bool = False):
    """
    Compute the overall relative gap ratio σ between two state dicts.

    Matches the official Shadow-FT implementation (sigma_v2.py):
      - Only compare model.layers.* keys (skip embed_tokens, lm_head, norms outside layers)
      - Per-key σ = sum(|A-B|) / (sum(|A|) + sum(|B|))
      - Overall σ = simple average across all per-key σ values

    Args:
        sd_a, sd_b: model state dicts
        per_component: if True, also return per-component breakdown

    Returns:
        overall_sigma: float
        component_sigmas: dict (only if per_component=True)
    """
    import re
    layer_pattern = re.compile(r"model\.layers\.\d+\.")

    common_keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))

    all_sigmas = []
    component_data = {}

    for name in common_keys:
        # Official: only model.layers.* keys
        if not layer_pattern.search(name):
            continue

        wa = sd_a[name]
        wb = sd_b[name]
        if wa.shape != wb.shape:
            continue

        sigma = compute_relative_gap_ratio(wa, wb)
        all_sigmas.append(sigma)

        if per_component:
            cat = categorize_param(name)
            if cat not in component_data:
                component_data[cat] = []
            component_data[cat].append(sigma)

    # Simple average across all per-key σ (matching official script)
    overall_sigma = float(np.mean(all_sigmas)) if all_sigmas else 0.0

    if per_component:
        component_sigmas = {}
        for cat, sigmas in component_data.items():
            component_sigmas[cat] = float(np.mean(sigmas))
        return overall_sigma, component_sigmas

    return overall_sigma


def compute_pairwise_per_layer(sd_a: dict, sd_b: dict):
    """Compute per-layer σ between two state dicts (official formula)."""
    common_keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))
    layer_data = {}

    for name in common_keys:
        layer_idx = extract_layer_idx(name)
        if layer_idx < 0:
            continue
        wa, wb = sd_a[name], sd_b[name]
        if wa.shape != wb.shape:
            continue
        sigma = compute_relative_gap_ratio(wa, wb)
        if layer_idx not in layer_data:
            layer_data[layer_idx] = []
        layer_data[layer_idx].append(sigma)

    # Simple average per layer (matching official)
    layer_sigmas = {}
    for layer_idx, sigmas in sorted(layer_data.items()):
        layer_sigmas[layer_idx] = float(np.mean(sigmas))

    return layer_sigmas


def plot_similarity_matrix(matrix: np.ndarray, labels: list, output_path: str, title: str = ""):
    """Plot a heatmap of the pairwise similarity matrix with annotations."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a diverging colormap - lower σ = more similar = cooler color
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(0.1, np.nanmax(matrix)))

    # Annotate cells with σ values
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if val > np.nanmax(matrix) * 0.6 else "black"
            if i == j:
                ax.text(j, i, "0", ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(j, i, f"{val:.4f}\n({val*100:.2f}%)",
                        ha="center", va="center", fontsize=8, color=text_color)

    # Draw group boundaries
    group_sizes = [len(v) for v in MODEL_GROUPS.values()]
    # Filter to only groups that have models in our labels
    actual_groups = []
    idx = 0
    for group_name, group_models in MODEL_GROUPS.items():
        present = [m for m in group_models if m in labels]
        if present:
            actual_groups.append((group_name, len(present), idx))
            idx += len(present)

    cumsum = 0
    for group_name, size, start in actual_groups:
        if cumsum > 0:
            ax.axhline(y=cumsum - 0.5, color="blue", linewidth=2, alpha=0.5)
            ax.axvline(x=cumsum - 0.5, color="blue", linewidth=2, alpha=0.5)
        cumsum += size

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    if not title:
        title = "Pairwise Weight Similarity Matrix (σ)\nSmaller σ = More Similar"
    ax.set_title(title, fontsize=14, pad=15)

    cbar = plt.colorbar(im, ax=ax, label="Relative Gap Ratio σ", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_component_matrix(component_matrices: dict, labels: list, output_dir: str):
    """Plot separate heatmaps for each component type."""
    components = sorted(component_matrices.keys())
    n_comp = len(components)

    cols = min(3, n_comp)
    rows = (n_comp + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n_comp == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    vmax = max(np.nanmax(m) for m in component_matrices.values() if np.nanmax(m) > 0)

    for idx, comp in enumerate(components):
        ax = axes[idx]
        mat = component_matrices[comp]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax)
        n = len(labels)
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = mat[i, j]
                    text_color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=text_color)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(comp, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_comp, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Component Weight Similarity Matrix (σ)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_component_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_layer_comparison(per_layer_data: dict, labels: list, output_dir: str):
    """
    Plot per-layer σ curves for selected model pairs.
    per_layer_data: {(i, j): {layer_idx: sigma}}
    """
    # Select interesting pairs: adjacent pipeline stages + independent models vs base
    n = len(labels)
    interesting_pairs = []

    # Pipeline: consecutive pairs
    pipeline_labels = ["Llama3.1-8B-Base", "Llama3.1-8B-Instruct", "Tulu3-8B-SFT", "Tulu3-8B-DPO", "Tulu3-8B-RLVR"]
    pipeline_indices = [labels.index(l) for l in pipeline_labels if l in labels]
    for k in range(len(pipeline_indices) - 1):
        interesting_pairs.append((pipeline_indices[k], pipeline_indices[k + 1]))

    # Independent models vs base
    base_idx = labels.index("Llama3.1-8B-Base") if "Llama3.1-8B-Base" in labels else 0
    for lbl in ["Llama3-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B", "Tulu3.1-8B"]:
        if lbl in labels:
            interesting_pairs.append((base_idx, labels.index(lbl)))

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(interesting_pairs)))

    for k, (i, j) in enumerate(interesting_pairs):
        key = (min(i, j), max(i, j))
        if key not in per_layer_data:
            continue
        layer_sigmas = per_layer_data[key]
        layers = sorted(layer_sigmas.keys())
        sigmas = [layer_sigmas[l] for l in layers]
        pair_label = f"{labels[i]} ↔ {labels[j]}"
        ax.plot(layers, sigmas, marker=".", markersize=3, label=pair_label, color=colors[k], alpha=0.8)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Relative Gap Ratio σ", fontsize=12)
    ax.set_title("Per-Layer Weight Similarity for Selected Model Pairs", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_layer_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delta_transferability(matrix: np.ndarray, labels: list, output_dir: str):
    """
    Visualize which model pair deltas are most transferable.
    For the Shadow-FT framework, smaller σ implies better delta transferability.
    Highlight the pipeline order: Base→SFT→DPO→RLVR.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Sort by similarity to base (first model)
    base_idx = 0
    order = sorted(range(n), key=lambda i: matrix[base_idx, i] if i != base_idx else -1)
    sorted_matrix = matrix[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]

    im = ax.imshow(sorted_matrix, cmap="RdYlGn_r", vmin=0, vmax=max(0.1, np.nanmax(matrix)))

    for i in range(n):
        for j in range(n):
            val = sorted_matrix[i, j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                text_color = "white" if val > np.nanmax(matrix) * 0.5 else "black"
                ax.text(j, i, f"{val*100:.2f}%", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_title("Delta Transferability Matrix\n(Sorted by similarity to Base, Green=More Transferable)", fontsize=13)
    plt.colorbar(im, ax=ax, label="σ (lower = more transferable)", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "delta_transferability.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise Weight Similarity Matrix for Llama-3.1-8B family models"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Base directory containing all model subdirectories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model specs as name:path pairs, e.g., 'Llama3.1-Base:/path/to/model'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weight_similarity_matrix_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--per_layer",
        action="store_true",
        default=True,
        help="Compute per-layer analysis (default: True)",
    )
    parser.add_argument(
        "--per_component",
        action="store_true",
        default=True,
        help="Compute per-component analysis (default: True)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model paths
    if args.models:
        model_configs = {}
        for spec in args.models:
            name, path = spec.split(":", 1)
            model_configs[name] = path
    elif args.model_dir:
        # Try to find models by default names
        model_configs = {}
        for name, hf_id in DEFAULT_MODELS.items():
            # Try common directory naming conventions
            candidates = [
                os.path.join(args.model_dir, hf_id),
                os.path.join(args.model_dir, hf_id.split("/")[-1]),
                os.path.join(args.model_dir, name),
            ]
            for cand in candidates:
                if os.path.isdir(cand):
                    model_configs[name] = cand
                    break
        if not model_configs:
            print(f"No models found in {args.model_dir}. Use --models to specify paths.")
            return
    else:
        print("Please provide --model_dir or --models. Example:")
        print("  python weight_similarity_matrix.py --models \\")
        for name, hf_id in DEFAULT_MODELS.items():
            print(f"    {name}:/path/to/{hf_id.split('/')[-1]} \\")
        print()
        print("Default model IDs (download from HuggingFace):")
        for name, hf_id in DEFAULT_MODELS.items():
            print(f"  {name}: {hf_id}")
        return

    labels = list(model_configs.keys())
    n = len(labels)

    print("=" * 70)
    print("Pairwise Weight Similarity Matrix Analysis")
    print("(Shadow-FT, arXiv: 2505.12716, Section 2.3)")
    print("=" * 70)
    print(f"\nModels ({n}):")
    for name, path in model_configs.items():
        print(f"  {name}: {path}")
    print(f"\nTotal pairs to compute: {n * (n - 1) // 2}")
    print()

    # Load all model state dicts
    state_dicts = {}
    for name, path in model_configs.items():
        print(f"Loading {name}...")
        t0 = time.time()
        state_dicts[name] = load_model_state_dict(path)
        dt = time.time() - t0
        print(f"  Loaded {len(state_dicts[name])} params in {dt:.1f}s")

    # Compute pairwise σ
    print(f"\nComputing pairwise σ...")
    sigma_matrix = np.zeros((n, n))
    component_matrices = {}
    per_layer_data = {}

    for i, j in itertools.combinations(range(n), 2):
        name_a, name_b = labels[i], labels[j]
        print(f"  {name_a} ↔ {name_b}...", end=" ", flush=True)
        t0 = time.time()

        if args.per_component:
            sigma, comp_sigmas = compute_pairwise_sigma(
                state_dicts[name_a], state_dicts[name_b], per_component=True
            )
            for comp, val in comp_sigmas.items():
                if comp not in component_matrices:
                    component_matrices[comp] = np.zeros((n, n))
                component_matrices[comp][i, j] = val
                component_matrices[comp][j, i] = val
        else:
            sigma = compute_pairwise_sigma(state_dicts[name_a], state_dicts[name_b])

        sigma_matrix[i, j] = sigma
        sigma_matrix[j, i] = sigma

        if args.per_layer:
            per_layer_data[(i, j)] = compute_pairwise_per_layer(
                state_dicts[name_a], state_dicts[name_b]
            )

        dt = time.time() - t0
        print(f"σ = {sigma:.6f} ({sigma*100:.3f}%) [{dt:.1f}s]")

    # Print results table
    print("\n" + "=" * 70)
    print("PAIRWISE SIMILARITY MATRIX (σ values)")
    print("=" * 70)

    # Header
    col_w = 12
    header = f"{'':>{col_w}}" + "".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{labels[i][:col_w]:>{col_w}}"
        for j in range(n):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{sigma_matrix[i,j]:>{col_w}.6f}"
        print(row)

    # Print as percentage table
    print(f"\nPAIRWISE DIFFERENCE PERCENTAGE")
    print("-" * len(header))
    for i in range(n):
        row = f"{labels[i][:col_w]:>{col_w}}"
        for j in range(n):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{sigma_matrix[i,j]*100:>{col_w-1}.3f}%"
        print(row)

    # Key insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")

    # Find most/least similar pairs
    upper_tri = []
    for i, j in itertools.combinations(range(n), 2):
        upper_tri.append((sigma_matrix[i, j], labels[i], labels[j]))
    upper_tri.sort()

    print("\nMost similar pairs (lowest σ):")
    for sigma, a, b in upper_tri[:5]:
        print(f"  {a} ↔ {b}: σ = {sigma:.6f} ({sigma*100:.3f}%)")

    print("\nLeast similar pairs (highest σ):")
    for sigma, a, b in upper_tri[-5:]:
        print(f"  {a} ↔ {b}: σ = {sigma:.6f} ({sigma*100:.3f}%)")

    # Pipeline progression
    pipeline = ["Llama3.1-8B-Base", "Tulu3-8B-SFT", "Tulu3-8B-DPO", "Tulu3-8B-RLVR"]
    pipeline_present = [l for l in pipeline if l in labels]
    if len(pipeline_present) > 1:
        print("\nTraining pipeline progression (Base→SFT→DPO→RLVR):")
        for k in range(len(pipeline_present) - 1):
            a, b = pipeline_present[k], pipeline_present[k + 1]
            i, j = labels.index(a), labels.index(b)
            s = sigma_matrix[i, j]
            print(f"  {a} → {b}: σ = {s:.6f} ({s*100:.3f}%)")
        # Cumulative from base
        base = pipeline_present[0]
        print(f"\nCumulative drift from {base}:")
        for p in pipeline_present[1:]:
            i, j = labels.index(base), labels.index(p)
            s = sigma_matrix[i, j]
            print(f"  {base} → {p}: σ = {s:.6f} ({s*100:.3f}%)")

    # Generate plots
    print(f"\nGenerating visualizations...")

    plot_similarity_matrix(
        sigma_matrix, labels,
        os.path.join(args.output_dir, "pairwise_similarity_matrix.png"),
        title="Pairwise Weight Similarity Matrix (σ)\nLlama-3.1-8B Family"
    )

    plot_delta_transferability(sigma_matrix, labels, args.output_dir)

    if args.per_component and component_matrices:
        plot_per_component_matrix(component_matrices, labels, args.output_dir)

    if args.per_layer and per_layer_data:
        plot_per_layer_comparison(per_layer_data, labels, args.output_dir)

    # Save full results
    results = {
        "models": {name: path for name, path in model_configs.items()},
        "sigma_matrix": sigma_matrix.tolist(),
        "labels": labels,
        "pairs": [
            {"model_a": a, "model_b": b, "sigma": float(s)}
            for s, a, b in upper_tri
        ],
    }
    if component_matrices:
        results["per_component_matrices"] = {
            comp: mat.tolist() for comp, mat in component_matrices.items()
        }

    json_path = os.path.join(args.output_dir, "pairwise_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone! All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
