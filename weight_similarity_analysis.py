"""
Weight Similarity Analysis between Base and Instruct models.

Reproduces Section 2.3 of "Shadow-FT: Tuning Instruct Model via Training on Paired Base Model"
(arXiv: 2505.12716)

Computes the relative gap ratio σ between paired Base and Instruct model weights:
    σ(W_B, W_I) = mean( |W_B - W_I| / (|W_B| + |W_I| + ε) )

where:
  - W_B: weight matrix from the Base model
  - W_I: weight matrix from the Instruct model
  - ε: small constant to avoid division by zero
  - σ = 0 means identical weights
  - σ = 1 means completely different weights
"""

import argparse
import torch
import numpy as np
from collections import defaultdict
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json


def compute_relative_gap_ratio(w_base: torch.Tensor, w_instruct: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute the relative gap ratio σ between two weight tensors.

    σ = mean( |W_B - W_I| / (|W_B| + |W_I| + ε) )

    Returns a float in [0, 1].
    """
    diff = torch.abs(w_base.float() - w_instruct.float())
    denom = torch.abs(w_base.float()) + torch.abs(w_instruct.float()) + eps
    sigma = (diff / denom).mean().item()
    return sigma


def load_model_state_dict(model_path: str) -> dict:
    """
    Load model weights from a HuggingFace model path.
    Supports both safetensors and pytorch bin formats.
    """
    from glob import glob

    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        state_dict = {}
        for sf_file in safetensor_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    bin_files = sorted(glob(os.path.join(model_path, "*.bin")))
    if bin_files:
        state_dict = {}
        for bin_file in bin_files:
            sd = torch.load(bin_file, map_location="cpu", weights_only=True)
            state_dict.update(sd)
        return state_dict

    raise FileNotFoundError(
        f"No safetensors or pytorch bin files found in {model_path}. "
        "Please provide a valid HuggingFace model directory."
    )


def categorize_param(name: str) -> str:
    """Categorize a parameter name into a component type."""
    if "embed" in name or "lm_head" in name:
        return "embedding"
    elif "self_attn" in name or "attention" in name:
        if "q_proj" in name:
            return "attn.q_proj"
        elif "k_proj" in name:
            return "attn.k_proj"
        elif "v_proj" in name:
            return "attn.v_proj"
        elif "o_proj" in name:
            return "attn.o_proj"
        else:
            return "attention"
    elif "mlp" in name:
        if "gate_proj" in name:
            return "mlp.gate_proj"
        elif "up_proj" in name:
            return "mlp.up_proj"
        elif "down_proj" in name:
            return "mlp.down_proj"
        else:
            return "mlp"
    elif "norm" in name or "layernorm" in name:
        return "norm"
    else:
        return "other"


def extract_layer_idx(name: str) -> int:
    """Extract the layer index from a parameter name. Returns -1 if not found."""
    import re
    match = re.search(r"layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return -1


def analyze_weight_similarity(base_path: str, instruct_path: str, output_dir: str = "weight_similarity_results"):
    """
    Perform weight similarity analysis between Base and Instruct models.

    Args:
        base_path: Path to the Base model directory
        instruct_path: Path to the Instruct model directory
        output_dir: Directory to save analysis results and plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Weight Similarity Analysis (Shadow-FT, arXiv: 2505.12716, Sec 2.3)")
    print("=" * 70)
    print(f"\nBase model:     {base_path}")
    print(f"Instruct model: {instruct_path}")
    print(f"Output dir:     {output_dir}\n")

    # Load weights
    print("Loading Base model weights...")
    base_sd = load_model_state_dict(base_path)
    print(f"  Loaded {len(base_sd)} parameters")

    print("Loading Instruct model weights...")
    instruct_sd = load_model_state_dict(instruct_path)
    print(f"  Loaded {len(instruct_sd)} parameters\n")

    # Find common keys
    common_keys = sorted(set(base_sd.keys()) & set(instruct_sd.keys()))
    base_only = set(base_sd.keys()) - set(instruct_sd.keys())
    instruct_only = set(instruct_sd.keys()) - set(base_sd.keys())

    if base_only:
        print(f"Warning: {len(base_only)} keys only in Base model")
    if instruct_only:
        print(f"Warning: {len(instruct_only)} keys only in Instruct model")
    print(f"Common parameters: {len(common_keys)}\n")

    # Compute per-parameter σ
    print("Computing relative gap ratio σ for each parameter...")
    results = []
    layer_results = defaultdict(list)  # layer_idx -> list of (name, sigma)
    category_results = defaultdict(list)  # category -> list of sigma

    for name in common_keys:
        w_base = base_sd[name]
        w_inst = instruct_sd[name]

        if w_base.shape != w_inst.shape:
            print(f"  Skipping {name}: shape mismatch {w_base.shape} vs {w_inst.shape}")
            continue

        sigma = compute_relative_gap_ratio(w_base, w_inst)
        layer_idx = extract_layer_idx(name)
        category = categorize_param(name)

        results.append({
            "name": name,
            "sigma": sigma,
            "layer": layer_idx,
            "category": category,
            "shape": list(w_base.shape),
            "numel": w_base.numel(),
        })

        layer_results[layer_idx].append((name, sigma))
        category_results[category].append(sigma)

    # Overall statistics
    all_sigmas = [r["sigma"] for r in results]
    overall_sigma = np.mean(all_sigmas)
    overall_sigma_std = np.std(all_sigmas)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall relative gap ratio σ:")
    print(f"  Mean:   {overall_sigma:.6f} ({overall_sigma * 100:.4f}%)")
    print(f"  Std:    {overall_sigma_std:.6f}")
    print(f"  Min:    {min(all_sigmas):.6f}")
    print(f"  Max:    {max(all_sigmas):.6f}")
    print(f"  Median: {np.median(all_sigmas):.6f}")

    # Per-category statistics
    print(f"\nPer-component σ:")
    print(f"  {'Component':<20} {'Mean σ':>10} {'Std σ':>10} {'Count':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*6}")
    category_summary = {}
    for cat in sorted(category_results.keys()):
        sigmas = category_results[cat]
        mean_s = np.mean(sigmas)
        std_s = np.std(sigmas)
        category_summary[cat] = {"mean": mean_s, "std": std_s, "count": len(sigmas)}
        print(f"  {cat:<20} {mean_s:>10.6f} {std_s:>10.6f} {len(sigmas):>6}")

    # Per-layer statistics (excluding non-layer params like embeddings)
    print(f"\nPer-layer σ (averaged over all params in each layer):")
    layer_sigmas = {}
    for layer_idx in sorted(layer_results.keys()):
        if layer_idx < 0:
            continue
        sigmas = [s for _, s in layer_results[layer_idx]]
        mean_s = np.mean(sigmas)
        layer_sigmas[layer_idx] = mean_s

    if layer_sigmas:
        print(f"  {'Layer':<8} {'Mean σ':>10}")
        print(f"  {'-'*8} {'-'*10}")
        for layer_idx in sorted(layer_sigmas.keys()):
            print(f"  {layer_idx:<8} {layer_sigmas[layer_idx]:>10.6f}")

    # =========================================================================
    # Plot 1: Per-layer average σ (like Figure 1 in the paper)
    # =========================================================================
    if layer_sigmas:
        fig, ax = plt.subplots(figsize=(12, 5))
        layers = sorted(layer_sigmas.keys())
        sigmas_per_layer = [layer_sigmas[l] for l in layers]
        ax.bar(layers, sigmas_per_layer, color="steelblue", alpha=0.8)
        ax.axhline(y=overall_sigma, color="red", linestyle="--", label=f"Overall mean σ = {overall_sigma:.4f}")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Relative Gap Ratio σ", fontsize=12)
        ax.set_title("Weight Similarity: Per-Layer Relative Gap Ratio σ\n(Base vs Instruct)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_layer_sigma.png"), dpi=150)
        plt.close(fig)
        print(f"\nSaved plot: {os.path.join(output_dir, 'per_layer_sigma.png')}")

    # =========================================================================
    # Plot 2: Per-component type σ
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = sorted(category_summary.keys())
    means = [category_summary[c]["mean"] for c in cats]
    stds = [category_summary[c]["std"] for c in cats]
    bars = ax.bar(range(len(cats)), means, yerr=stds, color="coral", alpha=0.8, capsize=4)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Relative Gap Ratio σ", fontsize=12)
    ax.set_title("Weight Similarity: Per-Component Relative Gap Ratio σ\n(Base vs Instruct)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_component_sigma.png"), dpi=150)
    plt.close(fig)
    print(f"Saved plot: {os.path.join(output_dir, 'per_component_sigma.png')}")

    # =========================================================================
    # Plot 3: Per-layer per-component heatmap
    # =========================================================================
    if layer_sigmas:
        all_cats = sorted(set(r["category"] for r in results if r["layer"] >= 0))
        all_layers = sorted(set(r["layer"] for r in results if r["layer"] >= 0))

        heatmap_data = np.full((len(all_cats), len(all_layers)), np.nan)
        cat_to_idx = {c: i for i, c in enumerate(all_cats)}
        layer_to_idx = {l: i for i, l in enumerate(all_layers)}

        for r in results:
            if r["layer"] >= 0 and r["category"] in cat_to_idx:
                ci = cat_to_idx[r["category"]]
                li = layer_to_idx[r["layer"]]
                if np.isnan(heatmap_data[ci, li]):
                    heatmap_data[ci, li] = r["sigma"]
                else:
                    heatmap_data[ci, li] = (heatmap_data[ci, li] + r["sigma"]) / 2

        fig, ax = plt.subplots(figsize=(16, 6))
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(all_cats)))
        ax.set_yticklabels(all_cats, fontsize=10)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_title("Weight Similarity Heatmap: σ per Layer and Component\n(Base vs Instruct)", fontsize=13)
        # Show fewer x-ticks if many layers
        if len(all_layers) > 40:
            step = len(all_layers) // 20
            ax.set_xticks(range(0, len(all_layers), step))
            ax.set_xticklabels([all_layers[i] for i in range(0, len(all_layers), step)])
        else:
            ax.set_xticks(range(len(all_layers)))
            ax.set_xticklabels(all_layers)
        plt.colorbar(im, ax=ax, label="σ")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "heatmap_sigma.png"), dpi=150)
        plt.close(fig)
        print(f"Saved plot: {os.path.join(output_dir, 'heatmap_sigma.png')}")

    # Save detailed results to JSON
    summary = {
        "base_model": base_path,
        "instruct_model": instruct_path,
        "overall_sigma_mean": float(overall_sigma),
        "overall_sigma_std": float(overall_sigma_std),
        "overall_sigma_min": float(min(all_sigmas)),
        "overall_sigma_max": float(max(all_sigmas)),
        "overall_sigma_median": float(np.median(all_sigmas)),
        "per_category": {k: {"mean": float(v["mean"]), "std": float(v["std"]), "count": v["count"]}
                         for k, v in category_summary.items()},
        "per_layer": {str(k): float(v) for k, v in layer_sigmas.items()} if layer_sigmas else {},
        "per_parameter": [{
            "name": r["name"],
            "sigma": float(r["sigma"]),
            "layer": r["layer"],
            "category": r["category"],
            "shape": r["shape"],
        } for r in results],
    }

    json_path = os.path.join(output_dir, "weight_similarity_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results: {json_path}")

    print("\n" + "=" * 70)
    print(f"CONCLUSION: The average relative gap ratio σ = {overall_sigma:.4f} ({overall_sigma * 100:.2f}%)")
    if overall_sigma < 0.05:
        print("The Base and Instruct models have HIGHLY SIMILAR weights (σ < 5%).")
    elif overall_sigma < 0.1:
        print("The Base and Instruct models have MODERATELY SIMILAR weights (5% < σ < 10%).")
    else:
        print("The Base and Instruct models have NOTABLY DIFFERENT weights (σ > 10%).")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Weight Similarity Analysis between Base and Instruct models "
                    "(Shadow-FT, arXiv: 2505.12716, Section 2.3)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the Base model directory (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--instruct_model",
        type=str,
        required=True,
        help="Path to the Instruct model directory (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weight_similarity_results",
        help="Directory to save results and plots (default: weight_similarity_results)",
    )
    args = parser.parse_args()

    analyze_weight_similarity(args.base_model, args.instruct_model, args.output_dir)


if __name__ == "__main__":
    main()
