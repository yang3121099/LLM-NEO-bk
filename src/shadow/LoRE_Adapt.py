import os
import shutil
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import re
import json

from huggingface_hub import save_torch_state_dict


def is_linear_param(name):
    """Return True if the parameter belongs to a linear/projection layer."""
    patterns = [
        r"k_proj", r"q_proj", r"v_proj",
        r"o_proj", r"up_proj", r"gate_proj", r"down_proj"
    ]
    return any(re.search(pattern, name.lower()) for pattern in patterns)


def load_weights(model_path):
    """Load weights from *.safetensors or pytorch_model.bin."""
    weights = {}

    safetensor_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")
    ]

    pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
    index_json = os.path.join(model_path, "pytorch_model.bin.index.json")

    if safetensor_files:
        for model_file_path in safetensor_files:
            with safe_open(model_file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)
    elif os.path.exists(pytorch_bin):
        print(f"Loading PyTorch model from {pytorch_bin}")
        weights = torch.load(pytorch_bin, map_location="cpu")
    elif os.path.exists(index_json):
        print(f"Loading shard bins listed in index {index_json}")
        shard_files = sorted(
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if re.match(r"pytorch_model-\d{5}-of-\d{5}\.bin", f)
        )
        for shard_path in shard_files:
            print(f"Loading shard {shard_path}")
            weights.update(torch.load(shard_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"No weights found in {model_path}")

    return weights


def save_safetensor_weights(model_path, weights, max_shard_size="5GB"):
    """Save weights as SafeTensors."""
    os.makedirs(model_path, exist_ok=True)
    save_torch_state_dict(
        state_dict=weights,
        save_directory=model_path,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
        safe_serialization=True,
    )


def copy_tokenizer_and_config(src_dir, dst_dir):
    """Copy tokenizer/config files."""
    for filename in os.listdir(src_dir):
        if filename.endswith(".safetensors.index.json"):
            continue
        if filename.startswith(("config", "tokenizer", "special", "generation")) \
           or filename.endswith(('.md', '.json', '.py')):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)


def compute_lore_svd(delta_tensor, tau=0.5):
    """
    SVD分解并按tau阈值截断，返回重构后的矩阵和分析信息
    """
    if delta_tensor.dim() != 2:
        return delta_tensor, None
    
    m, n = delta_tensor.shape
    
    # SVD
    U, S, Vt = torch.linalg.svd(delta_tensor.float(), full_matrices=False)
    
    # 累积方差
    variance = S ** 2
    total_var = variance.sum()
    if total_var < 1e-10:
        return delta_tensor, None
    
    cumsum_var = variance.cumsum(0) / total_var
    
    # 截断秩
    rank = (cumsum_var < tau).sum().item() + 1
    rank = min(rank, min(m, n))
    
    # 重构
    reconstructed = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]).to(delta_tensor.dtype)
    
    # 误差
    diff_norm = (delta_tensor - reconstructed).norm().item()
    orig_norm = delta_tensor.norm().item()
    
    info = {
        'shape': (m, n),
        'rank': rank,
        'max_rank': min(m, n),
        'compression': (m * rank + rank + rank * n) / (m * n) * 100,
        'explained_var': cumsum_var[rank - 1].item(),
        'relative_error': diff_norm / orig_norm if orig_norm > 0 else 0
    }
    
    return reconstructed, info


def process_lore(instruct_model, base_model, output_dir, tau=0.5):
    """
    计算LoRE-Adapt: Base + SVD_truncate(Instruct - Base)
    """
    print(f"\n{'='*60}")
    print(f"LoRE-Adapt (tau={tau})")
    print(f"{'='*60}")
    print(f"Instruct: {instruct_model}")
    print(f"Base: {base_model}")
    print(f"Output: {output_dir}")

    # 加载权重
    print("\n[+] Loading weights...")
    instruct_weights = load_weights(instruct_model)
    base_weights = load_weights(base_model)

    # 计算LoRE delta
    print(f"\n[+] Computing LoRE (tau={tau})...")
    
    lore_delta = {}
    all_infos = []
    
    for k in instruct_weights:
        if k not in base_weights or not is_linear_param(k):
            continue
        
        delta_full = instruct_weights[k] - base_weights[k]
        delta_lore, info = compute_lore_svd(delta_full, tau=tau)
        
        lore_delta[k] = delta_lore
        if info:
            all_infos.append((k, info))

    # ========================================
    # 分析报告
    # ========================================
    print(f"\n{'='*60}")
    print("LoRE 截断分析")
    print(f"{'='*60}")
    
    total_orig = sum(i['shape'][0] * i['shape'][1] for _, i in all_infos)
    total_comp = sum(i['shape'][0] * i['rank'] + i['rank'] + i['rank'] * i['shape'][1] for _, i in all_infos)
    avg_error = sum(i['relative_error'] for _, i in all_infos) / len(all_infos)
    avg_explained = sum(i['explained_var'] for _, i in all_infos) / len(all_infos)
    
    print(f"\n[总体]")
    print(f"  层数: {len(all_infos)}")
    print(f"  原始参数: {total_orig:,}")
    print(f"  压缩参数: {total_comp:,}")
    print(f"  压缩率: {total_comp/total_orig*100:.2f}%")
    print(f"  平均解释方差: {avg_explained:.4f}")
    print(f"  平均相对误差: {avg_error:.6f}")
    
    print(f"\n[采样层] (前5层)")
    print("-" * 70)
    print(f"{'Layer':<40} {'Shape':<12} {'Rank':<10} {'Error':<10}")
    print("-" * 70)
    for k, info in all_infos[:5]:
        short_name = k.split('.')[-3] + '.' + k.split('.')[-2] + '.' + k.split('.')[-1]
        print(f"{short_name:<40} {info['shape'][0]}×{info['shape'][1]:<6} {info['rank']}/{info['max_rank']:<6} {info['relative_error']:.6f}")
    print("-" * 70)

    # ========================================
    # 组合并保存
    # ========================================
    print(f"\n[+] Merging: Base + LoRE_delta...")
    
    new_weights = {
        k: (base_weights[k] + lore_delta[k]) if k in lore_delta else v
        for k, v in base_weights.items()
    }

    print(f"[+] Saving to {output_dir}...")
    save_safetensor_weights(output_dir, new_weights)
    copy_tokenizer_and_config(instruct_model, output_dir)

    print(f"\n[✓] Done! Output: {output_dir}")
    print(f"    压缩率: {total_comp/total_orig*100:.2f}%")
    print(f"    平均误差: {avg_error:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRE-Adapt: Base + SVD_truncate(Instruct - Base)")
    parser.add_argument("--instruct_model", type=str, required=True, help="Instruct model path")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tau", type=float, default=0.5, help="SVD threshold (default: 0.5)")

    args = parser.parse_args()

    process_lore(
        instruct_model=args.instruct_model,
        base_model=args.base_model,
        output_dir=args.output_dir,
        tau=args.tau
    )
