#!/usr/bin/env python3
# merge_delta_lora.py - 支持模型差值(delta)与LoRA权重融合
"""
功能：
1. 计算两个完整模型之间的差值 delta = model_a - model_b
2. 将delta与LoRA权重按比例融合
3. 应用到基座模型上并导出

使用场景：
- RE-Adapt: (Qwen3-8B - Qwen3-8B-Base) * ratio + LoRA * ratio
- LoRE: (LoRE-Adapt-model - Qwen3-8B-Base) * ratio + LoRA * ratio
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import json
import shutil
from tqdm import tqdm
import gc


def load_model_state_dict(model_path, device="cpu", dtype=torch.float32):
    """加载模型的state_dict，支持safetensors和bin格式"""
    model_path = Path(model_path)
    state_dict = {}
    
    # 尝试加载 safetensors 格式
    safetensor_files = list(model_path.glob("*.safetensors"))
    if safetensor_files:
        print(f"  Found {len(safetensor_files)} safetensors files")
        for f in tqdm(safetensor_files, desc="  Loading safetensors"):
            with safe_open(f, framework="pt", device=device) as sf:
                for key in sf.keys():
                    state_dict[key] = sf.get_tensor(key).to(dtype)
        return state_dict
    
    # 尝试加载 bin 格式
    bin_files = list(model_path.glob("pytorch_model*.bin")) + list(model_path.glob("model*.bin"))
    if bin_files:
        print(f"  Found {len(bin_files)} bin files")
        for f in tqdm(bin_files, desc="  Loading bin files"):
            w = torch.load(f, map_location=device)
            for k, v in w.items():
                state_dict[k] = v.to(dtype)
        return state_dict
    
    # 如果是HuggingFace模型ID，直接加载
    print(f"  Loading from HuggingFace: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    state_dict = {k: v.to(dtype) for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return state_dict


def compute_model_delta(model_a_path, model_b_path, dtype=torch.float32):
    """计算两个模型的差值: delta = model_a - model_b"""
    print(f"[step] Computing delta: {model_a_path} - {model_b_path}")
    
    print(f"  Loading model A: {model_a_path}")
    state_dict_a = load_model_state_dict(model_a_path, dtype=dtype)
    
    print(f"  Loading model B: {model_b_path}")
    state_dict_b = load_model_state_dict(model_b_path, dtype=dtype)
    
    # 计算差值
    delta = {}
    all_keys = set(state_dict_a.keys()) | set(state_dict_b.keys())
    
    for key in tqdm(all_keys, desc="  Computing delta"):
        if key in state_dict_a and key in state_dict_b:
            if state_dict_a[key].shape == state_dict_b[key].shape:
                delta[key] = state_dict_a[key] - state_dict_b[key]
            else:
                print(f"    [warning] Shape mismatch for {key}, skipping")
        elif key in state_dict_a:
            delta[key] = state_dict_a[key]
        # 如果只在B中存在，忽略
    
    # 清理内存
    del state_dict_a, state_dict_b
    gc.collect()
    
    print(f"  Delta computed with {len(delta)} parameters")
    return delta


def load_lora_weights(adapter_path):
    """加载LoRA权重文件"""
    adapter_path = Path(adapter_path)
    
    # 尝试加载 safetensors 格式
    safetensor_files = list(adapter_path.glob("adapter_model*.safetensors"))
    if safetensor_files:
        weights = {}
        for f in safetensor_files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weights[key] = sf.get_tensor(key)
        return weights
    
    # 尝试加载 bin 格式
    bin_files = list(adapter_path.glob("adapter_model*.bin"))
    if bin_files:
        weights = {}
        for f in bin_files:
            w = torch.load(f, map_location="cpu")
            weights.update(w)
        return weights
    
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def load_lora_config(adapter_path):
    """加载LoRA配置"""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def merge_lora_to_base_weights(lora_weights, lora_config, base_state_dict, lora_ratio=1.0):
    """
    将LoRA权重合并到基座模型权重中
    LoRA: W' = W + BA * scaling * ratio
    """
    print(f"[step] Merging LoRA weights with ratio {lora_ratio}")
    
    # 获取LoRA配置
    lora_alpha = lora_config.get("lora_alpha", 16)
    r = lora_config.get("r", 8)
    scaling = lora_alpha / r
    
    print(f"  LoRA config: r={r}, alpha={lora_alpha}, scaling={scaling}")
    
    # 解析LoRA权重结构
    # 典型的key格式: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    lora_a_weights = {}
    lora_b_weights = {}
    
    for key, weight in lora_weights.items():
        if "lora_A" in key:
            # 提取原始层的名称
            base_key = key.replace(".lora_A.weight", ".weight")
            base_key = base_key.replace("base_model.model.", "")
            lora_a_weights[base_key] = weight.float()
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.weight", ".weight")
            base_key = base_key.replace("base_model.model.", "")
            lora_b_weights[base_key] = weight.float()
    
    # 合并LoRA到基座权重
    merged_count = 0
    for base_key in tqdm(lora_a_weights.keys(), desc="  Merging LoRA"):
        if base_key in lora_b_weights and base_key in base_state_dict:
            lora_a = lora_a_weights[base_key]
            lora_b = lora_b_weights[base_key]
            
            # LoRA: delta_W = B @ A * scaling
            delta_w = (lora_b @ lora_a) * scaling * lora_ratio
            
            # 确保形状匹配
            if delta_w.shape == base_state_dict[base_key].shape:
                base_state_dict[base_key] = base_state_dict[base_key].float() + delta_w
                merged_count += 1
            else:
                print(f"    [warning] Shape mismatch for {base_key}: {delta_w.shape} vs {base_state_dict[base_key].shape}")
    
    print(f"  Merged {merged_count} LoRA layers")
    return base_state_dict


def apply_delta_to_base(base_state_dict, delta, delta_ratio=1.0):
    """将delta应用到基座模型权重"""
    print(f"[step] Applying delta with ratio {delta_ratio}")
    
    applied_count = 0
    for key in tqdm(delta.keys(), desc="  Applying delta"):
        if key in base_state_dict:
            if delta[key].shape == base_state_dict[key].shape:
                base_state_dict[key] = base_state_dict[key].float() + delta[key].float() * delta_ratio
                applied_count += 1
    
    print(f"  Applied delta to {applied_count} parameters")
    return base_state_dict


def save_model(state_dict, output_path, reference_model_path, dtype=torch.float16):
    """保存模型"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[step] Saving model to {output_path}")
    
    # 转换数据类型
    for key in state_dict:
        state_dict[key] = state_dict[key].to(dtype)
    
    # 分片保存 (每个文件约5GB)
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB in bytes
    current_shard = {}
    current_size = 0
    shard_idx = 1
    index = {"weight_map": {}}
    
    keys = list(state_dict.keys())
    for key in tqdm(keys, desc="  Saving shards"):
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > max_shard_size and current_shard:
            # 保存当前分片
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current_shard, output_path / shard_name)
            for k in current_shard:
                index["weight_map"][k] = shard_name
            current_shard = {}
            current_size = 0
            shard_idx += 1
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # 保存最后一个分片
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-{shard_idx:05d}.safetensors"
        save_file(current_shard, output_path / shard_name)
        for k in current_shard:
            index["weight_map"][k] = shard_name
    
    # 更新分片文件名中的总数
    for i in range(1, shard_idx + 1):
        old_name = output_path / f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = output_path / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old_name.exists():
            old_name.rename(new_name)
            # 更新index
            for k, v in index["weight_map"].items():
                if v == f"model-{i:05d}-of-XXXXX.safetensors":
                    index["weight_map"][k] = f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
    
    # 保存index
    index["metadata"] = {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())}
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    # 复制tokenizer和config
    ref_path = Path(reference_model_path)
    
    # 如果是HuggingFace ID，下载配置文件
    if not ref_path.exists():
        print(f"  Downloading config and tokenizer from {reference_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(reference_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        config = AutoConfig.from_pretrained(reference_model_path, trust_remote_code=True)
        config.save_pretrained(output_path)
    else:
        # 复制本地文件
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                      "special_tokens_map.json", "vocab.json", "merges.txt",
                      "generation_config.json"]:
            src = ref_path / fname
            if src.exists():
                shutil.copy(src, output_path / fname)
    
    print(f"  Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="模型差值(Delta)与LoRA融合工具")
    
    # Delta相关参数
    parser.add_argument("--model_a", required=True, 
                        help="Delta的被减数模型路径 (model_a - model_b = delta)")
    parser.add_argument("--model_b", required=True,
                        help="Delta的减数模型路径，通常是基座模型")
    parser.add_argument("--delta_ratio", type=float, default=0.5,
                        help="Delta的系数")
    
    # LoRA相关参数
    parser.add_argument("--lora_path", required=True,
                        help="LoRA adapter路径")
    parser.add_argument("--lora_ratio", type=float, default=0.5,
                        help="LoRA的系数")
    
    # 输出相关参数
    parser.add_argument("--target_base", required=True,
                        help="目标基座模型路径（融合结果将基于此模型）")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录")
    parser.add_argument("--merge_tag", default="merged",
                        help="输出目录的标签后缀")
    
    args = parser.parse_args()
    
    # 确定输出路径
    output_path = Path(args.output_dir) / f"merged-{args.merge_tag}"
    
    # 检查是否已存在
    if output_path.exists():
        if any(f.suffix == ".safetensors" for f in output_path.iterdir()):
            print(f"[skip] {output_path} already has safetensors files, skipping.")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Delta + LoRA Merge Tool")
    print("=" * 60)
    print(f"Delta: ({args.model_a}) - ({args.model_b}) * {args.delta_ratio}")
    print(f"LoRA: {args.lora_path} * {args.lora_ratio}")
    print(f"Target base: {args.target_base}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Step 1: 计算Delta
    print("\n[1/4] Computing model delta...")
    delta = compute_model_delta(args.model_a, args.model_b, dtype=torch.float32)
    
    # Step 2: 加载目标基座模型
    print("\n[2/4] Loading target base model...")
    base_state_dict = load_model_state_dict(args.target_base, dtype=torch.float32)
    
    # Step 3: 应用Delta
    print("\n[3/4] Applying delta to base model...")
    base_state_dict = apply_delta_to_base(base_state_dict, delta, args.delta_ratio)
    
    # 清理delta内存
    del delta
    gc.collect()
    
    # Step 4: 加载并合并LoRA
    print("\n[4/4] Loading and merging LoRA...")
    lora_weights = load_lora_weights(args.lora_path)
    lora_config = load_lora_config(args.lora_path)
    
    if lora_config is None:
        raise ValueError(f"Cannot find adapter_config.json in {args.lora_path}")
    
    base_state_dict = merge_lora_to_base_weights(
        lora_weights, lora_config, base_state_dict, args.lora_ratio
    )
    
    # 清理LoRA内存
    del lora_weights
    gc.collect()
    
    # Step 5: 保存模型
    print("\n[5/5] Saving merged model...")
    save_model(base_state_dict, output_path, args.target_base)
    
    # 保存合并配置信息
    merge_info = {
        "delta": {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "ratio": args.delta_ratio
        },
        "lora": {
            "path": args.lora_path,
            "ratio": args.lora_ratio
        },
        "target_base": args.target_base,
        "merge_tag": args.merge_tag
    }
    with open(output_path / "merge_info.json", "w") as f:
        json.dump(merge_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"[complete] Merged model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
