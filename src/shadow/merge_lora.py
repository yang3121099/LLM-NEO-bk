#!/usr/bin/env python3
# merge_lora_dual.py - 支持多个LoRA模块带系数融合
import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from safetensors.torch import save_file
import json
import shutil
from tqdm import tqdm


def load_lora_weights(adapter_path):
    """加载LoRA权重文件"""
    adapter_path = Path(adapter_path)
    
    # 尝试加载 safetensors 格式
    safetensor_files = list(adapter_path.glob("adapter_model*.safetensors"))
    if safetensor_files:
        from safetensors import safe_open
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


def merge_lora_weights(weights_list, ratios):
    """按比例合并多个LoRA权重"""
    if len(weights_list) != len(ratios):
        raise ValueError("weights_list 和 ratios 长度必须相同")
    
    merged_weights = {}
    all_keys = set()
    for w in weights_list:
        all_keys.update(w.keys())
    
    for key in tqdm(all_keys, desc="Merging LoRA weights"):
        merged_tensor = None
        for weights, ratio in zip(weights_list, ratios):
            if key in weights:
                tensor = weights[key].float() * ratio
                if merged_tensor is None:
                    merged_tensor = tensor
                else:
                    # 确保形状匹配
                    if merged_tensor.shape == tensor.shape:
                        merged_tensor = merged_tensor + tensor
                    else:
                        print(f"[warning] Shape mismatch for {key}: {merged_tensor.shape} vs {tensor.shape}, skipping addition")
        
        if merged_tensor is not None:
            merged_weights[key] = merged_tensor
    
    return merged_weights


def save_merged_adapter(merged_weights, output_path, reference_adapter_path):
    """保存合并后的adapter"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    reference_adapter_path = Path(reference_adapter_path)
    
    # 复制 adapter_config.json
    config_src = reference_adapter_path / "adapter_config.json"
    if config_src.exists():
        shutil.copy(config_src, output_path / "adapter_config.json")
    
    # 保存合并后的权重为 safetensors 格式
    save_file(merged_weights, output_path / "adapter_model.safetensors")
    print(f"[saved] Merged adapter saved to {output_path}")


def _parse_lora_key(key):
    """Parse a LoRA checkpoint key to extract the base module path and LoRA component.

    Handles keys like:
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    Returns:
      ("model.layers.0.self_attn.q_proj", "lora_A")  or None if not a LoRA key.
    """
    # Strip leading "base_model.model." prefix (may be absent)
    k = key
    for prefix in ("base_model.model.", ):
        if k.startswith(prefix):
            k = k[len(prefix):]

    if ".lora_A." in k:
        module_path = k.split(".lora_A.")[0]
        return module_path, "lora_A"
    elif ".lora_B." in k:
        module_path = k.split(".lora_B.")[0]
        return module_path, "lora_B"
    return None


def _build_suffix_map(state_dict_keys):
    """Build a mapping from suffix (e.g. 'layers.0.self_attn.q_proj.weight') to full key.

    This allows matching LoRA module paths against target model keys even when
    there is a prefix difference (e.g. 'model.layers...' vs 'model.language_model.layers...').
    """
    suffix_map = {}
    for key in state_dict_keys:
        # Try progressively shorter suffixes starting from 'layers.'
        idx = key.find("layers.")
        if idx >= 0:
            suffix = key[idx:]
            suffix_map[suffix] = key
    return suffix_map


def merge_and_export(base_model_path, merged_adapter_path, output_path, template="llama3"):
    """将合并后的adapter融合到基座模型并导出（手动应用 LoRA delta，避免 PEFT 加载兼容性问题）"""
    print(f"[loading] Base model from {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # Load adapter config for scaling factor
    adapter_cfg_path = Path(merged_adapter_path) / "adapter_config.json"
    lora_alpha = 16  # default
    lora_rank = 8    # default
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            cfg = json.load(f)
        lora_alpha = cfg.get("lora_alpha", lora_alpha)
        lora_rank = cfg.get("r", lora_rank)
    scaling = lora_alpha / lora_rank
    print(f"[info] LoRA scaling = {lora_alpha} / {lora_rank} = {scaling}")

    # Load adapter weights
    print(f"[loading] Adapter weights from {merged_adapter_path}")
    adapter_weights = load_lora_weights(merged_adapter_path)

    # Group by module path: {module_path: {"lora_A": tensor, "lora_B": tensor}}
    lora_pairs = {}
    for key, tensor in adapter_weights.items():
        parsed = _parse_lora_key(key)
        if parsed is None:
            continue
        module_path, component = parsed
        lora_pairs.setdefault(module_path, {})[component] = tensor

    # Build suffix map for flexible key matching (handles prefix differences
    # between Base and Instruct models, e.g. 'model.layers.' vs 'model.language_model.layers.')
    state_dict = model.state_dict()
    suffix_map = _build_suffix_map(state_dict.keys())

    def _resolve_weight_key(module_path):
        """Try exact match first, then fall back to suffix-based matching."""
        exact = f"{module_path}.weight"
        if exact in state_dict:
            return exact
        # Extract suffix from 'layers.' onward
        idx = module_path.find("layers.")
        if idx >= 0:
            suffix = f"{module_path[idx:]}.weight"
            if suffix in suffix_map:
                return suffix_map[suffix]
        return None

    # Apply LoRA delta: W_new = W + scaling * (B @ A)
    applied, skipped = 0, 0
    for module_path, parts in tqdm(lora_pairs.items(), desc="Applying LoRA deltas"):
        if "lora_A" not in parts or "lora_B" not in parts:
            print(f"  [skip] Incomplete pair for {module_path}")
            skipped += 1
            continue

        weight_key = _resolve_weight_key(module_path)
        if weight_key is None:
            print(f"  [skip] Target model has no parameter matching: {module_path}")
            skipped += 1
            continue

        A = parts["lora_A"].float()  # (rank, in_features)
        B = parts["lora_B"].float()  # (out_features, rank)
        W = state_dict[weight_key].float()

        delta = (B @ A) * scaling
        if delta.shape != W.shape:
            print(f"  [skip] Shape mismatch for {weight_key}: W={W.shape}, delta={delta.shape}")
            skipped += 1
            continue

        state_dict[weight_key] = (W + delta).to(torch.float16)
        applied += 1

    print(f"[info] Applied LoRA to {applied} modules, skipped {skipped}")
    model.load_state_dict(state_dict)

    print(f"[saving] Saving merged model to {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_path)

    print(f"[done] Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="多LoRA带系数融合工具")
    
    # 支持多个adapter路径
    parser.add_argument("--adapter_path", required=True, help="第一个LoRA adapter路径")
    parser.add_argument("--adapter_path2", default=None, help="第二个LoRA adapter路径")
    parser.add_argument("--adapter_path3", default=None, help="第三个LoRA adapter路径（可选）")
    
    # 对应的系数
    parser.add_argument("--adapter_ratio", type=float, default=1.0, help="第一个adapter的系数")
    parser.add_argument("--adapter_ratio2", type=float, default=1.0, help="第二个adapter的系数")
    parser.add_argument("--adapter_ratio3", type=float, default=1.0, help="第三个adapter的系数（可选）")
    
    # 其他参数
    parser.add_argument("--target_base", required=True, help="基座模型路径")
    parser.add_argument("--merge_tag", required=True, help="合并输出目录的标签")
    parser.add_argument("--template", default="llama3", help="模板类型")
    parser.add_argument("--output_dir", default=None, help="输出目录（默认在第一个adapter同级目录）")
    parser.add_argument("--only_merge_adapter", action="store_true", 
                        help="仅合并adapter权重，不融合到基座模型")
    
    args = parser.parse_args()
    
    # 收集所有adapter路径和系数
    adapter_paths = [args.adapter_path]
    ratios = [args.adapter_ratio]
    
    if args.adapter_path2:
        adapter_paths.append(args.adapter_path2)
        ratios.append(args.adapter_ratio2)
    
    if args.adapter_path3:
        adapter_paths.append(args.adapter_path3)
        ratios.append(args.adapter_ratio3)
    
    print(f"[info] Merging {len(adapter_paths)} adapters with ratios: {ratios}")
    for i, (path, ratio) in enumerate(zip(adapter_paths, ratios)):
        print(f"  - Adapter {i+1}: {path} (ratio={ratio})")
    
    # 确定输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(args.adapter_path).parent
    
    merged_dir = output_base / f"merged-{args.merge_tag}"
    
    # 检查是否已存在
    if merged_dir.exists():
        if any(f.suffix == ".safetensors" for f in merged_dir.iterdir()):
            print(f"[skip] {merged_dir} already has safetensors files, skipping.")
            return
    
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 加载所有LoRA权重
    print("[step 1/3] Loading LoRA weights...")
    weights_list = []
    for path in adapter_paths:
        print(f"  Loading from {path}")
        weights = load_lora_weights(path)
        weights_list.append(weights)
        print(f"    Loaded {len(weights)} parameters")
    
    # 步骤2: 按比例合并权重
    print("[step 2/3] Merging weights with ratios...")
    merged_weights = merge_lora_weights(weights_list, ratios)
    print(f"  Merged {len(merged_weights)} parameters")
    
    # 保存合并后的adapter（临时目录）
    temp_adapter_dir = merged_dir / "temp_merged_adapter"
    save_merged_adapter(merged_weights, temp_adapter_dir, adapter_paths[0])
    
    if args.only_merge_adapter:
        # 仅保存合并后的adapter
        final_adapter_dir = merged_dir / "merged_adapter"
        if temp_adapter_dir != final_adapter_dir:
            shutil.move(str(temp_adapter_dir), str(final_adapter_dir))
        print(f"[done] Merged adapter saved to {final_adapter_dir}")
    else:
        # 步骤3: 融合到基座模型并导出
        print("[step 3/3] Merging into base model and exporting...")
        merge_and_export(
            base_model_path=args.target_base,
            merged_adapter_path=str(temp_adapter_dir),
            output_path=str(merged_dir),
            template=args.template
        )
        
        # 清理临时adapter目录
        shutil.rmtree(temp_adapter_dir, ignore_errors=True)
    
    # 保存合并配置信息
    merge_info = {
        "adapters": adapter_paths,
        "ratios": ratios,
        "base_model": args.target_base,
        "merge_tag": args.merge_tag,
        "template": args.template
    }
    with open(merged_dir / "merge_info.json", "w") as f:
        json.dump(merge_info, f, indent=2)
    
    print(f"[complete] All done! Output: {merged_dir}")


if __name__ == "__main__":
    main()
