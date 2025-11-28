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


def merge_and_export(base_model_path, merged_adapter_path, output_path, template="llama3"):
    """将合并后的adapter融合到基座模型并导出"""
    print(f"[loading] Base model from {base_model_path}")
    
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print(f"[loading] Merged adapter from {merged_adapter_path}")
    # 加载合并后的adapter
    model = PeftModel.from_pretrained(model, merged_adapter_path)
    
    print("[merging] Merging adapter into base model...")
    # 融合并卸载adapter
    model = model.merge_and_unload()
    
    print(f"[saving] Saving merged model to {output_path}")
    # 保存合并后的完整模型
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
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
