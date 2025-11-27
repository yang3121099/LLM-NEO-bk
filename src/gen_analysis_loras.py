import os
import shutil
import torch
from safetensors.torch import load_file, save_file

# ================= 实验配置列表 =================
EXPERIMENTS = [
    {
        "id": "CodeZ1", # 命名标识
        # 1123 Code_Z1 路径
        "path_a": "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/B-2k-lora-rank128-lr0.0002-Code_Z1",
        "path_b": "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/I-2k-lora-rank128-lr0.0002-Code_Z1",
    },
    {
        "id": "Shadow2k", # 命名标识
        # 1122 Shadow_2k 路径 (注意这里使用了您提供的 /workspace 路径)
        "path_a": "/dockerdata/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank128-lr0.0002-Shadow_2k",
        "path_b": "/dockerdata/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank128-lr0.0002-Shadow_2k",
    }
]

# 统一输出总目录
OUTPUT_ROOT = "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124"

# 模块定义
KEYS_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"]
KEYS_MLP = ["gate_proj", "up_proj", "down_proj", "mlp"]
# ===========================================

def get_layer_index(key):
    parts = key.split('.')
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1

def get_max_layer(state_dict):
    max_layer = 0
    for k in state_dict.keys():
        idx = get_layer_index(k)
        if idx > max_layer:
            max_layer = idx
    return max_layer

def is_match(key, keyword_list):
    return any(k in key for k in keyword_list)

def save_new_lora(state_dict, folder_name, base_config_path):
    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    # 尝试复制 config
    src_cfg = os.path.join(base_config_path, "adapter_config.json")
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, os.path.join(output_dir, "adapter_config.json"))
    else:
        print(f"⚠️ 警告: 找不到 config {src_cfg}")
        
    print(f"✅ 生成: {folder_name} (Keys: {len(state_dict)})")

def generate_mixed_state_dict(dict_a, dict_b, all_keys, condition_func_a, condition_func_b=None):
    new_dict = {}
    for k in all_keys:
        # 获取 reference tensor 以确定 shape 和 device
        if k in dict_a:
            ref_tensor = dict_a[k]
        elif k in dict_b:
            ref_tensor = dict_b[k]
        else:
            continue # Should not happen based on all_keys definition

        if k in dict_a and condition_func_a(k):
            new_dict[k] = dict_a[k]
        elif condition_func_b and k in dict_b and condition_func_b(k):
            new_dict[k] = dict_b[k]
        else:
            # 补零 (Zero Padding) 以避免 missing keys 报错
            new_dict[k] = torch.zeros_like(ref_tensor)
            
    return new_dict

def process_experiment(exp_config):
    exp_id = exp_config["id"]
    path_a = exp_config["path_a"]
    path_b = exp_config["path_b"]
    
    print(f"\n>>> 开始处理实验组: {exp_id}")
    print(f"    Source A: {path_a}")
    print(f"    Source B: {path_b}")

    try:
        dict_a = load_file(os.path.join(path_a, "adapter_model.safetensors"))
        dict_b = load_file(os.path.join(path_b, "adapter_model.safetensors"))
    except Exception as e:
        print(f"❌ 加载失败，跳过 {exp_id}: {e}")
        return

    all_keys = set(dict_a.keys()) | set(dict_b.keys())
    max_layer = get_max_layer(dict_a)
    mid_point = (max_layer + 1) // 2
    
    # helper for saving with prefix
    def save(sd, suffix):
        save_new_lora(sd, f"{exp_id}_{suffix}", path_a)

    # ================= 1. 单体拆解 (Ablation) =================
    sources = [("ModelA", dict_a), ("ModelB", dict_b)]
    for name, data in sources:
        # Attn Only
        sd = generate_mixed_state_dict(data, {}, all_keys, lambda k: is_match(k, KEYS_ATTENTION))
        save(sd, f"Single_{name}_AttnOnly")
        
        # Mlp Only
        sd = generate_mixed_state_dict(data, {}, all_keys, lambda k: is_match(k, KEYS_MLP))
        save(sd, f"Single_{name}_MlpOnly")
        
        # Shallow
        sd = generate_mixed_state_dict(data, {}, all_keys, lambda k: get_layer_index(k) < mid_point and get_layer_index(k) != -1)
        save(sd, f"Single_{name}_Shallow")
        
        # Deep
        sd = generate_mixed_state_dict(data, {}, all_keys, lambda k: get_layer_index(k) >= mid_point)
        save(sd, f"Single_{name}_Deep")

    # ================= 2. 混合组合 (Composition) =================
    # Hybrid 1: Attn(A) + MLP(B)
    sd = generate_mixed_state_dict(dict_a, dict_b, all_keys,
                                 condition_func_a=lambda k: is_match(k, KEYS_ATTENTION),
                                 condition_func_b=lambda k: not is_match(k, KEYS_ATTENTION))
    save(sd, "Hybrid_Attn-A_Mlp-B")

    # Hybrid 2: MLP(A) + Attn(B)
    sd = generate_mixed_state_dict(dict_a, dict_b, all_keys,
                                 condition_func_a=lambda k: is_match(k, KEYS_MLP),
                                 condition_func_b=lambda k: not is_match(k, KEYS_MLP))
    save(sd, "Hybrid_Mlp-A_Attn-B")
    
    # Hybrid 3: Shallow(A) + Deep(B)
    sd = generate_mixed_state_dict(dict_a, dict_b, all_keys,
                                 condition_func_a=lambda k: get_layer_index(k) != -1 and get_layer_index(k) < mid_point,
                                 condition_func_b=lambda k: not (get_layer_index(k) != -1 and get_layer_index(k) < mid_point))
    save(sd, "Hybrid_Shallow-A_Deep-B")

    # Hybrid 4: Deep(A) + Shallow(B)
    sd = generate_mixed_state_dict(dict_a, dict_b, all_keys,
                                 condition_func_a=lambda k: get_layer_index(k) >= mid_point,
                                 condition_func_b=lambda k: not (get_layer_index(k) >= mid_point))
    save(sd, "Hybrid_Deep-A_Shallow-B")

def main():
    for exp in EXPERIMENTS:
        process_experiment(exp)
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()
