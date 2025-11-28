#!/bin/bash
# batch_merge_lora.sh - 批量执行多LoRA融合
# 命名规则: {base_type}-{adapter1}+{adapter2}-{ratio}

# ============= B2I 融合 (Base训练的LoRA -> Instruct模型) =============
mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-1.0"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 1.0 \
    --adapter_ratio2 1.0 \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-Shadow_2k+medical_o1_reasoning-1.0" \
    --template "qwen3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-0.5"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 0.5 \
    --adapter_ratio2 0.5 \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-Shadow_2k+medical_o1_reasoning-0.5" \
    --template "qwen3"

# ============= B2B 融合 (Base训练的LoRA -> Base模型) =============
mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-1.0"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 1.0 \
    --adapter_ratio2 1.0 \
    --target_base "Qwen/Qwen3-8B-Base" \
    --merge_tag "B2B-Shadow_2k+medical_o1_reasoning-1.0" \
    --template "qwen3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-0.5"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 0.5 \
    --adapter_ratio2 0.5 \
    --target_base "Qwen/Qwen3-8B-Base" \
    --merge_tag "B2B-Shadow_2k+medical_o1_reasoning-0.5" \
    --template "qwen3"

# ============= I2I 融合 (Instruct训练的LoRA -> Instruct模型) =============
mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-1.0"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 1.0 \
    --adapter_ratio2 1.0 \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-Shadow_2k+medical_o1_reasoning-1.0" \
    --template "qwen3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-0.5"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
    --adapter_path2 "/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-medical_o1_reasoning" \
    --adapter_ratio 0.5 \
    --adapter_ratio2 0.5 \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-Shadow_2k+medical_o1_reasoning-0.5" \
    --template "qwen3"

echo "===== All merging complete! ====="
