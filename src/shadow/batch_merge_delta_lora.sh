#!/bin/bash
# batch_merge_delta_lora.sh - 执行Delta + LoRA融合
# 
# RE-Adapt: (Qwen3-8B - Qwen3-8B-Base) * 0.5 + LoRA * 0.5
# LoRE: (LoRE-Adapt-model - Qwen3-8B-Base) * 0.5 + LoRA * 0.5

SCRIPT_PATH="/dockerdata/LLM-NEO-bk/src/shadow/merge_delta_lora.py"
RESULT_DIR="/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127"
LORA_PATH="/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3"

# ============= RE-Adapt =============
# Delta = Qwen3-8B (Instruct) - Qwen3-8B-Base
# 公式: Base + (Instruct - Base) * 0.5 + LoRA * 0.5
echo "===== RE-Adapt: (Qwen3-8B - Qwen3-8B-Base)*0.5 + Shadow_2k_re_adapt*0.5 ====="
python3 ${SCRIPT_PATH} \
    --model_a "Qwen/Qwen3-8B" \
    --model_b "Qwen/Qwen3-8B-Base" \
    --delta_ratio 0.5 \
    --lora_path "${LORA_PATH}" \
    --lora_ratio 0.5 \
    --target_base "Qwen/Qwen3-8B-Base" \
    --output_dir "${RESULT_DIR}" \
    --merge_tag "RE-Adapt-Shadow_2k_re_adapt-0.5+0.5"

# ============= LoRE =============
# Delta = LoRE-Adapt-model - Qwen3-8B-Base  
# 公式: Base + (LoRE-Adapt - Base) * 0.5 + LoRA * 0.5
echo "===== LoRE: (ICLR-LoRE-Adapt - Qwen3-8B-Base)*0.5 + Shadow_2k_re_adapt*0.5 ====="
python3 ${SCRIPT_PATH} \
    --model_a "yang31210999/ICLR-1123_OC-H200-Qwen3-LoRE-Adapt-d1" \
    --model_b "Qwen/Qwen3-8B-Base" \
    --delta_ratio 0.5 \
    --lora_path "${LORA_PATH}" \
    --lora_ratio 0.5 \
    --target_base "Qwen/Qwen3-8B-Base" \
    --output_dir "${RESULT_DIR}" \
    --merge_tag "LoRE-Shadow_2k_re_adapt-0.5+0.5"

echo "===== All Delta+LoRA merging complete! ====="
echo ""
echo "Output directories:"
echo "  - ${RESULT_DIR}/merged-RE-Adapt-Shadow_2k_re_adapt-0.5+0.5"
echo "  - ${RESULT_DIR}/merged-LoRE-Shadow_2k_re_adapt-0.5+0.5"
