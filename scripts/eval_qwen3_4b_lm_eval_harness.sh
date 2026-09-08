#!/usr/bin/env bash
###############################################################################
# eval_qwen3_4b_lm_eval_harness.sh
#
# lm-evaluation-harness 对比实验 (text-completion mode, no chat template)
# 模型: Qwen3-4B, Qwen3-4B-PTQTP-1.58b, Qwen3-1.7B, Qwen3-1.7B-PTQTP-1.58b
#
# Few-shot settings aligned with OpenCompass:
#   0-shot: arc_challenge, arc_easy, piqa
#   5-shot: boolq, winogrande
#  10-shot: hellaswag
###############################################################################
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/LLM-NEO-bk/outputs/lm-eval-harness-qwen3-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "Qwen/Qwen3-4B"
    "yang31210999/Qwen3-4B-PTQTP-1.58b"
    "Qwen/Qwen3-1.7B"
    "yang31210999/Qwen3-1.7B-PTQTP-1.58b"
)

# Task groups by few-shot count (aligned with OpenCompass configs)
TASKS_0SHOT="arc_challenge,arc_challenge_llama,arc_easy,piqa"
TASKS_5SHOT="boolq,winogrande"
TASKS_10SHOT="hellaswag"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   lm-evaluation-harness  Qwen3 Comparison                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Few-shot aligned with OpenCompass:                          ║"
echo "║   0-shot: arc_challenge, arc_challenge_llama, arc_easy, piqa  ║"
echo "║   5-shot: boolq, winogrande                                  ║"
echo "║  10-shot: hellaswag                                          ║"
echo "║  Models: Qwen3-4B / 4B-PTQTP / 1.7B / 1.7B-PTQTP           ║"
echo "║  Mode: text-completion (no chat template)                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for MODEL in "${MODELS[@]}"; do
    ABBR=$(basename "$MODEL")

    echo "=========================================="
    echo "  Evaluating: $MODEL"
    echo "=========================================="

    # --- 0-shot: arc_challenge, arc_easy, piqa ---
    echo "[0-shot] $TASKS_0SHOT"
    lm_eval --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True" \
        --tasks "$TASKS_0SHOT" \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/${ABBR}/0shot" \
        2>&1 | tee "${OUTPUT_DIR}/${ABBR}_0shot.log"

    # --- 5-shot: boolq, winogrande ---
    echo "[5-shot] $TASKS_5SHOT"
    lm_eval --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True" \
        --tasks "$TASKS_5SHOT" \
        --batch_size auto \
        --num_fewshot 5 \
        --output_path "${OUTPUT_DIR}/${ABBR}/5shot" \
        2>&1 | tee "${OUTPUT_DIR}/${ABBR}_5shot.log"

    # --- 10-shot: hellaswag ---
    echo "[10-shot] $TASKS_10SHOT"
    lm_eval --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True" \
        --tasks "$TASKS_10SHOT" \
        --batch_size auto \
        --num_fewshot 10 \
        --output_path "${OUTPUT_DIR}/${ABBR}/10shot" \
        2>&1 | tee "${OUTPUT_DIR}/${ABBR}_10shot.log"

    echo ""
    echo "[DONE] $ABBR → ${OUTPUT_DIR}/${ABBR}"
    echo ""
done

echo "=========================================="
echo "  All evaluations complete!"
echo "  Results: ${OUTPUT_DIR}"
echo "=========================================="
