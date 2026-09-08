#!/usr/bin/env bash
###############################################################################
# eval_qwen3_4b_lm_eval_harness.sh
#
# lm-evaluation-harness 对比实验 (Thinking disabled)
# 模型: Qwen3-4B, Qwen3-4B-PTQTP-1.58b, Qwen3-1.7B, Qwen3-1.7B-PTQTP-1.58b
# 数据集: arc_challenge, arc_easy, boolq, hellaswag, piqa, winogrande
###############################################################################
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/LLM-NEO-bk/outputs/lm-eval-harness-qwen3-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

TASKS="arc_challenge,arc_easy,boolq,hellaswag,piqa,winogrande"

MODELS=(
    "Qwen/Qwen3-4B"
    "yang31210999/Qwen3-4B-PTQTP-1.58b"
    "Qwen/Qwen3-1.7B"
    "yang31210999/Qwen3-1.7B-PTQTP-1.58b"
)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   lm-evaluation-harness  Qwen3 Comparison               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Tasks: arc_challenge, arc_easy, boolq, hellaswag,      ║"
echo "║         piqa, winogrande  (Thinking DISABLED)            ║"
echo "║  Models: Qwen3-4B / 4B-PTQTP / 1.7B / 1.7B-PTQTP       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

for MODEL in "${MODELS[@]}"; do
    ABBR=$(basename "$MODEL")
    echo "=========================================="
    echo "  Evaluating: $MODEL"
    echo "=========================================="

    lm_eval --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True,chat_template_kwargs={enable_thinking:False}" \
        --tasks "$TASKS" \
        --batch_size auto \
        --num_fewshot 0 \
        --apply_chat_template \
        --output_path "${OUTPUT_DIR}/${ABBR}" \
        2>&1 | tee "${OUTPUT_DIR}/${ABBR}.log"

    echo ""
    echo "[DONE] $ABBR → ${OUTPUT_DIR}/${ABBR}"
    echo ""
done

echo "=========================================="
echo "  All evaluations complete!"
echo "  Results: ${OUTPUT_DIR}"
echo "=========================================="
