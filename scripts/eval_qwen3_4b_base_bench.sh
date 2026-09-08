#!/usr/bin/env bash
###############################################################################
# eval_qwen3_4b_base_bench.sh
#
# Evaluate Qwen3-4B (Instruct) and Qwen3-4B-Base on 6 base benchmarks:
#   ARC-c (926652), ARC-e (1e0de5), BoolQ (ba58ea),
#   hellaswag (e42710), piqa (1194eb), winogrande (6447e6)
#
# Usage:
#   bash scripts/eval_qwen3_4b_base_bench.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

###############################################################################
# Step 1: Environment setup
###############################################################################
echo "============================================"
echo "  Step 1: Environment Setup"
echo "============================================"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate factory 2>/dev/null || {
    echo "[WARN] conda activate failed, running setup_env.sh first..."
    bash "$WORKSPACE/setup_env.sh"
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate factory
}

echo "[INFO] Python: $(which python3)"
echo "[INFO] Environment ready."

###############################################################################
# Step 2: Run OpenCompass eval
###############################################################################
echo ""
echo "============================================"
echo "  Step 2: Running OpenCompass Evaluation"
echo "============================================"
echo ""
echo "  Models:    Qwen3-4B (Instruct) + Qwen3-4B-Base"
echo "  Datasets:  ARC-c, ARC-e, BoolQ, hellaswag, piqa, winogrande"
echo "  Timestamp: $TIMESTAMP"
echo ""

cd "$WORKSPACE/opencompass"

python3 ./run.py ./eval_qwen3_4b_base_bench.py \
    -r "$TIMESTAMP" \
    --dump-eval-details

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results: $WORKSPACE/opencompass/outputs/eval-qwen3-4b-base-bench/"
echo "============================================"
