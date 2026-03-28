#!/usr/bin/env bash
###############################################################################
# download_models.sh — Pre-download model weights to HuggingFace cache
#
# Usage:
#   nohup bash scripts/download_models.sh > download_models.log 2>&1 &
#   tail -f download_models.log
###############################################################################
set -euo pipefail

echo "=== Model Download Started: $(date) ==="

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B-Base"
  "Qwen/Qwen3-8B"
)

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo ">>> Downloading: $MODEL  [$(date)]"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', resume_download=True)
print('  Done: ${MODEL}')
" && echo "<<< OK: $MODEL" || echo "<<< FAILED: $MODEL"
done

echo ""
echo "=== All downloads complete: $(date) ==="
