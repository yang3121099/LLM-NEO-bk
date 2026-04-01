#!/usr/bin/env bash
###############################################################################
# download_models.sh — Multi-threaded model download to HuggingFace cache
#
# Uses hf_transfer for 5-10x speedup, plus parallel model downloads.
#
# Usage:
#   nohup bash scripts/download_models.sh > download_models.log 2>&1 &
#   tail -f download_models.log
#
# Flags:
#   PARALLEL=4       Max parallel model downloads (default: 4)
#   HF_HUB_ENABLE_HF_TRANSFER=1  Already set by script (Rust multi-thread)
###############################################################################
set -euo pipefail

PARALLEL="${PARALLEL:-4}"

# Install hf_transfer if not present (Rust-based multi-thread downloader)
if ! python3 -c "import hf_transfer" 2>/dev/null; then
  echo "Installing hf_transfer for multi-threaded download ..."
  pip install hf_transfer -q
fi

# Enable hf_transfer: uses multiple connections per file (5-10x faster)
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Model Download Started: $(date) ==="
echo "=== hf_transfer: enabled (multi-thread per file) ==="
echo "=== Parallel models: $PARALLEL ==="
echo ""

MODELS=(
  "Qwen/Qwen2.5-32B"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen3-30B-A3B-Base"
  "Qwen/Qwen3-30B-A3B"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B-Base"
  "Qwen/Qwen3-8B"
)

download_one() {
  local model="$1"
  echo "[START] $model  $(date +%H:%M:%S)"
  if python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('${model}', resume_download=True)
" 2>&1; then
    echo "[DONE]  $model  $(date +%H:%M:%S)"
  else
    echo "[FAIL]  $model  $(date +%H:%M:%S)"
  fi
}

# Launch downloads in parallel, throttled to $PARALLEL at a time
PIDS=()
for MODEL in "${MODELS[@]}"; do
  # Throttle
  while (( ${#PIDS[@]} >= PARALLEL )); do
    new_pids=()
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
    done
    PIDS=("${new_pids[@]}")
    (( ${#PIDS[@]} >= PARALLEL )) && sleep 5
  done

  download_one "$MODEL" &
  PIDS+=($!)
done

# Wait for all
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo ""
echo "=== All downloads complete: $(date) ==="
