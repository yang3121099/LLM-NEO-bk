#!/usr/bin/env bash
###############################################################################
# eval_tooluse_evalscope.sh — EvalScope tool-use evaluation (BFCL + ToolBench)
#
# Prerequisites:
#   pip install 'evalscope[all]'        # or: pip install evalscope bfcl-eval
#   pip install vllm
#
# Usage:
#   bash scripts/eval_tooluse_evalscope.sh
#
# Each model is served via vLLM (auto port) and evaluated on:
#   - BFCL v3 (Berkeley Function Calling Leaderboard)
#   - ToolBench (API-Bank subset via EvalScope)
#
# Results are saved to results/evalscope/<model_abbr>/
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
EVALSCOPE_OUT="$RESULTS_DIR/evalscope"
mkdir -p "$EVALSCOPE_OUT"

# TP size (tensor parallel)
TP=1
# GPU memory utilization for vLLM
GPU_UTIL=0.9
# Max model length
MAX_MODEL_LEN=16384

###############################################################################
# Model list
# Format: "abbr|model_path"
#   - model_path can be a HuggingFace ID or a local path
#   - For local merged models, use $RESULTS_DIR as prefix
###############################################################################

# --- Resolve result paths dynamically ---
# These paths follow the pattern from generate_dataset_scripts.sh:
#   $RESULTS_DIR/<MMDD>/result-<model>-<MMDD>/<TAG>-2k-lora-rank128-lr0.0002-<suffix>/merged-<X2Y>
# Adjust MMDD to match your actual training run date.
MMDD="${RESULT_MMDD:-0326}"  # Override with: RESULT_MMDD=0727 bash scripts/eval_tooluse_evalscope.sh

LLAMA_BASE="Llama-3.1-8B"
QWEN_BASE="Qwen3-8B-Base"
LLAMA_REL="${MMDD}/result-${LLAMA_BASE}-${MMDD}"
QWEN_REL="${MMDD}/result-${QWEN_BASE}-${MMDD}"

MODELS=(
  # --- HF baselines (Instruct) ---
  "Llama-3.1-8B-Instruct-hf|meta-llama/Llama-3.1-8B-Instruct"
  "Qwen3-8B-hf|Qwen/Qwen3-8B"

  # --- dolci_mix 2k: B2I & I2I ---
  "${LLAMA_BASE}-dolci_mix-2k-B2I|${RESULTS_DIR}/${LLAMA_REL}/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I"
  "${LLAMA_BASE}-dolci_mix-2k-I2I|${RESULTS_DIR}/${LLAMA_REL}/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I"
  "${QWEN_BASE}-dolci_mix-2k-B2I|${RESULTS_DIR}/${QWEN_REL}/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I"
  "${QWEN_BASE}-dolci_mix-2k-I2I|${RESULTS_DIR}/${QWEN_REL}/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I"

  # --- nemotron_if 2k: B2I & I2I ---
  "${LLAMA_BASE}-nemotron_if-2k-B2I|${RESULTS_DIR}/${LLAMA_REL}/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I"
  "${LLAMA_BASE}-nemotron_if-2k-I2I|${RESULTS_DIR}/${LLAMA_REL}/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I"
  "${QWEN_BASE}-nemotron_if-2k-B2I|${RESULTS_DIR}/${QWEN_REL}/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I"
  "${QWEN_BASE}-nemotron_if-2k-I2I|${RESULTS_DIR}/${QWEN_REL}/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I"
)

###############################################################################
# Helper: start vLLM server, run eval, stop server
###############################################################################
VLLM_PORT=8234  # Starting port; incremented if needed

find_free_port() {
  local port=$VLLM_PORT
  while ss -tlnp | grep -q ":${port} "; do
    port=$((port + 1))
  done
  echo "$port"
}

wait_for_server() {
  local port=$1 max_wait=300 elapsed=0
  echo "  Waiting for vLLM server on port $port ..."
  while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if [[ $elapsed -ge $max_wait ]]; then
      echo "  ERROR: vLLM server did not start within ${max_wait}s"
      return 1
    fi
  done
  echo "  vLLM server ready (${elapsed}s)"
}

run_eval_for_model() {
  local abbr="$1" model_path="$2"
  local port out_dir vllm_pid

  # Check if model path exists (for local models)
  if [[ "$model_path" != */* ]] || [[ "$model_path" == /* ]]; then
    if [[ ! -d "$model_path" ]]; then
      echo "  SKIP: $abbr — model path not found: $model_path"
      return 0
    fi
  fi

  port=$(find_free_port)
  out_dir="$EVALSCOPE_OUT/$abbr"
  mkdir -p "$out_dir"

  echo ""
  echo "================================================================"
  echo "  Model: $abbr"
  echo "  Path:  $model_path"
  echo "  Port:  $port"
  echo "================================================================"

  # --- Start vLLM server ---
  echo "  Starting vLLM server ..."
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --port "$port" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype auto \
    > "$out_dir/vllm_server.log" 2>&1 &
  vllm_pid=$!

  if ! wait_for_server "$port"; then
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    echo "  FAILED to start vLLM for $abbr, skipping."
    return 1
  fi

  # --- Run BFCL evaluation ---
  echo "  Running BFCL v3 evaluation ..."
  python3 -m evalscope.run \
    --model "$abbr" \
    --api-url "http://localhost:${port}/v1" \
    --eval-type service \
    --eval-backend toolbench \
    --dataset bfcl \
    --work-dir "$out_dir/bfcl" \
    > "$out_dir/bfcl_eval.log" 2>&1 || {
      echo "  WARNING: BFCL eval failed, trying alternative config ..."
      # Alternative: use evalscope's TaskConfig directly
      python3 << PYEOF
import json, os
try:
    from evalscope.run import run_task
    from evalscope.config import TaskConfig

    task_cfg = TaskConfig(
        model="$abbr",
        api_url="http://localhost:${port}/v1",
        eval_type="service",
        datasets=["bfcl"],
        work_dir="$out_dir/bfcl",
    )
    run_task(task_cfg)
except Exception as e:
    print(f"  BFCL eval error: {e}")
    # Fallback: direct bfcl-eval CLI
    os.system(
        f'bfcl evaluate '
        f'--model "$abbr" '
        f'--api-base "http://localhost:{$port}/v1" '
        f'--output-dir "$out_dir/bfcl" '
        f'2>&1 | tee "$out_dir/bfcl_direct.log"'
    )
PYEOF
  }

  # --- Run ToolBench evaluation ---
  echo "  Running ToolBench evaluation ..."
  python3 << PYEOF
import json, os
try:
    from evalscope.run import run_task
    from evalscope.config import TaskConfig

    task_cfg = TaskConfig(
        model="$abbr",
        api_url="http://localhost:${port}/v1",
        eval_type="service",
        datasets=["toolbench"],
        work_dir="$out_dir/toolbench",
    )
    run_task(task_cfg)
    print("  ToolBench eval completed.")
except ImportError:
    print("  WARNING: evalscope not installed or TaskConfig API changed.")
    print("  Try: pip install 'evalscope[all]'")
except Exception as e:
    print(f"  ToolBench eval error: {e}")
PYEOF

  # --- Stop vLLM server ---
  echo "  Stopping vLLM server (PID=$vllm_pid) ..."
  kill "$vllm_pid" 2>/dev/null || true
  wait "$vllm_pid" 2>/dev/null || true
  sleep 2

  echo "  Done: $abbr"
  echo "  Results: $out_dir/"
}

###############################################################################
# Main
###############################################################################
echo "=========================================="
echo "  EvalScope Tool-Use Evaluation"
echo "  BFCL + ToolBench"
echo "=========================================="
echo ""
echo "Models to evaluate: ${#MODELS[@]}"
echo "Output: $EVALSCOPE_OUT/"
echo ""

# Check dependencies
if ! python3 -c "import evalscope" 2>/dev/null; then
  echo "WARNING: evalscope not installed. Install with:"
  echo "  pip install 'evalscope[all]'"
  echo ""
fi

if ! python3 -c "import vllm" 2>/dev/null; then
  echo "WARNING: vllm not installed. Install with:"
  echo "  pip install vllm"
  echo ""
fi

for ENTRY in "${MODELS[@]}"; do
  IFS='|' read -r ABBR MODEL_PATH <<< "$ENTRY"
  run_eval_for_model "$ABBR" "$MODEL_PATH" || true
done

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="
echo "Results saved to: $EVALSCOPE_OUT/"
echo ""
echo "To view results:"
echo "  ls -la $EVALSCOPE_OUT/*/"
