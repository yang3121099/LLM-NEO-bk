#!/usr/bin/env bash
###############################################################################
# eval_tooluse_evalscope.sh — EvalScope tool-use evaluation (BFCL + ToolBench)
#
# Model list uses the SAME (abbr, path) format as OpenCompass eval configs.
# Copy entries directly between eval_*.py and this script.
#
# Prerequisites:
#   pip install 'evalscope[all]'
#   pip install vllm
#
# Usage:
#   bash scripts/eval_tooluse_evalscope.sh
#
# Results → results/evalscope/<model_abbr>/
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
EVALSCOPE_OUT="$RESULTS_DIR/evalscope"
mkdir -p "$EVALSCOPE_OUT"

# ---------------------------------------------------------------------------
# Auto-detect GPU count (same as OpenCompass eval configs + training scripts)
# ---------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
echo "INFO  Detected ${NUM_GPUS} GPU(s), using tp=${NUM_GPUS} for vLLM"

# vLLM settings
TP=$NUM_GPUS
GPU_UTIL=0.9
MAX_MODEL_LEN=8192

###############################################################################
# Model list — SAME (abbr, path) format as OpenCompass eval configs
#
# Format: "abbr|path"
# Copy (abbr, path) entries from eval_2k_*.py or eval_full_*.py and convert:
#   ('Llama-3.1-8B-dolci_mix-2k-B2I','$RESULTS_DIR/...')
#   →  "Llama-3.1-8B-dolci_mix-2k-B2I|$RESULTS_DIR/..."
###############################################################################

# ======= Original Instruct baselines =======
HF_BASELINES=(
  "Llama-3.1-8B-Instruct-hf|meta-llama/Llama-3.1-8B-Instruct"
  "Qwen3-8B-Instruct-hf|Qwen/Qwen3-8B"
)

# ======= Trained models (B2I Shadow-FT, I2I baseline) =======
# 2k experiments
TRAINED_MODELS=(
  "Llama-3.1-8B-dolci_mix-2k-B2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I"
  "Llama-3.1-8B-dolci_mix-2k-I2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I"
  "Llama-3.1-8B-nemotron_if-2k-B2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I"
  "Llama-3.1-8B-nemotron_if-2k-I2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I"
  "Qwen3-8B-dolci_mix-2k-B2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I"
  "Qwen3-8B-dolci_mix-2k-I2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I"
  "Qwen3-8B-nemotron_if-2k-B2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I"
  "Qwen3-8B-nemotron_if-2k-I2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I"
  # Tool-use experiments (dolci_instruct_tool_use, 2k samples)
  "Llama-3.1-8B-dolci_tool-2k-B2I|$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I"
  "Llama-3.1-8B-dolci_tool-2k-I2I|$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I"
  "Qwen3-8B-dolci_tool-2k-B2I|$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I"
  "Qwen3-8B-dolci_tool-2k-I2I|$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I"
  # Full-scale experiments (uncomment after training completes)
  # "Llama-3.1-8B-openr1-220k-B2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-220k-lora-rank128-lr0.0002-openr1/merged-B2I"
  # "Llama-3.1-8B-openr1-220k-I2I|$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-220k-lora-rank128-lr0.0002-openr1/merged-I2I"
  # "Qwen3-8B-openr1-220k-B2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-220k-lora-rank128-lr0.0002-openr1/merged-B2I"
  # "Qwen3-8B-openr1-220k-I2I|$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-220k-lora-rank128-lr0.0002-openr1/merged-I2I"
)

# Combine all models
MODELS=("${HF_BASELINES[@]}" "${TRAINED_MODELS[@]}")

###############################################################################
# Helper functions
###############################################################################
VLLM_PORT=8234

find_free_port() {
  local port=$VLLM_PORT
  while ss -tlnp 2>/dev/null | grep -q ":${port} "; do
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

  # Check local path existence
  if [[ "$model_path" == /* ]] && [[ ! -d "$model_path" ]]; then
    echo "  SKIP: $abbr — model path not found: $model_path"
    return 0
  fi

  port=$(find_free_port)
  out_dir="$EVALSCOPE_OUT/$abbr"
  mkdir -p "$out_dir"

  echo ""
  echo "================================================================"
  echo "  Model: $abbr"
  echo "  Path:  $model_path"
  echo "  Port:  $port  |  TP: $TP"
  echo "================================================================"

  # --- Determine tool-call parser based on model family ---
  local tool_parser="hermes"
  case "${model_path,,}" in
    *llama*)  tool_parser="llama3_json" ;;
    *qwen*)   tool_parser="hermes" ;;
    *mistral*) tool_parser="mistral" ;;
  esac
  echo "  Tool-call parser: $tool_parser"

  # --- Helper: start/stop vLLM for one benchmark ---
  # Restart between benchmarks to avoid OOM / KV cache fragmentation
  run_one_benchmark() {
    local bench_name="$1" bench_dataset="$2"
    local vllm_pid

    echo ""
    echo "  --- $bench_name ---"
    echo "  Starting vLLM server ..."
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --port "$port" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --trust-remote-code \
      --dtype auto \
      --enable-auto-tool-choice \
      --tool-call-parser "$tool_parser" \
      > "$out_dir/${bench_name}_vllm.log" 2>&1 &
    vllm_pid=$!

    if ! wait_for_server "$port"; then
      kill "$vllm_pid" 2>/dev/null || true
      wait "$vllm_pid" 2>/dev/null || true
      echo "  FAILED to start vLLM for $bench_name, skipping."
      return 1
    fi

    echo "  Running $bench_name evaluation ..."
    python3 -c "
from evalscope.run import run_task
from evalscope.config import TaskConfig
task_cfg = TaskConfig(
    model='$model_path',
    api_url='http://localhost:${port}/v1',
    eval_type='server',
    datasets=['${bench_dataset}'],
    work_dir='$out_dir/${bench_name}',
)
results = run_task(task_cfg)
print(f'  ${bench_name} results: {results}')
" 2>&1 | tee "$out_dir/${bench_name}_eval.log" || echo "  WARNING: $bench_name eval failed"

    echo "  Stopping vLLM server (after $bench_name) ..."
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    sleep 5  # give GPU memory time to free
  }

  run_one_benchmark "bfcl" "bfcl_v3" || true
  run_one_benchmark "toolbench" "tool_bench" || true

  echo "  Done: $abbr → $out_dir/"
}

###############################################################################
# Main
###############################################################################
echo "=========================================="
echo "  EvalScope Tool-Use Evaluation"
echo "  BFCL + ToolBench"
echo "  GPUs: $NUM_GPUS (tp=$TP)"
echo "=========================================="
echo ""
echo "Models to evaluate: ${#MODELS[@]}"
echo "Output: $EVALSCOPE_OUT/"
echo ""

# Check dependencies
python3 -c "import evalscope" 2>/dev/null || {
  echo "WARNING: evalscope not installed. Run: pip install 'evalscope[all]'"
  echo ""
}
python3 -c "import vllm" 2>/dev/null || {
  echo "WARNING: vllm not installed. Run: pip install vllm"
  echo ""
}
python3 -c "import bfcl" 2>/dev/null || {
  echo "WARNING: bfcl-eval not installed. Run: pip install bfcl-eval==2025.10.27.1"
  echo "         (Required for BFCL-v3 evaluation)"
  echo ""
}

for ENTRY in "${MODELS[@]}"; do
  IFS='|' read -r ABBR MODEL_PATH <<< "$ENTRY"
  run_eval_for_model "$ABBR" "$MODEL_PATH" || true
done

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="
echo "Results: $EVALSCOPE_OUT/"
echo ""
echo "To view results:"
echo "  ls -la $EVALSCOPE_OUT/*/"
