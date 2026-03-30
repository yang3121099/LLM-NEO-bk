#!/usr/bin/env bash
###############################################################################
# batch_merge_eval_delete.sh — Auto-discover → parallel merge → eval → cleanup
#
# Features:
#   - Auto-scans results/ for Qwen3-8B and Llama-3.1-8B adapter dirs (0327-0329)
#   - Parallel merge: up to MERGE_JOBS processes simultaneously
#   - Keeps up to MAX_MERGED models on disk; cleans oldest evaluated ones
#   - Math-only eval (math-500, gsm8k, gsm8k_0shot, svamp, aime2024)
#   - All results share one run_tag for unified summary
#
# Usage:
#   nohup bash scripts/batch_merge_eval_delete.sh 2>&1 | tee batch_eval.log &
#
# Flags:
#   DRY_RUN=1        Print commands without executing
#   SKIP_MERGE=1     Skip merge (use already-merged models)
#   SKIP_DELETE=1    Keep all merged models after eval
#   MAX_MERGED=20    Max merged models on disk (default: 20)
#   MERGE_JOBS=4     Parallel merge processes (default: 4)
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
OC_DIR="$WORKSPACE_DIR/opencompass"
MERGE_SCRIPT="$WORKSPACE_DIR/src/shadow/merge_lora.py"
RUN_TAG="batch-$(date +%m%d%H%M)"

DRY_RUN="${DRY_RUN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_DELETE="${SKIP_DELETE:-0}"
MAX_MERGED="${MAX_MERGED:-20}"
MERGE_JOBS="${MERGE_JOBS:-4}"

# ---------------------------------------------------------------------------
# Auto-detect GPU count
# ---------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

echo "INFO  GPUs=$NUM_GPUS  MAX_MERGED=$MAX_MERGED  MERGE_JOBS=$MERGE_JOBS"
echo "INFO  RUN_TAG=$RUN_TAG"

###############################################################################
# Model definitions
###############################################################################
declare -A MODEL_MAP=(
  ["Llama-3.1-8B"]="meta-llama/Llama-3.1-8B|meta-llama/Llama-3.1-8B-Instruct|llama3"
  ["Qwen3-8B-Base"]="Qwen/Qwen3-8B-Base|Qwen/Qwen3-8B|qwen3"
)

# Dataset suffix → size tag mapping
declare -A DATASET_SIZES=(
  ["openr1"]="220k"
  ["deepmath"]="309k"
  ["dolci_mix"]="2k"
  ["nemotron_if"]="2k"
  ["opus3k"]="2k"
  ["deepmath_demo"]="2k"
)

###############################################################################
# Auto-discover adapter directories
###############################################################################
discover_adapters() {
  echo "INFO  Scanning $RESULTS_DIR for adapter directories ..."
  local count=0

  # Scan date dirs 0327..0330 (covering possible training dates)
  for date_dir in "$RESULTS_DIR"/03{27,28,29,30}; do
    [[ -d "$date_dir" ]] || continue
    local mmdd
    mmdd=$(basename "$date_dir")

    for result_dir in "$date_dir"/result-*-"$mmdd"; do
      [[ -d "$result_dir" ]] || continue
      local model_base
      model_base=$(basename "$result_dir" | sed "s/^result-//; s/-${mmdd}$//")

      # Look up model info
      local model_info="${MODEL_MAP[$model_base]:-}"
      if [[ -z "$model_info" ]]; then
        echo "  WARN: Unknown model '$model_base', skipping"
        continue
      fi
      IFS='|' read -r HF_BASE HF_INSTRUCT TEMPLATE <<< "$model_info"

      # Scan adapter dirs: B-*-lora-* and I-*-lora-*
      for adapter_dir in "$result_dir"/{B,I}-*-lora-rank128-*; do
        [[ -d "$adapter_dir" ]] || continue
        local dirname
        dirname=$(basename "$adapter_dir")

        # Parse: {TAG}-{K}-lora-rank128-lr0.0002-{SUFFIX}
        local src_tag k_tag suffix
        src_tag=$(echo "$dirname" | cut -d- -f1)          # B or I
        k_tag=$(echo "$dirname" | cut -d- -f2)            # 220k, 309k, 2k
        suffix=$(echo "$dirname" | sed 's/.*lr[0-9.]*-//')  # openr1, deepmath, etc.

        # Determine merge tag and target
        local merge_tag target_model
        if [[ "$src_tag" == "B" ]]; then
          merge_tag="B2I"
          target_model="$HF_INSTRUCT"
        else
          merge_tag="I2I"
          target_model="$HF_INSTRUCT"
        fi

        # Check if adapter has trained weights (adapter_model.safetensors or checkpoint-*)
        if [[ ! -f "$adapter_dir/adapter_model.safetensors" ]] && \
           [[ ! -f "$adapter_dir/adapter_model.bin" ]] && \
           ! ls "$adapter_dir"/checkpoint-*/adapter_model.safetensors &>/dev/null 2>&1 && \
           ! ls "$adapter_dir"/checkpoint-*/adapter_model.bin &>/dev/null 2>&1; then
          continue
        fi

        local abbr="${model_base}-${suffix}-${k_tag}-${merge_tag}"

        # Skip if already evaluated (check opencompass outputs)
        # Add to list
        ADAPTERS+=("${adapter_dir}|${target_model}|${merge_tag}|${TEMPLATE}|${abbr}")
        count=$((count + 1))
        echo "  FOUND: $abbr"
        echo "         $adapter_dir"
      done
    done
  done

  echo "INFO  Discovered $count adapter(s)"
}

# Initialize adapter list
ADAPTERS=()
discover_adapters

# Add HF baselines (always evaluated, never deleted)
BASELINES=(
  "NOMERGE|meta-llama/Llama-3.1-8B-Instruct|_|llama3|Llama-3.1-8B-Instruct-hf"
  "NOMERGE|Qwen/Qwen3-8B|_|qwen3|Qwen3-8B-Instruct-hf"
)

ALL_JOBS=("${BASELINES[@]}" "${ADAPTERS[@]}")
TOTAL=${#ALL_JOBS[@]}

if [[ $TOTAL -eq 0 ]]; then
  echo "ERROR: No adapters found and no baselines. Check $RESULTS_DIR."
  exit 1
fi

###############################################################################
# Math-only eval config (multi-model batch)
###############################################################################
generate_eval_config_batch() {
  local config_file="$1"
  shift
  local -a models_args=("$@")  # "abbr|path" pairs

  # Header
  cat > "$config_file" << 'PYEOF'
# Auto-generated math-only eval config
import subprocess as _sp
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate

_NUM_GPUS = max(1, len(_sp.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()))

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = [
PYEOF

  for entry in "${models_args[@]}"; do
    IFS='|' read -r abbr path <<< "$entry"
    cat >> "$config_file" << MODELEOF
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='${abbr}',
        path='${path}',
        engine_config=dict(session_len=4096, max_batch_size=4096, tp=_NUM_GPUS),
        gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=2048,
        run_cfg=dict(num_gpus=_NUM_GPUS),
    ),
MODELEOF
  done

  echo "]" >> "$config_file"
}

###############################################################################
# Parallel merge helper
###############################################################################
merge_one() {
  local adapter_path="$1" target_base="$2" merge_tag="$3" template="$4" abbr="$5"
  local merged_path="${adapter_path}/merged-${merge_tag}"

  # Skip if already merged
  if [[ -d "$merged_path" ]] && [[ -f "$merged_path/config.json" ]]; then
    echo "  CACHED: $abbr (already merged)"
    return 0
  fi

  echo "  MERGE:  $abbr ..."
  python3 "$MERGE_SCRIPT" \
    --adapter_path "$adapter_path" \
    --target_base "$target_base" \
    --merge_tag "$merge_tag" \
    --template "$template" \
    > /dev/null 2>&1 && {
      echo "  MERGED: $abbr"
    } || {
      echo "  FAIL:   $abbr (merge error)"
      return 1
    }
}

###############################################################################
# Disk cleanup: delete oldest evaluated merged models beyond MAX_MERGED
###############################################################################
# Track merged model paths in order of creation for cleanup
MERGED_ON_DISK=()   # paths of merged models currently on disk
EVALUATED=()        # paths that have been evaluated (safe to delete)

cleanup_if_needed() {
  if [[ "$SKIP_DELETE" == "1" ]]; then return; fi

  local n_merged=${#MERGED_ON_DISK[@]}
  if (( n_merged <= MAX_MERGED )); then return; fi

  local to_delete=$(( n_merged - MAX_MERGED ))
  echo "INFO  Disk cleanup: $n_merged merged models on disk (max=$MAX_MERGED), deleting $to_delete oldest evaluated ones"

  local deleted=0
  for (( i=0; i<${#EVALUATED[@]} && deleted<to_delete; i++ )); do
    local path="${EVALUATED[$i]}"
    if [[ -d "$path" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "  DRY_RUN: rm -rf $path"
      else
        rm -rf "$path"
        echo "  DELETED: $path"
      fi
      deleted=$((deleted + 1))
      # Remove from MERGED_ON_DISK
      local new_merged=()
      for mp in "${MERGED_ON_DISK[@]}"; do
        [[ "$mp" != "$path" ]] && new_merged+=("$mp")
      done
      MERGED_ON_DISK=("${new_merged[@]}")
    fi
  done
  # Remove deleted entries from EVALUATED
  EVALUATED=("${EVALUATED[@]:$deleted}")
}

###############################################################################
# Main pipeline
###############################################################################
echo ""
echo "=========================================="
echo "  Batch Merge + Eval + Delete Pipeline"
echo "  Total jobs:    $TOTAL (${#BASELINES[@]} baselines + ${#ADAPTERS[@]} adapters)"
echo "  Run tag:       $RUN_TAG"
echo "  Merge jobs:    $MERGE_JOBS parallel"
echo "  Max on disk:   $MAX_MERGED merged models"
echo "  Eval tasks:    math-500, gsm8k, gsm8k_0shot, svamp, aime2024"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Parallel merge (all adapters, MERGE_JOBS at a time)
# ---------------------------------------------------------------------------
if [[ "$SKIP_MERGE" != "1" ]]; then
  echo "=== Phase 1: Parallel Merge (${#ADAPTERS[@]} adapters, $MERGE_JOBS parallel) ==="

  MERGE_PIDS=()
  MERGE_ABBRS=()
  MERGE_IDX=0

  for entry in "${ADAPTERS[@]}"; do
    IFS='|' read -r ADAPTER_PATH TARGET_BASE MERGE_TAG TEMPLATE ABBR <<< "$entry"

    if [[ ! -d "$ADAPTER_PATH" ]]; then
      echo "  SKIP: $ABBR — adapter path not found"
      continue
    fi

    # Wait if we've hit max parallel jobs
    while (( ${#MERGE_PIDS[@]} >= MERGE_JOBS )); do
      # Wait for any one to finish
      new_pids=()
      new_abbrs=()
      for (( pi=0; pi<${#MERGE_PIDS[@]}; pi++ )); do
        if kill -0 "${MERGE_PIDS[$pi]}" 2>/dev/null; then
          new_pids+=("${MERGE_PIDS[$pi]}")
          new_abbrs+=("${MERGE_ABBRS[$pi]}")
        fi
      done
      MERGE_PIDS=("${new_pids[@]}")
      MERGE_ABBRS=("${new_abbrs[@]}")
      if (( ${#MERGE_PIDS[@]} >= MERGE_JOBS )); then
        sleep 5
      fi
    done

    # Also check disk limit before launching new merge
    # Count currently existing merged dirs
    current_merged=0
    for e in "${ADAPTERS[@]}"; do
      IFS='|' read -r _ap _ _mt _ _ <<< "$e"
      [[ -d "${_ap}/merged-${_mt}" ]] && current_merged=$((current_merged + 1))
    done
    while (( current_merged >= MAX_MERGED )); do
      echo "  WAIT: $current_merged merged models on disk (max=$MAX_MERGED), waiting for eval to free space ..."
      sleep 30
      current_merged=0
      for e in "${ADAPTERS[@]}"; do
        IFS='|' read -r _ap _ _mt _ _ <<< "$e"
        [[ -d "${_ap}/merged-${_mt}" ]] && current_merged=$((current_merged + 1))
      done
    done

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: merge_one $ADAPTER_PATH $TARGET_BASE $MERGE_TAG $TEMPLATE $ABBR &"
    else
      merge_one "$ADAPTER_PATH" "$TARGET_BASE" "$MERGE_TAG" "$TEMPLATE" "$ABBR" &
      MERGE_PIDS+=($!)
      MERGE_ABBRS+=("$ABBR")
    fi

    MERGE_IDX=$((MERGE_IDX + 1))
  done

  # Wait for remaining merge jobs
  if (( ${#MERGE_PIDS[@]} > 0 )); then
    echo "INFO  Waiting for ${#MERGE_PIDS[@]} remaining merge job(s) ..."
    for pid in "${MERGE_PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi

  echo "=== Phase 1 Complete: All merges done ==="
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 2: Sequential eval (GPU-bound, one model at a time via batch config)
# Then delete evaluated models in batches to stay under MAX_MERGED
# ---------------------------------------------------------------------------
echo "=== Phase 2: Evaluation + Cleanup ==="

SUCCEEDED=0
FAILED=0
SKIPPED=0

# Build list of (abbr, model_path, need_delete, merged_path) for eval
EVAL_QUEUE=()

# Baselines first
for entry in "${BASELINES[@]}"; do
  IFS='|' read -r _AP TARGET_BASE _MT _TPL ABBR <<< "$entry"
  EVAL_QUEUE+=("${ABBR}|${TARGET_BASE}|no|_")
done

# Then trained models
for entry in "${ADAPTERS[@]}"; do
  IFS='|' read -r ADAPTER_PATH TARGET_BASE MERGE_TAG TEMPLATE ABBR <<< "$entry"
  MERGED_PATH="${ADAPTER_PATH}/merged-${MERGE_TAG}"
  if [[ -d "$MERGED_PATH" ]] && [[ -f "$MERGED_PATH/config.json" ]]; then
    EVAL_QUEUE+=("${ABBR}|${MERGED_PATH}|yes|${MERGED_PATH}")
    MERGED_ON_DISK+=("$MERGED_PATH")
  else
    echo "  SKIP eval: $ABBR — merged model not found at $MERGED_PATH"
    SKIPPED=$((SKIPPED + 1))
  fi
done

EVAL_TOTAL=${#EVAL_QUEUE[@]}
echo "INFO  Models to evaluate: $EVAL_TOTAL"

for (( idx=0; idx<EVAL_TOTAL; idx++ )); do
  IFS='|' read -r ABBR MODEL_PATH NEED_DELETE MERGED_PATH <<< "${EVAL_QUEUE[$idx]}"

  echo ""
  echo "----------------------------------------------------------------"
  echo "  [$((idx+1))/$EVAL_TOTAL] EVAL: $ABBR"
  echo "  Path: $MODEL_PATH"
  echo "----------------------------------------------------------------"

  # Generate single-model eval config
  EVAL_CONFIG="$OC_DIR/_eval_batch_${ABBR}.py"
  generate_eval_config_batch "$EVAL_CONFIG" "${ABBR}|${MODEL_PATH}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: cd $OC_DIR && python3 ./run.py $EVAL_CONFIG -r $RUN_TAG"
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    if (cd "$OC_DIR" && python3 ./run.py "$EVAL_CONFIG" -r "$RUN_TAG"); then
      echo "  EVAL OK: $ABBR"
      SUCCEEDED=$((SUCCEEDED + 1))
    else
      echo "  EVAL FAIL: $ABBR"
      FAILED=$((FAILED + 1))
    fi
  fi

  rm -f "$EVAL_CONFIG"

  # Track evaluated models for cleanup
  if [[ "$NEED_DELETE" == "yes" ]]; then
    EVALUATED+=("$MERGED_PATH")
    cleanup_if_needed
  fi
done

# Final cleanup: delete all remaining evaluated merged models
if [[ "$SKIP_DELETE" != "1" ]] && (( ${#EVALUATED[@]} > 0 )); then
  echo ""
  echo "=== Final Cleanup: Deleting ${#EVALUATED[@]} evaluated merged model(s) ==="
  for path in "${EVALUATED[@]}"; do
    if [[ -d "$path" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "  DRY_RUN: rm -rf $path"
      else
        rm -rf "$path"
        echo "  DELETED: $path"
      fi
    fi
  done
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo ""
echo "  All results under: $OC_DIR/outputs/"
echo "  Run tag: $RUN_TAG"
echo ""
echo "  To view aggregated summary:"
echo "    cd $OC_DIR"
echo "    python3 -m opencompass.cli.main summarize -w outputs/ -r $RUN_TAG"
echo ""
