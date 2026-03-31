#!/usr/bin/env bash
###############################################################################
# batch_merge_eval_delete.sh — Auto-discover → parallel merge → batch eval
#
# For each Qwen3-8B experiment, merges 10 evenly-spaced checkpoints (every 10%),
# then evaluates all merged models + Qwen baseline on math benchmarks.
#
# Usage:
#   DRY_RUN=1 bash scripts/batch_merge_eval_delete.sh   # preview
#   nohup bash scripts/batch_merge_eval_delete.sh 2>&1 | tee batch_eval.log &
#
# Flags:
#   DRY_RUN=1        Print commands without executing
#   SKIP_MERGE=1     Skip merge phase (eval already-merged models)
#   MERGE_JOBS=4     Parallel merge processes (default: 4)
#   NUM_CKPTS=10     Number of evenly-spaced checkpoints to merge (default: 10)
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
OC_DIR="$WORKSPACE_DIR/opencompass"
MERGE_SCRIPT="$WORKSPACE_DIR/src/shadow/merge_lora.py"
RUN_TAG="${RUN_TAG:-batch-$(date +%m%d%H%M)}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
MERGE_JOBS="${MERGE_JOBS:-4}"
NUM_CKPTS="${NUM_CKPTS:-10}"

# ---------------------------------------------------------------------------
# GPU auto-detect
# ---------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

echo "INFO  GPUs=$NUM_GPUS  MERGE_JOBS=$MERGE_JOBS  NUM_CKPTS=$NUM_CKPTS"
echo "INFO  RUN_TAG=$RUN_TAG"

###############################################################################
# Qwen3-8B only (no Llama)
###############################################################################
HF_INSTRUCT="Qwen/Qwen3-8B"
TEMPLATE="qwen3"

###############################################################################
# Phase 0: Auto-discover adapter directories + select checkpoints
###############################################################################
MERGE_QUEUE=()

echo ""
echo "=== Phase 0: Auto-discover adapters ==="
echo "INFO  Scanning $RESULTS_DIR ..."

# Broadly scan all result-*Qwen3-8B* directories under any date folder
for result_dir in "$RESULTS_DIR"/*/result-Qwen3-8B*; do
  [[ -d "$result_dir" ]] || continue
  echo "INFO  Found: $result_dir"

  for adapter_dir in "$result_dir"/B-*-lora-* "$result_dir"/I-*-lora-*; do
    [[ -d "$adapter_dir" ]] || continue

    dirname=$(basename "$adapter_dir")
    src_tag=${dirname%%-*}                                  # B or I
    k_tag=$(echo "$dirname" | cut -d- -f2)                  # 220k, 309k, 2k
    suffix=$(echo "$dirname" | sed 's/.*lr[0-9.]*-//')      # openr1, deepmath

    # Determine merge tag
    if [[ "$src_tag" == "B" ]]; then
      merge_tag="B2I"
    else
      merge_tag="I2I"
    fi

    # Collect all checkpoints, sorted by step number
    all_ckpts=()
    while IFS= read -r ckpt; do
      all_ckpts+=("$ckpt")
    done < <(find "$adapter_dir" -maxdepth 1 -type d -name "checkpoint-*" | \
             awk -F'checkpoint-' '{print $NF, $0}' | sort -n | cut -d' ' -f2-)

    total=${#all_ckpts[@]}

    if [[ $total -eq 0 ]]; then
      echo "  SKIP: $dirname — no checkpoints"
      continue
    fi

    echo "  $dirname: $total checkpoints"

    # Select NUM_CKPTS evenly-spaced (always include first and last)
    if (( total <= NUM_CKPTS )); then
      selected=("${all_ckpts[@]}")
    else
      selected=()
      for (( i=0; i<NUM_CKPTS; i++ )); do
        idx=$(( i * (total - 1) / (NUM_CKPTS - 1) ))
        selected+=("${all_ckpts[$idx]}")
      done
      # Deduplicate
      declare -A _seen=()
      deduped=()
      for s in "${selected[@]}"; do
        [[ -z "${_seen[$s]:-}" ]] && deduped+=("$s") && _seen[$s]=1
      done
      selected=("${deduped[@]}")
      unset _seen
    fi

    for ckpt in "${selected[@]}"; do
      step=$(basename "$ckpt" | sed 's/checkpoint-//')
      model_base=$(basename "$result_dir" | sed 's/^result-//; s/-[0-9]*$//')
      abbr="${model_base}-${suffix}-${k_tag}-step${step}-${merge_tag}"
      MERGE_QUEUE+=("${ckpt}|${HF_INSTRUCT}|${merge_tag}|${TEMPLATE}|${abbr}")
      echo "    → checkpoint-${step} (${merge_tag})"
    done
  done
done

echo ""
echo "INFO  Total merge jobs: ${#MERGE_QUEUE[@]}"

if [[ ${#MERGE_QUEUE[@]} -eq 0 ]]; then
  echo "WARN  No adapters found. Contents of $RESULTS_DIR:"
  ls -R "$RESULTS_DIR" 2>/dev/null | head -30 || echo "  (empty)"
  echo ""
  echo "Will only evaluate Qwen3-8B baseline."
fi

# Print all discovered jobs
echo ""
echo "=== Merge Queue ==="
for (( i=0; i<${#MERGE_QUEUE[@]}; i++ )); do
  IFS='|' read -r CKPT_PATH _T _M _TPL ABBR <<< "${MERGE_QUEUE[$i]}"
  echo "  [$((i+1))] $ABBR"
  echo "       $CKPT_PATH"
done
echo ""

###############################################################################
# Phase 1: Parallel merge
###############################################################################
merge_one() {
  local adapter_path="$1" target_base="$2" merge_tag="$3" template="$4" abbr="$5"
  local merged_path="${adapter_path}/merged-${merge_tag}"

  if [[ -d "$merged_path" ]] && [[ -f "$merged_path/config.json" ]]; then
    echo "  CACHED: $abbr"
    return 0
  fi

  echo "  MERGE:  $abbr ..."
  if python3 "$MERGE_SCRIPT" \
    --adapter_path "$adapter_path" \
    --target_base "$target_base" \
    --merge_tag "$merge_tag" \
    --template "$template" > /dev/null 2>&1; then
    echo "  OK:     $abbr"
  else
    echo "  FAIL:   $abbr"
    return 1
  fi
}

if [[ "$SKIP_MERGE" != "1" ]] && [[ ${#MERGE_QUEUE[@]} -gt 0 ]]; then
  echo "=== Phase 1: Parallel Merge (${#MERGE_QUEUE[@]} jobs, $MERGE_JOBS parallel) ==="

  PIDS=()
  for entry in "${MERGE_QUEUE[@]}"; do
    IFS='|' read -r CKPT_PATH TARGET MTAG TPL ABBR <<< "$entry"

    # Throttle
    while (( ${#PIDS[@]} >= MERGE_JOBS )); do
      new_pids=()
      for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
      done
      PIDS=("${new_pids[@]}")
      (( ${#PIDS[@]} >= MERGE_JOBS )) && sleep 3
    done

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: merge $ABBR"
    else
      merge_one "$CKPT_PATH" "$TARGET" "$MTAG" "$TPL" "$ABBR" &
      PIDS+=($!)
    fi
  done

  # Wait for all remaining
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  echo "=== Phase 1 Complete ==="
fi

###############################################################################
# Phase 2: Generate eval config with ALL models, run once
###############################################################################
echo ""
echo "=== Phase 2: Evaluation ==="

# Collect all models for eval
EVAL_MODELS=()

# Qwen baseline
EVAL_MODELS+=("Qwen3-8B-Instruct-hf|${HF_INSTRUCT}")

# Merged models
for entry in "${MERGE_QUEUE[@]}"; do
  IFS='|' read -r CKPT_PATH _TARGET MTAG _TPL ABBR <<< "$entry"
  MERGED_PATH="${CKPT_PATH}/merged-${MTAG}"
  if [[ "$DRY_RUN" == "1" ]]; then
    EVAL_MODELS+=("${ABBR}|${MERGED_PATH}")
  elif [[ -d "$MERGED_PATH" ]] && [[ -f "$MERGED_PATH/config.json" ]]; then
    EVAL_MODELS+=("${ABBR}|${MERGED_PATH}")
  else
    echo "  SKIP eval: $ABBR — merged model not found"
  fi
done

EVAL_TOTAL=${#EVAL_MODELS[@]}
echo "INFO  Models to evaluate: $EVAL_TOTAL"

# Generate single eval config with all models
EVAL_CONFIG="$OC_DIR/_eval_batch_all.py"

cat > "$EVAL_CONFIG" << 'PYEOF'
# Auto-generated math-only batch eval config
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

for entry in "${EVAL_MODELS[@]}"; do
  IFS='|' read -r ABBR MODEL_PATH <<< "$entry"
  cat >> "$EVAL_CONFIG" << MODELEOF
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='${ABBR}',
        path='${MODEL_PATH}',
        engine_config=dict(session_len=4096, max_batch_size=4096, tp=_NUM_GPUS),
        gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=2048,
        run_cfg=dict(num_gpus=_NUM_GPUS),
    ),
MODELEOF
done

echo "]" >> "$EVAL_CONFIG"

echo ""
echo "=== Eval Models ==="
for entry in "${EVAL_MODELS[@]}"; do
  IFS='|' read -r ABBR MPATH <<< "$entry"
  echo "  $ABBR"
  echo "    $MPATH"
done
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: cd $OC_DIR && python3 ./run.py _eval_batch_all.py -r $RUN_TAG"
  echo ""
  echo "DRY_RUN complete. Review above, then run without DRY_RUN=1."
else
  echo "Starting evaluation ..."
  (cd "$OC_DIR" && python3 ./run.py "_eval_batch_all.py" -r "$RUN_TAG") || {
    echo "ERROR: Evaluation failed"
    echo "Config preserved at: $EVAL_CONFIG"
    exit 1
  }
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "  Models evaluated: $EVAL_TOTAL"
echo "  Run tag:          $RUN_TAG"
echo "  Config:           $EVAL_CONFIG"
echo "  Results:          $OC_DIR/outputs/"
echo ""
