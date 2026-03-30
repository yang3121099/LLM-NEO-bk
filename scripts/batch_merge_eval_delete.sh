#!/usr/bin/env bash
###############################################################################
# batch_merge_eval_delete.sh — Sequential merge → eval → delete pipeline
#
# Designed for space-constrained environments: processes one merged model at
# a time, runs math-only evaluation, then deletes the merged model to free
# disk space. Results persist in OpenCompass outputs.
#
# Usage:
#   nohup bash scripts/batch_merge_eval_delete.sh 2>&1 | tee batch_eval.log &
#
# Customize:
#   - Edit ADAPTERS array below to add/remove merge jobs
#   - Set WORKSPACE_DIR if running from a different location
#   - Set SKIP_MERGE=1 to skip merge (eval already-merged models)
#   - Set SKIP_DELETE=1 to keep merged models after eval
#   - Set DRY_RUN=1 to print commands without executing
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
OC_DIR="$WORKSPACE_DIR/opencompass"
MERGE_SCRIPT="$WORKSPACE_DIR/src/shadow/merge_lora.py"
RUN_TAG="batch-$(date +%m%d%H%M)"

SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_DELETE="${SKIP_DELETE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Auto-detect GPU count
# ---------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
echo "INFO  GPUs: $NUM_GPUS"

###############################################################################
# ADAPTER LIST — edit this section to match your trained checkpoints
#
# Format: "adapter_path | target_base | merge_tag | template | abbr"
#
#   adapter_path : path to LoRA adapter dir (checkpoint-XXXX or final)
#   target_base  : HF model to merge into (e.g. Instruct for B2I)
#   merge_tag    : B2I / I2I
#   template     : llama3 / qwen3
#   abbr         : short name for eval results
#
# The merged model is saved to: {adapter_path}/merged-{merge_tag}/
###############################################################################

ADAPTERS=(
  # === OpenR1-Math-220k (Llama-3.1-8B) ===
  # B→I (Base adapter merged onto Instruct)
  "$RESULTS_DIR/0329/result-Llama-3.1-8B-0329/B-220k-lora-rank128-lr0.0002-openr1|meta-llama/Llama-3.1-8B-Instruct|B2I|llama3|Llama-3.1-8B-openr1-220k-B2I"
  # I→I (Instruct adapter merged onto Instruct)
  "$RESULTS_DIR/0329/result-Llama-3.1-8B-0329/I-220k-lora-rank128-lr0.0002-openr1|meta-llama/Llama-3.1-8B-Instruct|I2I|llama3|Llama-3.1-8B-openr1-220k-I2I"

  # === DeepMath-103K (Llama-3.1-8B) ===
  "$RESULTS_DIR/0329/result-Llama-3.1-8B-0329/B-309k-lora-rank128-lr0.0002-deepmath|meta-llama/Llama-3.1-8B-Instruct|B2I|llama3|Llama-3.1-8B-deepmath-309k-B2I"
  "$RESULTS_DIR/0329/result-Llama-3.1-8B-0329/I-309k-lora-rank128-lr0.0002-deepmath|meta-llama/Llama-3.1-8B-Instruct|I2I|llama3|Llama-3.1-8B-deepmath-309k-I2I"

  # === Baselines (Llama-3.1-8B-Instruct, no merge needed) ===
  # Use special "NOMERGE" target_base to skip merge step
  "NOMERGE|meta-llama/Llama-3.1-8B-Instruct|_|llama3|Llama-3.1-8B-Instruct-hf"
)

###############################################################################
# Math-only eval config (generated inline for each model)
###############################################################################
generate_eval_config() {
  local abbr="$1" model_path="$2" config_file="$3"

  cat > "$config_file" << PYEOF
# Auto-generated math-only eval config for: $abbr
import os as _os
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
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='${abbr}',
        path='${model_path}',
        engine_config=dict(session_len=4096, max_batch_size=4096, tp=_NUM_GPUS),
        gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=2048,
        run_cfg=dict(num_gpus=_NUM_GPUS),
    )
]
PYEOF
}

###############################################################################
# Main loop: merge → eval → delete, one model at a time
###############################################################################
TOTAL=${#ADAPTERS[@]}
echo ""
echo "=========================================="
echo "  Batch Merge + Eval + Delete Pipeline"
echo "  Models: $TOTAL"
echo "  Run tag: $RUN_TAG"
echo "  Math-only: math-500, gsm8k, gsm8k_0shot, svamp, aime2024"
echo "=========================================="
echo ""

SUCCEEDED=0
FAILED=0
SKIPPED=0

for (( idx=0; idx<TOTAL; idx++ )); do
  IFS='|' read -r ADAPTER_PATH TARGET_BASE MERGE_TAG TEMPLATE ABBR <<< "${ADAPTERS[$idx]}"

  echo ""
  echo "================================================================"
  echo "  [$((idx+1))/$TOTAL] $ABBR"
  echo "================================================================"

  # --- Determine merged model path ---
  if [[ "$ADAPTER_PATH" == "NOMERGE" ]]; then
    # HF baseline — no merge, no delete
    MERGED_PATH="$TARGET_BASE"
    NEED_DELETE=false
    echo "  Type: HF baseline (no merge)"
  else
    MERGED_PATH="${ADAPTER_PATH}/merged-${MERGE_TAG}"
    NEED_DELETE=true
    echo "  Adapter:  $ADAPTER_PATH"
    echo "  Target:   $TARGET_BASE"
    echo "  Merge:    $MERGE_TAG"
    echo "  Merged →  $MERGED_PATH"
  fi

  # --- Step 1: Merge ---
  if [[ "$ADAPTER_PATH" != "NOMERGE" ]] && [[ "$SKIP_MERGE" != "1" ]]; then
    if [[ ! -d "$ADAPTER_PATH" ]]; then
      echo "  SKIP: adapter path not found: $ADAPTER_PATH"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo "  [1/3] Merging ..."
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: python3 $MERGE_SCRIPT --adapter_path $ADAPTER_PATH --target_base $TARGET_BASE --merge_tag $MERGE_TAG --template $TEMPLATE"
    else
      python3 "$MERGE_SCRIPT" \
        --adapter_path "$ADAPTER_PATH" \
        --target_base "$TARGET_BASE" \
        --merge_tag "$MERGE_TAG" \
        --template "$TEMPLATE" || {
          echo "  FAILED: merge for $ABBR"
          FAILED=$((FAILED + 1))
          continue
        }
    fi
    echo "  Merge done."
  else
    echo "  [1/3] Merge: skipped"
  fi

  # --- Step 2: Eval ---
  echo "  [2/3] Evaluating (math-only) ..."
  EVAL_CONFIG="$OC_DIR/_eval_batch_${ABBR}.py"
  generate_eval_config "$ABBR" "$MERGED_PATH" "$EVAL_CONFIG"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: cd $OC_DIR && python3 ./run.py $EVAL_CONFIG -r $RUN_TAG"
  else
    (cd "$OC_DIR" && python3 ./run.py "$EVAL_CONFIG" -r "$RUN_TAG") || {
      echo "  FAILED: eval for $ABBR"
      FAILED=$((FAILED + 1))
      # Still try to delete to free space
      if [[ "$NEED_DELETE" == true ]] && [[ "$SKIP_DELETE" != "1" ]] && [[ -d "$MERGED_PATH" ]]; then
        echo "  [3/3] Deleting merged model to free space (despite eval failure) ..."
        rm -rf "$MERGED_PATH"
      fi
      rm -f "$EVAL_CONFIG"
      continue
    }
  fi
  echo "  Eval done."

  # --- Step 3: Delete merged model ---
  if [[ "$NEED_DELETE" == true ]] && [[ "$SKIP_DELETE" != "1" ]]; then
    echo "  [3/3] Deleting merged model to free space ..."
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: rm -rf $MERGED_PATH"
    else
      rm -rf "$MERGED_PATH"
      echo "  Deleted: $MERGED_PATH"
    fi
  else
    echo "  [3/3] Delete: skipped"
  fi

  # Clean up temp eval config
  rm -f "$EVAL_CONFIG"
  SUCCEEDED=$((SUCCEEDED + 1))
  echo "  OK: $ABBR"
done

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
echo "  Results: $OC_DIR/outputs/*/$RUN_TAG/"
echo ""
echo "  To view summary:"
echo "    cd $OC_DIR && python3 ./run.py outputs/*/$RUN_TAG/ -m summary"
echo ""
