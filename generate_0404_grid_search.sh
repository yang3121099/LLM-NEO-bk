#!/usr/bin/env bash
###############################################################################
# generate_0404_grid_search.sh — 32B grid search + 70B experiments
#
# Phase 1: Qwen2.5-32B grid search (Shadow_2k, multiple rank/lr combos)
# Phase 2: Meta-Llama-3-70B × 3 configs (Shadow_2k)
#
# All results go under results/0404/. Eval = Math-7 only.
#
# Execution order:
#   1. 32B grid search (5 new configs) → eval
#   2. 70B × 3 configs → eval
#
# Usage:
#   bash generate_0404_grid_search.sh
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_DIR"

# --- Fixed training constants ---
EFFECTIVE_BS=256
DEFAULT_PER_GPU_BS=1
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
CUTOFF_LEN=4096
VAL_SIZE=0.01

TIMESTAMP=$(date +%m%d%H%M%S)
MONTHDAY=0404

# --- Dataset ---
DATASETS=(
  "Shadow_2k|shadow2k|2000|200"
)

# --- Helpers ---
to_decimal() { LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//; s/\.$/.0/'; }
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then echo "$((num / 1000))k"
  else awk -v n="$num" 'BEGIN{ printf "%.1fk", n/1000 }'; fi
}

###############################################################################
# Generate training script
###############################################################################
generate_train_script() {
  local MODEL_SHORT="$1" HF_BASE="$2" HF_INSTRUCT="$3" TEMPLATE="$4"
  local RANK="$5" LR="$6" CFG_NAME="$7"
  local DATASET="$8" SUFFIX="$9" MAX_SAMPLES="${10}" SAVE_STEPS="${11}" PER_GPU_BS="${12:-$DEFAULT_PER_GPU_BS}"

  local LR_DEC LR_TAG K REL_ROOT SCRIPT_FILE
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"
  K=$(format_k "$MAX_SAMPLES")
  REL_ROOT="${MONTHDAY}/result-${MODEL_SHORT}-${MONTHDAY}"
  SCRIPT_FILE="$SCRIPT_DIR/train_${MODEL_SHORT}_${SUFFIX}_${K}_${CFG_NAME}_${TIMESTAMP}.sh"

  cat > "$SCRIPT_FILE" << HEADER
#!/usr/bin/env bash
###############################################################################
# ${MODEL_SHORT} Shadow-FT: ${CFG_NAME} (rank=${RANK}, lr=${LR})
# Dataset: ${DATASET} (${K} samples)
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$WORKSPACE_DIR"
RESULTS_DIR="$RESULTS_DIR"

NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "\$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
echo "INFO: Detected \$NUM_GPUS GPU(s)"

PER_GPU_BS=${PER_GPU_BS}
GRAD_ACCUM=\$(( ${EFFECTIVE_BS} / (PER_GPU_BS * NUM_GPUS) ))
if [[ "\$GRAD_ACCUM" -lt 1 ]]; then
    PER_GPU_BS=1
    GRAD_ACCUM=\$(( ${EFFECTIVE_BS} / NUM_GPUS ))
fi
echo "INFO: PER_GPU_BS=\$PER_GPU_BS  GRAD_ACCUM=\$GRAD_ACCUM  effective=${EFFECTIVE_BS}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True

HEADER

  for TAG_INFO in "B|$HF_BASE" "I|$HF_INSTRUCT"; do
    IFS='|' read -r TAG M_PATH <<< "$TAG_INFO"
    DIR="${TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
    REL_OUTDIR="${REL_ROOT}/$DIR"

    cat >> "$SCRIPT_FILE" << TRAIN
###############################################################################
##### Train ${TAG} (${M_PATH}) #####
###############################################################################
mkdir -p "\$RESULTS_DIR/${REL_OUTDIR}"
cd "\$WORKSPACE_DIR"

DS_ARG=""
if [[ "\$NUM_GPUS" -gt 1 ]]; then
    export FORCE_TORCHRUN=1
    export NNODES=1
    export NPROC_PER_NODE=\$NUM_GPUS
    export MASTER_PORT=\$(( RANDOM % 10000 + 20000 ))
    export NCCL_TIMEOUT=7200
    export TORCH_NCCL_TIMEOUT=7200000
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
    DS_ARG="--deepspeed \$WORKSPACE_DIR/examples/deepspeed/ds_z3_config.json"
fi

llamafactory-cli train \\
  --model_name_or_path "${M_PATH}" \\
  --stage sft \\
  --do_train true \\
  --finetuning_type lora --lora_rank ${RANK} \\
  --dataset "${DATASET}" \\
  --template "${TEMPLATE}" \\
  --cutoff_len ${CUTOFF_LEN} \\
  --max_samples ${MAX_SAMPLES} \\
  --output_dir "\$RESULTS_DIR/${REL_OUTDIR}" \\
  --per_device_train_batch_size \$PER_GPU_BS \\
  --gradient_accumulation_steps \$GRAD_ACCUM \\
  --learning_rate ${LR_DEC} \\
  --num_train_epochs ${NUM_EPOCHS} \\
  --logging_steps ${LOGGING_STEPS} \\
  --save_steps ${SAVE_STEPS} \\
  --save_only_model True \\
  --plot_loss true \\
  --lr_scheduler_type ${LR_SCHEDULER} \\
  --warmup_ratio ${WARMUP_RATIO} \\
  --bf16 ${BF16} \\
  --val_size ${VAL_SIZE} \\
  --per_device_eval_batch_size 1 \\
  --eval_strategy steps \\
  --eval_steps 10000 \\
  --trust_remote_code True \\
  --flash_attn fa2 \\
  --overwrite_output_dir true \\
  --overwrite_cache false \\
  --use_fast_tokenizer True \\
  --preprocessing_num_workers 16 \\
  \$DS_ARG

TRAIN
  done

  # Merge B2I and I2I
  for MERGE_INFO in "B|B2I|$HF_INSTRUCT" "I|I2I|$HF_INSTRUCT"; do
    IFS='|' read -r SRC_TAG MERGE_TAG TGT_MODEL <<< "$MERGE_INFO"
    SRC_DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"

    cat >> "$SCRIPT_FILE" << MERGE
### Merge: ${MERGE_TAG} (${SRC_DIR}) ###
python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
    --adapter_path "\$RESULTS_DIR/${REL_ROOT}/${SRC_DIR}" \\
    --target_base "${TGT_MODEL}" \\
    --merge_tag "${MERGE_TAG}" \\
    --template "${TEMPLATE}"

MERGE
  done

  chmod +x "$SCRIPT_FILE"
  echo "$SCRIPT_FILE"
}

###############################################################################
# Generate eval config (Math-7 only)
###############################################################################
generate_eval_config() {
  local CONFIG_FILE="$1"
  shift
  local -a EVAL_ENTRIES=("$@")

  cat > "$CONFIG_FILE" << 'PYHEADER'
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
import os as _os

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results')

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 only #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

PYHEADER

  {
    echo "Baseline_settings = ["
    echo "('Qwen2.5-32B-Instruct-hf', 'Qwen/Qwen2.5-32B-Instruct'),"
    echo "('Llama-3-70B-Instruct-hf', 'meta-llama/Meta-Llama-3-70B-Instruct'),"
    echo ""
    for entry in "${EVAL_ENTRIES[@]}"; do
      echo "$entry"
    done
    echo "]"
  } >> "$CONFIG_FILE"

  cat >> "$CONFIG_FILE" << 'PYFOOTER'

models = []

for abbr, path in Baseline_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=8),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=8),
        )
    )
PYFOOTER
}

###############################################################################
# Main
###############################################################################
echo ""
echo "=========================================="
echo "  0404 Grid Search + 70B Experiments"
echo "  Timestamp: $TIMESTAMP"
echo "=========================================="

ALL_EVAL_ENTRIES=()

# ==========================================================================
# Phase 1: Qwen2.5-32B grid search (Shadow_2k)
# Already done in 0402: cfgA(r256,lr1e-4), cfgB(r128,lr1e-4), cfgC(r256,lr5e-5)
# New configs to fill the grid:
# ==========================================================================
echo ""
echo "=== Phase 1: Qwen2.5-32B Full Grid Search ==="

GRID_32B=(
  # rank=64 (all new)
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|64|2e-4|r64_lr2e4|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|64|1e-4|r64_lr1e4|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|64|5e-5|r64_lr5e5|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|64|2e-5|r64_lr2e5|1"
  # lr=2e-4 (rank 128/256/512)
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|128|2e-4|r128_lr2e4|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|256|2e-4|r256_lr2e4|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|512|2e-4|r512_lr2e4|1"
  # rank=512 lr=1e-4
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|512|1e-4|r512_lr1e4|1"
)

for EXEC in "${GRID_32B[@]}"; do
  IFS='|' read -r M_SHORT HF_BASE HF_INST TPL RANK LR CFG PER_GPU_BS <<< "$EXEC"
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"
  echo "  ${M_SHORT} ${CFG} (rank=${RANK}, lr=${LR})"

  for DS in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET SUFFIX MAX_SAMPLES SAVE_STEPS <<< "$DS"
    K=$(format_k "$MAX_SAMPLES")
    SCRIPT=$(generate_train_script "$M_SHORT" "$HF_BASE" "$HF_INST" "$TPL" \
             "$RANK" "$LR" "$CFG" "$DATASET" "$SUFFIX" "$MAX_SAMPLES" "$SAVE_STEPS" "$PER_GPU_BS")
    echo "    $SCRIPT"

    REL_ROOT="${MONTHDAY}/result-${M_SHORT}-${MONTHDAY}"
    for MERGE_TAG in "B2I" "I2I"; do
      [[ "$MERGE_TAG" == "B2I" ]] && SRC_TAG="B" || SRC_TAG="I"
      DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
      REL="${REL_ROOT}/${DIR}/merged-${MERGE_TAG}"
      ABBR="${M_SHORT}-${SUFFIX}-${K}-${CFG}-${MERGE_TAG}"
      ALL_EVAL_ENTRIES+=("    ('${ABBR}','\$RESULTS_DIR/${REL}'),")
    done
  done
done

# Include previously done experiments in eval
echo ""
echo "  (also includes previously done configs in eval)"

# 0402 results: cfgA, cfgB, cfgC
for OLD_CFG_INFO in "256|1e-4|cfgA" "128|1e-4|cfgB" "256|5e-5|cfgC"; do
  IFS='|' read -r O_RANK O_LR O_CFG <<< "$OLD_CFG_INFO"
  O_LR_DEC=$(to_decimal "$O_LR")
  O_LR_TAG="lr${O_LR_DEC}"
  OLD_ROOT="0402/result-Qwen2.5-32B-0402"
  for MT in "B2I" "I2I"; do
    [[ "$MT" == "B2I" ]] && ST="B" || ST="I"
    OLD_DIR="${ST}-2k-lora-rank${O_RANK}-${O_LR_TAG}-shadow2k"
    OLD_REL="${OLD_ROOT}/${OLD_DIR}/merged-${MT}"
    OLD_ABBR="Qwen2.5-32B-shadow2k-2k-${O_CFG}-${MT}"
    ALL_EVAL_ENTRIES+=("    ('${OLD_ABBR}','\$RESULTS_DIR/${OLD_REL}'),")
  done
done

# 0404 already done results
for OLD_CFG_INFO in \
  "128|5e-5|r128_lr5e5" \
  "128|2e-5|r128_lr2e5" \
  "256|2e-5|r256_lr2e5" \
  "512|5e-5|r512_lr5e5" \
  "512|2e-5|r512_lr2e5"; do
  IFS='|' read -r O_RANK O_LR O_CFG <<< "$OLD_CFG_INFO"
  O_LR_DEC=$(to_decimal "$O_LR")
  O_LR_TAG="lr${O_LR_DEC}"
  OLD_ROOT="0404/result-Qwen2.5-32B-0404"
  for MT in "B2I" "I2I"; do
    [[ "$MT" == "B2I" ]] && ST="B" || ST="I"
    OLD_DIR="${ST}-2k-lora-rank${O_RANK}-${O_LR_TAG}-shadow2k"
    OLD_REL="${OLD_ROOT}/${OLD_DIR}/merged-${MT}"
    OLD_ABBR="Qwen2.5-32B-shadow2k-2k-${O_CFG}-${MT}"
    ALL_EVAL_ENTRIES+=("    ('${OLD_ABBR}','\$RESULTS_DIR/${OLD_REL}'),")
  done
done

# ==========================================================================
# Phase 2: Meta-Llama-3-70B × 3 configs (Shadow_2k, BS=1)
# ==========================================================================
echo ""
echo "=== Phase 2: Meta-Llama-3-70B ==="

GRID_70B=(
  "Llama-3-70B|meta-llama/Meta-Llama-3-70B|meta-llama/Meta-Llama-3-70B-Instruct|llama3|256|5e-5|r256_lr5e5|1"
  "Llama-3-70B|meta-llama/Meta-Llama-3-70B|meta-llama/Meta-Llama-3-70B-Instruct|llama3|256|2e-5|r256_lr2e5|1"
  "Llama-3-70B|meta-llama/Meta-Llama-3-70B|meta-llama/Meta-Llama-3-70B-Instruct|llama3|128|5e-5|r128_lr5e5|1"
)

for EXEC in "${GRID_70B[@]}"; do
  IFS='|' read -r M_SHORT HF_BASE HF_INST TPL RANK LR CFG PER_GPU_BS <<< "$EXEC"
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"
  echo "  ${M_SHORT} ${CFG} (rank=${RANK}, lr=${LR})"

  for DS in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET SUFFIX MAX_SAMPLES SAVE_STEPS <<< "$DS"
    K=$(format_k "$MAX_SAMPLES")
    SCRIPT=$(generate_train_script "$M_SHORT" "$HF_BASE" "$HF_INST" "$TPL" \
             "$RANK" "$LR" "$CFG" "$DATASET" "$SUFFIX" "$MAX_SAMPLES" "$SAVE_STEPS" "$PER_GPU_BS")
    echo "    $SCRIPT"

    REL_ROOT="${MONTHDAY}/result-${M_SHORT}-${MONTHDAY}"
    for MERGE_TAG in "B2I" "I2I"; do
      [[ "$MERGE_TAG" == "B2I" ]] && SRC_TAG="B" || SRC_TAG="I"
      DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
      REL="${REL_ROOT}/${DIR}/merged-${MERGE_TAG}"
      ABBR="${M_SHORT}-${SUFFIX}-${K}-${CFG}-${MERGE_TAG}"
      ALL_EVAL_ENTRIES+=("    ('${ABBR}','\$RESULTS_DIR/${REL}'),")
    done
  done
done

# --- Generate eval config ---
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_0404_grid_${TIMESTAMP}.py"
generate_eval_config "$EVAL_CONFIG" "${ALL_EVAL_ENTRIES[@]}"
echo ""
echo "INFO  Eval config: $EVAL_CONFIG"

###############################################################################
# Output summary
###############################################################################
echo ""
echo "=========================================="
echo "  Scripts Generated!"
echo "=========================================="
echo ""
echo "=== Phase 1: 32B Grid Search (8 new runs) ==="
echo "  NEW: rank=64 × {2e-4,1e-4,5e-5,2e-5} + {r128,r256,r512}×lr2e-4 + r512/lr1e-4"
echo "  DONE(0402): r256/1e-4, r128/1e-4, r256/5e-5"
echo "  DONE(0404): r128/5e-5, r128/2e-5, r256/2e-5, r512/5e-5, r512/2e-5"
echo ""
echo "=== Phase 2: 70B (run after 32B eval) ==="
echo "  r256/lr5e-5, r256/lr2e-5, r128/lr5e-5  (BS=1, ZeRO-3)"
echo ""
echo "=== Eval (Math-7, shared -r tag) ==="
echo "  cd opencompass && python3 ./run.py $(basename "$EVAL_CONFIG") -r eval0404"
echo ""
echo "=== Prerequisites ==="
echo "  bash src/copy_files.sh \$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')"
echo ""
