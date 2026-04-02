#!/usr/bin/env bash
###############################################################################
# generate_32b_experiments.sh — Qwen2.5-32B & Qwen3-30B-A3B Shadow-FT
#
# Generates training scripts with 2K samples, hyperparameter sweep,
# plus comprehensive eval config (all benchmarks except code).
#
# Execution order:
#   1. Qwen2.5-32B   cfgA (rank=256, lr=1e-4)  ← primary
#   2. Qwen3-30B-A3B cfgA (rank=256, lr=1e-4)  ← MoE comparison
#   3. Qwen2.5-32B   cfgC (rank=256, lr=5e-5)  ← conservative
#   4. Qwen2.5-32B   cfgB (rank=128, lr=1e-4)  ← ablation
#
# Usage:
#   bash generate_32b_experiments.sh
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_DIR"

# --- Fixed training constants ---
EFFECTIVE_BS=256
DEFAULT_PER_GPU_BS=1      # 32B model → BS=1 per GPU
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
CUTOFF_LEN=4096
VAL_SIZE=0.01

TIMESTAMP=$(date +%m%d%H%M%S)
MONTHDAY=$(date +%m%d)

# --- Single Shadow_2k dataset ---
DATASETS=(
  "Shadow_2k|shadow2k|2000|200"
)

# --- Hyperparameter configs ---
# Format: "rank|lr|config_name"
CONFIGS=(
  "256|1e-4|cfgA"     # Primary: large rank + moderate LR
  "128|1e-4|cfgB"     # Ablation: 8B-optimal rank, halved LR
  "256|5e-5|cfgC"     # Conservative: large rank + safe LR
)

# --- Models ---
# Format: "model_short|hf_base|hf_instruct|template"
MODELS=(
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen"
  "Qwen3-30B-A3B|Qwen/Qwen3-30B-A3B-Base|Qwen/Qwen3-30B-A3B|qwen3"
)

# --- Helpers ---
to_decimal() { LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//; s/\.$/.0/'; }
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then echo "$((num / 1000))k"
  else awk -v n="$num" 'BEGIN{ printf "%.1fk", n/1000 }'; fi
}

###############################################################################
# Generate training script for one (model, config, dataset) combination
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
# Generate comprehensive eval config (all benchmarks except code)
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

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################

    ######################### Reasoning-9 (general reasoning) #########################
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets  # few-shot
    from opencompass.configs.datasets.bbh.bbh_0shot_nocot_gen_925fc4 import bbh_datasets as bbh3_datasets  # 0-shot
    from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
    from opencompass.configs.datasets.TheoremQA.ThroremQA_0shot_cot_gen_8acdf7 import TheoremQA_datasets

    ######################### Math-7 (mathematical) #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets   # noqa: F401, F403
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets  # minerva_math
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets  # MATH
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets  # 0-shot eval_v2
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

PYHEADER

  # Baselines + trained models
  {
    echo "Baseline_settings = ["
    echo "('Qwen2.5-32B-Instruct-hf', 'Qwen/Qwen2.5-32B-Instruct'),"
    echo "('Qwen3-30B-A3B-hf', 'Qwen/Qwen3-30B-A3B'),"
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
# Main: Generate scripts in specified execution order
###############################################################################
echo ""
echo "=========================================="
echo "  32B Experiment Generator"
echo "  Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

ALL_EVAL_ENTRIES=()
ORDERED_SCRIPTS=()

# --- Execution order ---
# Format: "model_short|hf_base|hf_instruct|template|rank|lr|cfg|per_gpu_bs"
# 1. Qwen2.5-32B cfgA     (dense 32B → bs=1)
# 2. Qwen3-30B-A3B cfgA   (MoE 3B active → bs=4)
# 3. Qwen2.5-32B cfgC
# 4. Qwen2.5-32B cfgB
EXEC_ORDER=(
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|256|1e-4|cfgA|1"
  "Qwen3-30B-A3B|Qwen/Qwen3-30B-A3B-Base|Qwen/Qwen3-30B-A3B|qwen3|256|1e-4|cfgA|4"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|256|5e-5|cfgC|1"
  "Qwen2.5-32B|Qwen/Qwen2.5-32B|Qwen/Qwen2.5-32B-Instruct|qwen|128|1e-4|cfgB|1"
)

for EXEC in "${EXEC_ORDER[@]}"; do
  IFS='|' read -r M_SHORT HF_BASE HF_INST TPL RANK LR CFG PER_GPU_BS <<< "$EXEC"
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"

  echo "--- ${M_SHORT} ${CFG} (rank=${RANK}, lr=${LR}, bs=${PER_GPU_BS}) ---"

  for DS in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET SUFFIX MAX_SAMPLES SAVE_STEPS <<< "$DS"
    K=$(format_k "$MAX_SAMPLES")

    SCRIPT=$(generate_train_script "$M_SHORT" "$HF_BASE" "$HF_INST" "$TPL" \
             "$RANK" "$LR" "$CFG" "$DATASET" "$SUFFIX" "$MAX_SAMPLES" "$SAVE_STEPS" "$PER_GPU_BS")
    ORDERED_SCRIPTS+=("$SCRIPT")
    echo "  $SCRIPT"

    # Collect eval entries
    REL_ROOT="${MONTHDAY}/result-${M_SHORT}-${MONTHDAY}"
    for MERGE_TAG in "B2I" "I2I"; do
      if [[ "$MERGE_TAG" == "B2I" ]]; then SRC_TAG="B"; else SRC_TAG="I"; fi
      DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
      REL="${REL_ROOT}/${DIR}/merged-${MERGE_TAG}"
      ABBR="${M_SHORT}-${SUFFIX}-${K}-${CFG}-${MERGE_TAG}"
      ALL_EVAL_ENTRIES+=("    ('${ABBR}','\$RESULTS_DIR/${REL}'),")
    done
  done
done

# --- Generate eval config ---
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_32b_${TIMESTAMP}.py"
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
echo "=== Execution Order ==="
echo ""
STEP=1
PREV_MODEL=""
for EXEC in "${EXEC_ORDER[@]}"; do
  IFS='|' read -r M_SHORT _ _ _ RANK LR CFG BS <<< "$EXEC"
  LABEL="${M_SHORT} ${CFG} (rank=${RANK}, lr=${LR}, bs=${BS})"
  if [[ "$LABEL" != "$PREV_MODEL" ]]; then
    echo "--- Step ${STEP}: ${LABEL} ---"
    STEP=$((STEP + 1))
    PREV_MODEL="$LABEL"
  fi
  for DS in "${DATASETS[@]}"; do
    IFS='|' read -r _ SUFFIX MAX_SAMPLES _ <<< "$DS"
    K=$(format_k "$MAX_SAMPLES")
    echo "  bash scripts/train_${M_SHORT}_${SUFFIX}_${K}_${CFG}_${TIMESTAMP}.sh"
  done
done
echo ""
echo "--- Evaluation (after each step or all at once) ---"
echo "  cd opencompass && python3 ./run.py eval_32b_${TIMESTAMP}.py -r eval32b"
echo "  # Supports -r resume: run after each training step to accumulate results"
echo ""
echo "=== Prerequisites ==="
echo "  nohup bash scripts/download_models.sh > download_models.log 2>&1 &"
echo "  bash scripts/download_eval_data.sh"
echo "  bash src/copy_files.sh \$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')"
echo ""
echo "=== Datasets (16 benchmarks, no code) ==="
echo "  Math(7):    math-500, minerva_math, MATH, gsm8k, gsm8k_0shot, aime2024, svamp"
echo "  General(9): mmlu, mmlu_pro, bbh(few-shot), bbh(0-shot), drop, winogrande, ARC-c, GPQA, TheoremQA"
echo ""
