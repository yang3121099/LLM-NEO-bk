#!/usr/bin/env bash
###############################################################################
# generate_70b_medical.sh — Meta-Llama-3-70B Medical Shadow-FT
#
# Training: medical_o1_reasoning_2k (2K samples, post Llama-3 cutoff)
# Eval:     Medical (MedQA + medmcqa + MedBench) + Math-7 (alignment preservation)
#
# Configs: cfgC-equivalent (rank=256, lr=5e-5) + two variants
#
# Usage:
#   bash generate_70b_medical.sh
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
MONTHDAY=$(date +%m%d)

# --- Dataset ---
DATASETS=(
  "medical_o1_reasoning_2k|med2k|2000|200"
)

# --- Model ---
MODEL_SHORT="Llama3-70B"
HF_BASE="meta-llama/Meta-Llama-3-70B"
HF_INSTRUCT="meta-llama/Meta-Llama-3-70B-Instruct"
TEMPLATE="llama3"

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
  local RANK="$1" LR="$2" CFG_NAME="$3"
  local DATASET="$4" SUFFIX="$5" MAX_SAMPLES="$6" SAVE_STEPS="$7"

  local LR_DEC LR_TAG K REL_ROOT SCRIPT_FILE
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"
  K=$(format_k "$MAX_SAMPLES")
  REL_ROOT="${MONTHDAY}/result-${MODEL_SHORT}-${MONTHDAY}"
  SCRIPT_FILE="$SCRIPT_DIR/train_${MODEL_SHORT}_${SUFFIX}_${K}_${CFG_NAME}_${TIMESTAMP}.sh"

  cat > "$SCRIPT_FILE" << HEADER
#!/usr/bin/env bash
###############################################################################
# ${MODEL_SHORT} Medical Shadow-FT: ${CFG_NAME} (rank=${RANK}, lr=${LR})
# Dataset: ${DATASET} (${K} samples)
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$WORKSPACE_DIR"
RESULTS_DIR="$RESULTS_DIR"

NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "\$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
echo "INFO: Detected \$NUM_GPUS GPU(s)"

PER_GPU_BS=${DEFAULT_PER_GPU_BS}
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
# Generate eval config (Medical + Math-7)
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
    ######################### Medical (medbench commented out) #########################
    from opencompass.configs.datasets.MedQA.MedQA_gen_3bf756 import MedQA_datasets
    from opencompass.configs.datasets.medmcqa.medmcqa_gen_60c8f5 import medmcqa_datasets
    # from opencompass.configs.datasets.MedBench.medbench_gen_0b4fff import medbench_datasets
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_gen_a67965 import proteinlmbench_datasets
    from opencompass.configs.datasets.Medbullets.medbullets_gen_60c8f5 import medbullets_datasets

    ######################### Math-7 (alignment preservation) #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ---------------------------------------------------------------------------
# Inline summarizer (Medical + Math-7)
# ---------------------------------------------------------------------------

medical_groups = [
    dict(
        name="Medical",
        subsets=[
            ["MedQA_US", "accuracy"],
            ["MedQA_Mainland", "accuracy"],
            ["MedQA_Taiwan", "accuracy"],
            ["medmcqa", "accuracy"],
            ["ProteinLMBench", "accuracy"],
            ["medbullets", "accuracy"],
        ],
    )
]

math_groups = [
    dict(
        name="Math",
        subsets=[
            ["math", "accuracy"],
            ["math-500", "accuracy"],
            ["minerva_math", "accuracy"],
            ["gsm8k", "accuracy"],
            ["gsm8k_0shot", "accuracy"],
            ["aime2024", "accuracy"],
            ["svamp", "accuracy"],
        ],
    )
]

average_groups = [
    {"name": "average_medical", "subsets": [["Medical", "naive_average"]]},
    {"name": "average_math", "subsets": [["Math", "naive_average"]]},
    {
        "name": "overall_average",
        "subsets": [
            ["average_medical", "naive_average"],
            ["average_math", "naive_average"],
        ],
    },
]

dataset_abbrs = [
    "--------- Medical ---------",
    ["MedQA_US", "accuracy"],
    ["MedQA_Mainland", "accuracy"],
    ["MedQA_Taiwan", "accuracy"],
    ["medmcqa", "accuracy"],
    ["ProteinLMBench", "accuracy"],
    ["medbullets", "accuracy"],
    "",
    "--------- Math ---------",
    ["math", "accuracy"],
    ["math-500", "accuracy"],
    ["minerva_math", "accuracy"],
    ["gsm8k", "accuracy"],
    ["gsm8k_0shot", "accuracy"],
    ["aime2024", "accuracy"],
    ["svamp", "accuracy"],
    "",
    "--------- Section AVG ---------",
    ["Medical", "naive_average"],
    ["Math", "naive_average"],
    "",
    "--------- Overall AVG ---------",
    ["overall_average", "naive_average"],
]

summary_groups = medical_groups + math_groups + average_groups

summarizer = dict(
    dataset_abbrs=dataset_abbrs,
    summary_groups=summary_groups,
)

PYHEADER

  {
    echo "Baseline_settings = ["
    echo "('Meta-Llama-3-70B-Instruct-hf', 'meta-llama/Meta-Llama-3-70B-Instruct'),"
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
echo "  70B Medical Shadow-FT Experiments"
echo "  Model: ${MODEL_SHORT} (${HF_BASE})"
echo "  Dataset: medical_o1_reasoning_2k"
echo "  Timestamp: $TIMESTAMP"
echo "=========================================="

ALL_EVAL_ENTRIES=()

# ==========================================================================
# Configs: based on 32B cfgC best (rank=256, lr=5e-5) + two variants
# ==========================================================================
GRID_70B=(
  # cfgC equivalent
  "256|5e-5|med_cfgC"
  # Higher rank variant
  "512|5e-5|med_r512"
  # Lower LR variant
  "256|2e-5|med_lr2e5"
)

echo ""
echo "=== Training Configs ==="

for EXEC in "${GRID_70B[@]}"; do
  IFS='|' read -r RANK LR CFG <<< "$EXEC"
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"
  echo "  ${MODEL_SHORT} ${CFG} (rank=${RANK}, lr=${LR})"

  for DS in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET SUFFIX MAX_SAMPLES SAVE_STEPS <<< "$DS"
    K=$(format_k "$MAX_SAMPLES")
    SCRIPT=$(generate_train_script "$RANK" "$LR" "$CFG" "$DATASET" "$SUFFIX" "$MAX_SAMPLES" "$SAVE_STEPS")
    echo "    $SCRIPT"

    REL_ROOT="${MONTHDAY}/result-${MODEL_SHORT}-${MONTHDAY}"
    for MERGE_TAG in "B2I" "I2I"; do
      [[ "$MERGE_TAG" == "B2I" ]] && SRC_TAG="B" || SRC_TAG="I"
      DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
      REL="${REL_ROOT}/${DIR}/merged-${MERGE_TAG}"
      ABBR="${MODEL_SHORT}-${SUFFIX}-${K}-${CFG}-${MERGE_TAG}"
      ALL_EVAL_ENTRIES+=("    ('${ABBR}','\$RESULTS_DIR/${REL}'),")
    done
  done
done

# --- Generate eval config ---
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_70b_medical_${TIMESTAMP}.py"
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
echo "=== Training (3 configs × B+I = 6 runs) ==="
echo "  med_cfgC:  rank=256, lr=5e-5  (best from 32B grid search)"
echo "  med_r512:  rank=512, lr=5e-5  (higher capacity)"
echo "  med_lr2e5: rank=256, lr=2e-5  (more conservative)"
echo ""
echo "=== Eval (Medical-6 + Math-7) ==="
echo "  Medical: MedQA (US/Mainland/Taiwan), medmcqa, ProteinLMBench, medbullets"
echo "  Math:    AIME2024, MATH, minerva_math, SVAMP, GSM8K, GSM8K-0shot, MATH-500"
echo "  Baseline: Meta-Llama-3-70B-Instruct"
echo ""
echo "  cd opencompass && python3 ./run.py $(basename "$EVAL_CONFIG") -r eval70b_med"
echo ""
echo "=== Prerequisites ==="
echo "  bash src/copy_files.sh \$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')"
echo ""
echo "=== Execution order ==="
echo "  1. Run training scripts sequentially (each takes ~several hours on 8×A100/H100)"
echo "  2. Run eval after all merges complete"
echo ""
