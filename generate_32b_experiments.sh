#!/usr/bin/env bash
###############################################################################
# generate_32b_experiments.sh — Qwen3-32B hyperparameter sweep for Shadow-FT
#
# Generates training scripts for multiple LoRA rank × LR combinations,
# plus a batch merge+eval script. Designed for reviewer rebuttal experiments.
#
# Hyperparameter configs (based on 8B optimal scaling):
#   Config A (primary):  rank=256  lr=1e-4   ← 32B sweet spot
#   Config B (baseline): rank=128  lr=1e-4   ← 8B best rank, halved LR
#   Config C (safe):     rank=256  lr=5e-5   ← conservative, very stable
#
# Usage:
#   bash generate_32b_experiments.sh
#   # Then run generated scripts in scripts/
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_DIR="$WORKSPACE_DIR/scripts"
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
mkdir -p "$RESULTS_DIR" "$SCRIPT_DIR"

# --- Model ---
HF_BASE="Qwen/Qwen3-32B-Base"
HF_INSTRUCT="Qwen/Qwen3-32B"
TEMPLATE="qwen3"
MODEL_SHORT="Qwen3-32B"

# --- Fixed training constants ---
EFFECTIVE_BS=256
DEFAULT_PER_GPU_BS=1      # 32B model → BS=1 per GPU to fit memory
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
CUTOFF_LEN=4096
VAL_SIZE=0.01

# --- Dataset: OpenR1-Math-220k (primary reviewer rebuttal dataset) ---
DATASET="openr1_math_220k"
SUFFIX="openr1"
MAX_SAMPLES=220000
SAVE_STEPS=100

# --- Hyperparameter sweep ---
# Format: "rank|lr|config_name"
CONFIGS=(
  "256|1e-4|cfgA"     # Primary: large rank + moderate LR (32B sweet spot)
  "128|1e-4|cfgB"     # Baseline: 8B-optimal rank, halved LR for 32B
  "256|5e-5|cfgC"     # Conservative: large rank + safe LR
)

# --- Helpers ---
to_decimal() { LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//; s/\.$/.0/'; }
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then echo "$((num / 1000))k"
  else awk -v n="$num" 'BEGIN{ printf "%.1fk", n/1000 }'; fi
}

TIMESTAMP=$(date +%m%d%H%M%S)
MONTHDAY=$(date +%m%d)
K=$(format_k "$MAX_SAMPLES")
REL_ROOT="${MONTHDAY}/result-${MODEL_SHORT}-Base-${MONTHDAY}"

echo "=========================================="
echo "  Qwen3-32B Shadow-FT Experiment Generator"
echo "  Timestamp: $TIMESTAMP"
echo "  Dataset: $DATASET ($K samples)"
echo "=========================================="
echo ""

# --- Generate one training script per config ---
ALL_SCRIPTS=()
ALL_EVAL_ENTRIES=()

for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r RANK LR CFG_NAME <<< "$CFG"
  LR_DEC=$(to_decimal "$LR")
  LR_TAG="lr${LR_DEC}"

  SCRIPT_FILE="$SCRIPT_DIR/train_32b_${SUFFIX}_${K}_${CFG_NAME}_${TIMESTAMP}.sh"
  ALL_SCRIPTS+=("$SCRIPT_FILE")

  echo "INFO  Config ${CFG_NAME}: rank=${RANK} lr=${LR}"

  cat > "$SCRIPT_FILE" << HEADER
#!/usr/bin/env bash
###############################################################################
# Qwen3-32B Shadow-FT: ${CFG_NAME} (rank=${RANK}, lr=${LR})
# Dataset: ${DATASET} (${K} samples)
# Generated: ${TIMESTAMP}
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$WORKSPACE_DIR"
RESULTS_DIR="$RESULTS_DIR"

# GPU auto-detection
NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "\$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
echo "INFO: Detected \$NUM_GPUS GPU(s)"

# BS calculation: bs × acc × gpus = ${EFFECTIVE_BS}
PER_GPU_BS=${DEFAULT_PER_GPU_BS}
GRAD_ACCUM=\$(( ${EFFECTIVE_BS} / (PER_GPU_BS * NUM_GPUS) ))
if [[ "\$GRAD_ACCUM" -lt 1 ]]; then
    PER_GPU_BS=1
    GRAD_ACCUM=\$(( ${EFFECTIVE_BS} / NUM_GPUS ))
fi
echo "INFO: PER_GPU_BS=\$PER_GPU_BS  GRAD_ACCUM=\$GRAD_ACCUM  NUM_GPUS=\$NUM_GPUS  (effective=${EFFECTIVE_BS})"

# Environment
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

HEADER

  # Training for Base (B) and Instruct (I)
  for TAG_INFO in "B|$HF_BASE" "I|$HF_INSTRUCT"; do
    IFS='|' read -r TAG M_PATH <<< "$TAG_INFO"
    DIR="${TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
    REL_OUTDIR="${REL_ROOT}/$DIR"

    cat >> "$SCRIPT_FILE" << TRAIN
###############################################################################
##### Train ${TAG} (${M_PATH}) — ${CFG_NAME} #####
###############################################################################
mkdir -p "\$RESULTS_DIR/${REL_OUTDIR}"
cd "\$WORKSPACE_DIR"

# Multi-GPU: FORCE_TORCHRUN + DeepSpeed ZeRO-2
DS_ARG=""
if [[ "\$NUM_GPUS" -gt 1 ]]; then
    export FORCE_TORCHRUN=1
    export NNODES=1
    export NPROC_PER_NODE=\$NUM_GPUS
    export MASTER_PORT=\$(( RANDOM % 10000 + 20000 ))
    export NCCL_TIMEOUT=7200
    DS_ARG="--deepspeed \$WORKSPACE_DIR/examples/deepspeed/ds_z2_config.json"
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
  --overwrite_cache false \\
  --use_fast_tokenizer True \\
  --preprocessing_num_workers 16 \\
  \$DS_ARG

TRAIN
  done

  # Merge commands: every 10 checkpoints + final
  for MERGE_INFO in "B|B2I|$HF_INSTRUCT" "I|I2I|$HF_INSTRUCT"; do
    IFS='|' read -r SRC_TAG MERGE_TAG TGT_MODEL <<< "$MERGE_INFO"
    SRC_DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
    REL_ADAP="${REL_ROOT}/${SRC_DIR}"

    cat >> "$SCRIPT_FILE" << MERGE
###############################################################################
##### Merge ${MERGE_TAG}: ${SRC_DIR} #####
###############################################################################
echo "=== Merging ${MERGE_TAG} checkpoints ==="
ADAPTER_BASE="\$RESULTS_DIR/${REL_ADAP}"
CKPTS=(\$(find "\$ADAPTER_BASE" -maxdepth 1 -type d -name "checkpoint-*" | \\
         awk -F'checkpoint-' '{print \$NF, \$0}' | sort -n | cut -d' ' -f2-))
TOTAL_CKPTS=\${#CKPTS[@]}
echo "Found \$TOTAL_CKPTS checkpoints"

for (( ci=0; ci<TOTAL_CKPTS; ci++ )); do
    CKPT="\${CKPTS[\$ci]}"
    STEP=\$(basename "\$CKPT" | sed 's/checkpoint-//')
    if (( (ci + 1) % 10 == 0 )) || (( ci == TOTAL_CKPTS - 1 )); then
        echo "  Merging step \$STEP (${MERGE_TAG}) ..."
        python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
            --adapter_path "\$CKPT" \\
            --target_base "${TGT_MODEL}" \\
            --merge_tag "${MERGE_TAG}" \\
            --template "${TEMPLATE}"
    fi
done

# Merge final adapter
echo "  Merging final adapter (${MERGE_TAG}) ..."
python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
    --adapter_path "\$ADAPTER_BASE" \\
    --target_base "${TGT_MODEL}" \\
    --merge_tag "${MERGE_TAG}" \\
    --template "${TEMPLATE}"

MERGE
  done

  chmod +x "$SCRIPT_FILE"
  echo "  → $SCRIPT_FILE"

  # Collect eval entries
  for MERGE_TAG in "B2I" "I2I"; do
    if [[ "$MERGE_TAG" == "B2I" ]]; then SRC_TAG="B"; else SRC_TAG="I"; fi
    DIR="${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX}"
    REL="${REL_ROOT}/${DIR}/merged-${MERGE_TAG}"
    ABBR="${MODEL_SHORT}-${SUFFIX}-${K}-${CFG_NAME}-${MERGE_TAG}"
    ALL_EVAL_ENTRIES+=("${ABBR}|\$RESULTS_DIR/${REL}")
  done
done

# --- Generate eval Python config directly ---
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_32b_${TIMESTAMP}.py"

cat > "$EVAL_CONFIG" << 'PYHEADER'
# Auto-generated Qwen3-32B eval config
import subprocess as _sp
import os as _os
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate

_NUM_GPUS = max(1, len(_sp.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()))
RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results')

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Qwen3-32B Instruct baseline
HF_baselines = [
    ('Qwen3-32B-Instruct-hf', 'Qwen/Qwen3-32B'),
]

# Trained models (B2I Shadow-FT, I2I baseline)
Baseline_settings = [
PYHEADER

# Append trained model entries
for ENTRY in "${ALL_EVAL_ENTRIES[@]}"; do
  IFS='|' read -r ABBR REL_PATH <<< "$ENTRY"
  echo "    ('${ABBR}', '${REL_PATH}')," >> "$EVAL_CONFIG"
done

cat >> "$EVAL_CONFIG" << 'PYFOOTER'
]

models = []

for abbr, path in HF_baselines:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=2048, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=1024,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )

for abbr, path in Baseline_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=2048, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=1024,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )
PYFOOTER

echo "INFO  Generated eval config: $EVAL_CONFIG"

echo ""
echo "=========================================="
echo "  Qwen3-32B Experiment Scripts Generated!"
echo "=========================================="
echo ""
echo "=== Training (run in order of priority) ==="
for S in "${ALL_SCRIPTS[@]}"; do
  echo "  bash $S"
done
echo ""
echo "=== Evaluation (after training completes) ==="
echo "  cd opencompass && python3 ./run.py eval_32b_${TIMESTAMP}.py -r eval32b"
echo ""
echo "=== Model download (run first on slow networks) ==="
echo "  nohup bash scripts/download_models.sh > download_models.log 2>&1 &"
echo ""
echo "=== Hyperparameter configs ==="
for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r RANK LR CFG_NAME <<< "$CFG"
  echo "  ${CFG_NAME}: rank=${RANK}  lr=${LR}"
done
echo ""
echo "=== Recommendation ==="
echo "  Start with cfgA (rank=256, lr=1e-4) — most likely to beat 8B results."
echo "  Run cfgB as ablation (same rank=128 as 8B, but LR halved for 32B)."
echo "  Use cfgC only if cfgA shows loss instability."
echo ""
