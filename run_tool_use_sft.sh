#!/usr/bin/env bash
###############################################################################
# run_tool_use_sft.sh — Tool-Use SFT training pipeline
#
# Trains Llama3.1-8B and Qwen3-8B on the Dolci-Instruct-SFT tool-use subset
# (2k samples) using LoRA, then merges for Shadow-FT (B2I) and baseline (I2I).
#
# Prerequisites:
#   1. python3 scripts/prepare_dolci_tool_use.py   # Download & convert data
#   2. bash run_tool_use_sft.sh                     # Run training
#
# Usage:
#   bash run_tool_use_sft.sh              # Train all models
#   bash run_tool_use_sft.sh --model llama  # Train Llama only
#   bash run_tool_use_sft.sh --model qwen   # Train Qwen only
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
DATA_DIR="$WORKSPACE_DIR/data"
SCRIPT_DIR="$WORKSPACE_DIR/scripts"

# --- Parse arguments --------------------------------------------------------
FILTER_MODEL="${1:-}"
[[ "$FILTER_MODEL" == "--model" ]] && FILTER_MODEL="${2:-}" || true

# --- Check data exists ------------------------------------------------------
TOOL_DATA="$DATA_DIR/dolci_instruct_tool_use_converted.json"
if [[ ! -f "$TOOL_DATA" ]]; then
    echo "Tool-use dataset not found. Preparing data ..."
    python3 "$SCRIPT_DIR/prepare_dolci_tool_use.py"
    if [[ ! -f "$TOOL_DATA" ]]; then
        echo "ERROR: Data preparation failed. File not found: $TOOL_DATA"
        exit 1
    fi
fi

SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$TOOL_DATA'))))")
echo "INFO: Tool-use dataset has $SAMPLE_COUNT samples"

# --- Training constants -----------------------------------------------------
DATASET="dolci_instruct_tool_use"
SUFFIX="dolci_tool"
MAX_SAMPLES=2000
LORA_RANK=128
LR="0.0002"
EFFECTIVE_BS=256
DEFAULT_PER_GPU_BS=2
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
CUTOFF_LEN=4096
VAL_SIZE=0.01
SAVE_STEPS=1000

# --- GPU auto-detection -----------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "WARNING: No GPU detected, defaulting to 1"
    NUM_GPUS=1
fi
echo "INFO: Detected $NUM_GPUS GPU(s)"

PER_GPU_BS=$DEFAULT_PER_GPU_BS
GRAD_ACCUM=$(( EFFECTIVE_BS / (PER_GPU_BS * NUM_GPUS) ))
if [[ "$GRAD_ACCUM" -lt 1 ]]; then
    PER_GPU_BS=1
    GRAD_ACCUM=$(( EFFECTIVE_BS / NUM_GPUS ))
fi
echo "INFO: PER_GPU_BS=$PER_GPU_BS  GRAD_ACCUM=$GRAD_ACCUM  NUM_GPUS=$NUM_GPUS  (effective=$EFFECTIVE_BS)"

# --- Environment ------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

# --- Model definitions -------------------------------------------------------
declare -A MODELS
# name -> "base_path||instruct_path||template"
MODELS["Llama3.1-8B"]="meta-llama/Llama-3.1-8B||meta-llama/Llama-3.1-8B-Instruct||llama3"
MODELS["Qwen3-8B"]="Qwen/Qwen3-8B-Base||Qwen/Qwen3-8B||qwen3"

MONTHDAY=$(date +%m%d)
TIMESTAMP=$(date +%m%d%H%M%S)

# --- Helper: run training for one model ------------------------------------
train_model() {
    local MODEL_NAME="$1"
    local INFO="${MODELS[$MODEL_NAME]}"
    local B_MODEL="${INFO%%||*}"
    local REST="${INFO#*||}"
    local I_MODEL="${REST%%||*}"
    local TEMPLATE="${REST##*||}"
    local MODEL_BASE
    MODEL_BASE=$(basename "$B_MODEL")

    local REL_ROOT="${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

    echo ""
    echo "========================================================================"
    echo "  Training: $MODEL_NAME ($MODEL_BASE)"
    echo "  Base:     $B_MODEL"
    echo "  Instruct: $I_MODEL"
    echo "  Template: $TEMPLATE"
    echo "  Dataset:  $DATASET ($MAX_SAMPLES samples)"
    echo "========================================================================"

    # Multi-GPU setup
    DS_ARG=""
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        export FORCE_TORCHRUN=1
        export NNODES=1
        export NPROC_PER_NODE=$NUM_GPUS
        export MASTER_PORT=$(( RANDOM % 10000 + 20000 ))
        export NCCL_TIMEOUT=7200
        DS_ARG="--deepspeed $WORKSPACE_DIR/examples/deepspeed/ds_z2_config.json"
    fi

    # Train on Base model (B)
    for TAG_INFO in "B|$B_MODEL" "I|$I_MODEL"; do
        IFS='|' read -r TAG M_PATH <<< "$TAG_INFO"
        DIR="${TAG}-2k-lora-rank${LORA_RANK}-lr${LR}-${SUFFIX}"
        REL_OUTDIR="${REL_ROOT}/$DIR"

        echo ""
        echo "###############################################################################"
        echo "##### Train ${TAG} (${M_PATH}) #####"
        echo "###############################################################################"
        mkdir -p "$RESULTS_DIR/${REL_OUTDIR}"
        cd "$WORKSPACE_DIR"

        llamafactory-cli train \
          --model_name_or_path "${M_PATH}" \
          --stage sft \
          --do_train true \
          --finetuning_type lora --lora_rank ${LORA_RANK} \
          --dataset "${DATASET}" \
          --template "${TEMPLATE}" \
          --cutoff_len ${CUTOFF_LEN} \
          --max_samples ${MAX_SAMPLES} \
          --output_dir "$RESULTS_DIR/${REL_OUTDIR}" \
          --per_device_train_batch_size $PER_GPU_BS \
          --gradient_accumulation_steps $GRAD_ACCUM \
          --learning_rate ${LR} \
          --num_train_epochs ${NUM_EPOCHS} \
          --logging_steps ${LOGGING_STEPS} \
          --save_steps ${SAVE_STEPS} \
          --save_only_model True \
          --plot_loss true \
          --lr_scheduler_type ${LR_SCHEDULER} \
          --warmup_ratio ${WARMUP_RATIO} \
          --bf16 ${BF16} \
          --val_size ${VAL_SIZE} \
          --per_device_eval_batch_size 1 \
          --eval_strategy steps \
          --eval_steps 10000 \
          --trust_remote_code True \
          --flash_attn fa2 \
          --overwrite_cache false \
          --use_fast_tokenizer True \
          --preprocessing_num_workers 16 \
          $DS_ARG

        echo "  Done: $TAG training for $MODEL_NAME"
    done

    # --- Merge LoRA adapters ---
    echo ""
    echo "###############################################################################"
    echo "##### Merge LoRA adapters for $MODEL_NAME #####"
    echo "###############################################################################"

    # B2I: Shadow-FT (Base-trained LoRA -> Instruct)
    local B_DIR="B-2k-lora-rank${LORA_RANK}-lr${LR}-${SUFFIX}"
    local I_DIR="I-2k-lora-rank${LORA_RANK}-lr${LR}-${SUFFIX}"

    echo "  Merging B2I (Shadow-FT) ..."
    python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
      --adapter_path "$RESULTS_DIR/${REL_ROOT}/${B_DIR}" \
      --target_base "${I_MODEL}" \
      --merge_tag "B2I" \
      --template "${TEMPLATE}"

    # I2I: Baseline (Instruct-trained LoRA -> Instruct)
    echo "  Merging I2I (baseline) ..."
    python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
      --adapter_path "$RESULTS_DIR/${REL_ROOT}/${I_DIR}" \
      --target_base "${I_MODEL}" \
      --merge_tag "I2I" \
      --template "${TEMPLATE}"

    echo ""
    echo "  Merge complete for $MODEL_NAME"
    echo "  B2I: $RESULTS_DIR/${REL_ROOT}/${B_DIR}/merged-B2I"
    echo "  I2I: $RESULTS_DIR/${REL_ROOT}/${I_DIR}/merged-I2I"
}

# --- Main: train selected models -------------------------------------------
mkdir -p "$RESULTS_DIR"

for MODEL_NAME in "Llama3.1-8B" "Qwen3-8B"; do
    # Filter by model name if specified
    if [[ -n "$FILTER_MODEL" ]]; then
        case "$FILTER_MODEL" in
            llama|Llama|LLAMA) [[ "$MODEL_NAME" != *Llama* ]] && continue ;;
            qwen|Qwen|QWEN)   [[ "$MODEL_NAME" != *Qwen* ]] && continue ;;
            *) [[ "$MODEL_NAME" != *"$FILTER_MODEL"* ]] && continue ;;
        esac
    fi

    train_model "$MODEL_NAME"
done

echo ""
echo "========================================================================"
echo "  Tool-Use SFT Training Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate with OpenCompass (T-Eval):"
echo "     cd opencompass && python3 ./run.py ./eval_generated.py -r $TIMESTAMP"
echo ""
echo "  2. Evaluate with EvalScope (BFCL + ToolBench):"
echo "     python3 scripts/eval_tooluse_evalscope.py"
echo ""
