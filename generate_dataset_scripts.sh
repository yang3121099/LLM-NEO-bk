#!/usr/bin/env bash
###############################################################################
# generate_dataset_scripts.sh — Generate Shadow-FT training scripts for
# multiple datasets × models × sample sizes
#
# Usage:  bash generate_dataset_scripts.sh
#
# Generates one script per (dataset, model, sample_size) combination.
# Each script does: Train Base LoRA + Train Instruct LoRA + Merge B2I/I2I
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# --- Models ------------------------------------------------------------------
BASE_MODELS=(
  "Llama3.1-8B"
  "Qwen3-8B"
)

# --- Training constants ------------------------------------------------------
USE_LORA=true
LORA_RANK=128
LR=2e-4
PER_DEVICE_TRAIN_BS=16
GRAD_ACCUM_STEPS=16
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
SAVE_STEPS=1000
CUTOFF_LEN=16384
VAL_SIZE=0.01

# --- Helpers -----------------------------------------------------------------
to_decimal() { LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//; s/\.$/.0/'; }
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then echo "$((num / 1000))k"
  else awk -v n="$num" 'BEGIN{ printf "%.1fk", n/1000 }'; fi
}

LR_DEC="$(to_decimal "$LR")"
LR_TAG="lr${LR_DEC}"

# --- Dataset definitions: name | suffix | samples | save_steps --------------
#     Format: "dataset_name|suffix|max_samples|save_steps"
EXPERIMENTS=(
  # Dolci Instruct Mix
  "dolci_instruct_mix|dolci_mix|100000|1000"
  "dolci_instruct_mix|dolci_mix|20000|1000"
  # Dolci Instruct Tool Use
  "dolci_instruct_tool_use|dolci_tool|100000|1000"
  "dolci_instruct_tool_use|dolci_tool|20000|1000"
  # Nemotron IF Chat
  "nemotron_if_chat_v1|nemotron_if|100000|1000"
  "nemotron_if_chat_v1|nemotron_if|20000|1000"
)

# --- Resolve model pairs -----------------------------------------------------
MODEL_PAIRS=()
MODEL_NAMES=()
TEMPLATES=()

for NAME in "${BASE_MODELS[@]}"; do
  BLOCK=$(awk -v n="\"$NAME\"" '
    $0~n {print; getline;
           while ($0 !~ /\}/) {print; getline}; print; exit}' \
    "$MODEL_PAIR_FILE")
  [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in $MODEL_PAIR_FILE"; exit 1; }

  HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')

  MODEL_PAIRS+=("${HF_BASE}||${HF_INST}")
  MODEL_NAMES+=("$(basename "$HF_BASE")")

  case "$HF_BASE" in
    *Llama-3*|*Llama3*) TEMPLATES+=("llama3") ;;
    *Qwen3*)            TEMPLATES+=("qwen3")  ;;
    *Qwen2*)            TEMPLATES+=("qwen")   ;;
    *Mistral*)          TEMPLATES+=("mistral") ;;
    *gemma*)            TEMPLATES+=("gemma")   ;;
    *)                  TEMPLATES+=("default") ;;
  esac

  echo "INFO  Model: $NAME  Base=$HF_BASE  Instruct=$HF_INST"
done

# --- Generate scripts --------------------------------------------------------
MONTHDAY=$(date +%m%d)
TIMESTAMP=$(date +%m%d%H%M%S)

for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r DATASET SUFFIX MAX_SAMPLES EXP_SAVE_STEPS <<< "$EXP"
  K="$(format_k "$MAX_SAMPLES")"

  for i in "${!MODEL_PAIRS[@]}"; do
    PAIR="${MODEL_PAIRS[$i]}"
    MODEL_BASE="${MODEL_NAMES[$i]}"
    template="${TEMPLATES[$i]}"
    B_MODEL="${PAIR%%||*}"
    I_MODEL="${PAIR##*||}"

    SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${SUFFIX}_${K}_${MODEL_BASE}_${TIMESTAMP}.sh"
    REL_ROOT="${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

    cat > "$SCRIPT_FILE" << HEADER
#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated $(date '+%F %T') #####
# Model     : $MODEL_BASE ($B_MODEL / $I_MODEL)
# Dataset   : $DATASET ($MAX_SAMPLES samples)
# LoRA rank : $LORA_RANK
# Template  : $template
# BS=$PER_DEVICE_TRAIN_BS × grad_accum=$GRAD_ACCUM_STEPS = effective ${PER_DEVICE_TRAIN_BS}*${GRAD_ACCUM_STEPS}

##### Paths (resolved at runtime) #####
WORKSPACE_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
RESULTS_DIR="\$WORKSPACE_DIR/results"

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

HEADER

    # --- Training commands for Base and Instruct ---
    for TAG_INFO in "B|$B_MODEL" "I|$I_MODEL"; do
      IFS='|' read -r TAG M_PATH <<< "$TAG_INFO"
      DIR="${TAG}-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}"
      REL_OUTDIR="${REL_ROOT}/$DIR"

      cat >> "$SCRIPT_FILE" << TRAIN
###############################################################################
##### Train ${TAG} (${M_PATH}) #####
###############################################################################
mkdir -p "\$RESULTS_DIR/${REL_OUTDIR}"
cd "\$WORKSPACE_DIR"
llamafactory-cli train \\
  --model_name_or_path "${M_PATH}" \\
  --stage sft \\
  --do_train true \\
  --finetuning_type lora --lora_rank ${LORA_RANK} \\
  --dataset "${DATASET}" \\
  --template "${template}" \\
  --cutoff_len ${CUTOFF_LEN} \\
  --max_samples ${MAX_SAMPLES} \\
  --output_dir "\$RESULTS_DIR/${REL_OUTDIR}" \\
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS} \\
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \\
  --learning_rate ${LR_DEC} \\
  --num_train_epochs ${NUM_EPOCHS} \\
  --logging_steps ${LOGGING_STEPS} \\
  --save_steps ${EXP_SAVE_STEPS} \\
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
  --use_fast_tokenizer True

TRAIN
    done

    # --- Merge commands ---
    for MERGE in "B|I|" "I|I|" "I|B|# " "B|B|# "; do
      IFS='|' read -r SRC_TAG TGT_TAG PREFIX <<< "$MERGE"

      SRC_DIR="${SRC_TAG}-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}"
      REL_ADAP="${REL_ROOT}/${SRC_DIR}"
      MERGE_TAG="${SRC_TAG}2${TGT_TAG}"

      if [[ "$TGT_TAG" == "I" ]]; then TGT_MODEL="$I_MODEL"; else TGT_MODEL="$B_MODEL"; fi

      cat >> "$SCRIPT_FILE" << MERGE_CMD
### Merge: ${MERGE_TAG} (adapter=${SRC_TAG}, target=${TGT_TAG}) ###
${PREFIX}mkdir -p "\$RESULTS_DIR/${REL_ADAP}/merged-${MERGE_TAG}"
${PREFIX}python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
${PREFIX}  --adapter_path "\$RESULTS_DIR/${REL_ADAP}" \\
${PREFIX}  --target_base "${TGT_MODEL}" \\
${PREFIX}  --merge_tag "${MERGE_TAG}" \\
${PREFIX}  --template "${template}"

MERGE_CMD
    done

    chmod +x "$SCRIPT_FILE"
    echo "INFO  Generated: $SCRIPT_FILE"
  done
done

echo ""
echo "========================================"
echo "  Script generation complete!"
echo "========================================"
echo ""
echo "Generated scripts in: $SCRIPT_OUTPUT_DIR/"
echo ""
echo "Run examples:"
for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r DATASET SUFFIX MAX_SAMPLES _ <<< "$EXP"
  K="$(format_k "$MAX_SAMPLES")"
  echo "  bash scripts/train_${SUFFIX}_${K}_${MODEL_NAMES[0]}_${TIMESTAMP}.sh"
done
echo ""
