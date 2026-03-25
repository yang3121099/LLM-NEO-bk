#!/usr/bin/env bash
###############################################################################
# run_openr1_200k.sh — Shadow-FT pipeline for OpenR1-Math-220k (200k samples)
#
# Usage:  bash run_openr1_200k.sh
#
# Saves LoRA checkpoints every ~2k samples for on-demand merge + eval.
# Storage-friendly: does NOT auto-merge all checkpoints.
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# --- Training mode -----------------------------------------------------------
USE_LORA=true
is_lora() { [[ "${USE_LORA,,}" == "true" ]]; }

# --- Hyperparameters ---------------------------------------------------------
lora_ranks=(128)
learning_rates_lora=(2e-4)
learning_rates_sft=(1e-5)

if is_lora; then
  learning_rates=("${learning_rates_lora[@]}")
else
  learning_rates=("${learning_rates_sft[@]}")
fi

# --- Dataset -----------------------------------------------------------------
DATASET="openr1_math_220k"
SUFFIX_NAME="openr1_200k"
CUTOFF_LEN=16384
SAMPLES=(200000)

# --- Training constants ------------------------------------------------------
LOGGING_STEPS=10
# Save every ~2k samples: 2000 / (bs=2 * grad_accum=8) = 125 steps
SAVE_STEPS=125
PER_DEVICE_TRAIN_BS=2
GRAD_ACCUM_STEPS=8
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.05
BF16=true
VAL_SIZE=0.005
PER_DEVICE_EVAL_BS=1
EVAL_STRATEGY="steps"
EVAL_STEPS=10000
OVERWRITE_CACHE=false

# --- Helpers -----------------------------------------------------------------
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then
    echo "$((num / 1000))k"
  else
    awk -v n="$num" 'BEGIN{ printf "%.1fk", n/1000 }'
  fi
}

to_decimal() {
  LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//; s/\.$/.0/'
}

lr_tag_dec() { echo "lr$(to_decimal "$1")"; }

EVAL_LINES=()
add_eval() {
  local rel="$1" comment_level="${2:-1}"
  local prefix="#"; [[ "$comment_level" -eq 2 ]] && prefix="##"
  local short
  short=$(awk -F/ '{if(NF>=3) print $(NF-2)"/"$(NF-1)"/"$NF; else print $0}' <<< "${rel%/}")
  EVAL_LINES+=("$prefix ('$short','\$RESULTS_DIR/$rel'),")
}

###############################################################################
##### 1. Base-model resolution                                             #####
###############################################################################
MODEL_DIR=""
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"

BASE_MODELS=(
  "Llama3.1-8B"
  "Qwen3-8B"
)

MODEL_PAIRS=()

for NAME in "${BASE_MODELS[@]}"; do
  BLOCK=$(awk -v n="\"$NAME\"" '
    $0~n {print; getline;
           while ($0 !~ /\}/) {print; getline}; print; exit}' \
    "$MODEL_PAIR_FILE")
  [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in $MODEL_PAIR_FILE"; exit 1; }

  HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  [[ -z $HF_BASE || -z $HF_INST ]] && { echo "ERROR: malformed JSON for $NAME"; exit 1; }

  if [[ -n $MODEL_DIR ]]; then
    BASE_PATH="$MODEL_DIR/${HF_BASE#*/}"
    INST_PATH="$MODEL_DIR/${HF_INST#*/}"
    [[ -d $BASE_PATH ]] || { echo "ERROR: $BASE_PATH missing"; exit 1; }
    [[ -d $INST_PATH ]] || { echo "ERROR: $INST_PATH missing"; exit 1; }
  else
    BASE_PATH=$HF_BASE
    INST_PATH=$HF_INST
  fi

  MODEL_PAIRS+=("${BASE_PATH}||${INST_PATH}")
  echo "INFO  Base=$BASE_PATH  Instruct=$INST_PATH"
done

###############################################################################
##### 2. Generate per-model training scripts                               #####
###############################################################################
MONTHDAY=$(date +%m%d)
TIMESTAMP=$(date +%m%d%H%M%S)

ALL_EVAL_INSTRUCT=()
ALL_EVAL_BASE=()

for PAIR in "${MODEL_PAIRS[@]}"; do
  B_MODEL="${PAIR%%||*}"
  I_MODEL="${PAIR##*||}"
  MODEL_BASE=$(basename "$B_MODEL")
  SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_openr1_200k_${MODEL_BASE}_${TIMESTAMP}.sh"
  : > "$SCRIPT_FILE"

  # --------------- Resolve chat template ---------------
  case "$B_MODEL" in
    *Llama-3*|*Llama3*) template="llama3" ;;
    *Llama-2*|*Llama2*) template="llama2" ;;
    *Qwen3*)            template="qwen3"  ;;
    *Qwen2*)            template="qwen"   ;;
    *internlm2*)        template="intern2" ;;
    *mistral_small*)    template="mistral_small" ;;
    *Mistral*)          template="mistral" ;;
    *Falcon*)           template="falcon"  ;;
    *gemma3*)           template="gemma3"  ;;
    *gemma*)            template="gemma"   ;;
    *Yi*)               template="yi"      ;;
    *Baichuan*)         template="baichuan2" ;;
    *) echo "ERROR: unknown template for $B_MODEL"; exit 1 ;;
  esac

  REL_ROOT="${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

  # --------------- Header ---------------
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo ""
    echo "##### Auto-generated $(date '+%F %T') — OpenR1-Math-220k 200k #####"
    echo "# Model     : $MODEL_BASE"
    echo "# LoRA mode : $USE_LORA"
    echo "# Template  : $template"
    echo "# Dataset   : $DATASET (${SAMPLES[0]} samples)"
    echo "# Save ckpt every $SAVE_STEPS steps (~2k samples)"
    echo ""
    echo '##### Paths (resolved at runtime) #####'
    echo 'WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"'
    echo 'RESULTS_DIR="$WORKSPACE_DIR/results"'
    echo ""
  } >> "$SCRIPT_FILE"

  # --------------- Environment ---------------
  {
    echo "##### Environment #####"
    echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"
    echo "export HF_HUB_OFFLINE=0"
    echo "export HF_DATASETS_OFFLINE=0"
    echo "export HF_DATASETS_TRUST_REMOTE_CODE=1"
    echo "export TRUST_REMOTE_CODE=True"
    echo "export HF_ALLOW_CODE_EVAL=1"
    echo ""
  } >> "$SCRIPT_FILE"

  EVAL_LINES=()

  # ===================================================================
  #  generate_train  <model_path> <tag: B|I> <learning_rate>
  # ===================================================================
  generate_train() {
    local M_PATH=$1 TAG=$2 LR=$3
    local LR_DEC LR_TAG
    LR_DEC="$(to_decimal "$LR")"
    LR_TAG="$(lr_tag_dec "$LR")"

    for MAX in "${SAMPLES[@]}"; do
      local K; K="$(format_k "$MAX")"
      local DIR="${TAG}-${K}-$(is_lora && echo lora-rank${lora_ranks[0]} || echo sft)-${LR_TAG}-${SUFFIX_NAME}"
      local REL_OUTDIR="${REL_ROOT}/$DIR"

      {
        echo "###### ${TAG}  max=${MAX}  lr=${LR_DEC} ######"
        echo "mkdir -p \"\$RESULTS_DIR/${REL_OUTDIR}\""
        echo 'cd "$WORKSPACE_DIR"'
        echo "llamafactory-cli train \\"
        echo "  --model_name_or_path \"$M_PATH\" \\"
        echo "  --stage sft \\"
        echo "  --do_train true \\"
        if is_lora; then
          echo "  --finetuning_type lora --lora_rank ${lora_ranks[0]} \\"
        else
          echo "  --finetuning_type full \\"
        fi
        echo "  --dataset \"$DATASET\" \\"
        echo "  --template \"$template\" \\"
        echo "  --cutoff_len $CUTOFF_LEN \\"
        echo "  --max_samples $MAX \\"
        echo "  --output_dir \"\$RESULTS_DIR/${REL_OUTDIR}\" \\"
        echo "  --per_device_train_batch_size $PER_DEVICE_TRAIN_BS \\"
        echo "  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \\"
        echo "  --learning_rate $LR_DEC \\"
        echo "  --num_train_epochs $NUM_EPOCHS \\"
        echo "  --logging_steps $LOGGING_STEPS \\"
        echo "  --save_steps $SAVE_STEPS \\"
        echo "  --save_only_model True \\"
        echo "  --plot_loss true \\"
        echo "  --lr_scheduler_type $LR_SCHEDULER \\"
        echo "  --warmup_ratio $WARMUP_RATIO \\"
        echo "  --bf16 $BF16 \\"
        echo "  --val_size $VAL_SIZE \\"
        echo "  --per_device_eval_batch_size $PER_DEVICE_EVAL_BS \\"
        echo "  --eval_strategy $EVAL_STRATEGY \\"
        echo "  --eval_steps $EVAL_STEPS \\"
        echo "  --trust_remote_code True \\"
        echo "  --flash_attn fa2 \\"
        echo "  --overwrite_cache $OVERWRITE_CACHE \\"
        echo "  --use_fast_tokenizer True"
        echo ""
      } >> "$SCRIPT_FILE"

      if ! is_lora && [[ $TAG == I ]]; then
        add_eval "$REL_OUTDIR" 1
      fi
    done
  }

  # --------------- Training ---------------
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "##### Training (Base + Instruct with LoRA) — OpenR1-Math 200k             #####" >> "$SCRIPT_FILE"
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "" >> "$SCRIPT_FILE"

  for LR in "${learning_rates[@]}"; do
    generate_train "$B_MODEL" B "$LR"
    generate_train "$I_MODEL" I "$LR"
  done

  # --------------- Merge helper (only final checkpoint) ---------------
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "##### LoRA Delta Merge (final checkpoint only — merge others on demand)   #####" >> "$SCRIPT_FILE"
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "" >> "$SCRIPT_FILE"

  if is_lora; then
    merge_lora() {
      local SRC=$1 SRC_TAG=$2 TGT=$3 TGT_TAG=$4 LR=$5 COMMENT_PREFIX=${6:-}
      local LR_DEC LR_TAG
      LR_DEC="$(to_decimal "$LR")"
      LR_TAG="$(lr_tag_dec "$LR")"

      for RANK in "${lora_ranks[@]}"; do
        for MAX in "${SAMPLES[@]}"; do
          local K; K="$(format_k "$MAX")"
          local REL_ADAP="${REL_ROOT}/${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${SUFFIX_NAME}"
          local TAG="${SRC_TAG}2${TGT_TAG}"
          local REL_MERGED="${REL_ADAP}/merged-${TAG}"

          {
            echo "### Merge: ${TAG} (adapter=${SRC_TAG}, target=${TGT_TAG}) ###"
            echo "${COMMENT_PREFIX}mkdir -p \"\$RESULTS_DIR/${REL_MERGED}\""
            echo "${COMMENT_PREFIX}python3 \"\$WORKSPACE_DIR/src/shadow/merge_lora.py\" \\"
            echo "${COMMENT_PREFIX}  --adapter_path \"\$RESULTS_DIR/${REL_ADAP}\" \\"
            echo "${COMMENT_PREFIX}  --target_base \"$TGT\" \\"
            echo "${COMMENT_PREFIX}  --merge_tag \"$TAG\" \\"
            echo "${COMMENT_PREFIX}  --template \"$template\""
            echo ""
          } >> "$SCRIPT_FILE"

          if [[ -z "$COMMENT_PREFIX" ]]; then
            add_eval "$REL_MERGED" 1
          else
            add_eval "$REL_MERGED" 2
          fi
        done
      done
    }

    for LR in "${learning_rates[@]}"; do
      merge_lora "$B_MODEL" B "$I_MODEL" I "$LR"          # B2I  — Shadow-FT
      merge_lora "$I_MODEL" I "$I_MODEL" I "$LR"          # I2I  — baseline
      merge_lora "$I_MODEL" I "$B_MODEL" B "$LR" "# "     # I2B  (commented out)
      merge_lora "$B_MODEL" B "$B_MODEL" B "$LR" "# "     # B2B  (commented out)
    done
  fi

  # --------------- Checkpoint merge helper ---------------
  {
    echo "###############################################################################"
    echo "##### Merge a specific checkpoint (run manually)                          #####"
    echo "###############################################################################"
    echo "# To merge an intermediate checkpoint, e.g. checkpoint-250 (~4k samples):"
    echo "#   python3 \$WORKSPACE_DIR/src/shadow/merge_lora.py \\"
    echo "#     --adapter_path \$RESULTS_DIR/<path>/checkpoint-250 \\"
    echo "#     --target_base <instruct_model_path> \\"
    echo "#     --merge_tag B2I-ckpt250 \\"
    echo "#     --template $template"
    echo ""
  } >> "$SCRIPT_FILE"

  # --------------- Eval list ---------------
  {
    echo "###############################################################################"
    echo "##### Evaluation list (model paths for OpenCompass)                       #####"
    echo "###############################################################################"
    echo ""
    for line in "${EVAL_LINES[@]}"; do
      echo "$line"
    done
    echo ""
  } >> "$SCRIPT_FILE"

  for line in "${EVAL_LINES[@]}"; do
    if [[ "$line" == "## "* ]]; then
      ALL_EVAL_BASE+=("$line")
    else
      ALL_EVAL_INSTRUCT+=("$line")
    fi
  done

  echo "INFO  Generated: $SCRIPT_FILE"
done

echo ""
echo "========================================"
echo "  OpenR1-Math-220k 200k script generation complete!"
echo "========================================"
echo ""
echo "Generated training scripts: $SCRIPT_OUTPUT_DIR/train_openr1_200k_*_${TIMESTAMP}.sh"
echo ""
echo "Checkpoints saved every $SAVE_STEPS steps (~2k samples)."
echo "Merge specific checkpoints on demand to save disk space."
echo ""
