#!/usr/bin/env bash
###############################################################################
# run.sh — Shadow-FT pipeline script generator
#
# Usage:  bash run.sh
#
# This script auto-generates a training shell script under ./scripts/ that:
#   1) Trains LoRA on the Base model  (B)
#   2) Trains LoRA on the Instruct model (I)  — as the baseline
#   3) Merges the Base-trained LoRA delta onto Instruct  (B2I = Shadow-FT)
#   4) Merges the Instruct-trained LoRA onto Instruct    (I2I = ordinary SFT baseline)
#   5) Outputs an evaluation config (eval_generated.py) for OpenCompass
#
# After generation, run the training script, then launch evaluation.
###############################################################################
set -euo pipefail

###############################################################################
##### 0. Global Configuration                                              #####
###############################################################################
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# --- Training mode -----------------------------------------------------------
USE_LORA=true                   # true -> LoRA, false -> full SFT
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
DATASET="Shadow_2k"
SUFFIX_NAME="Shadow_2k"
CUTOFF_LEN=4096
SAMPLES=(2000)

# --- Training constants ------------------------------------------------------
LOGGING_STEPS=1
SAVE_STEPS=1000
PER_DEVICE_TRAIN_BS=16
GRAD_ACCUM_STEPS=16
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
VAL_SIZE=0.01
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

# Collect eval entries: add_eval <rel_path> [comment_level]
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
MODEL_DIR=""  # empty = use HuggingFace hub IDs directly
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"

# >>>  Edit this list to choose which models to train  <<<
BASE_MODELS=(
  "Qwen3.5-0.8B"
  "Qwen3.5-2B"
  "Qwen3.5-4B"
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
  SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${MODEL_BASE}_${TIMESTAMP}.sh"
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

  # Relative path prefix for this model's results (no leading $RESULTS_DIR)
  REL_ROOT="${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

  # --------------- Header ---------------
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo ""
    echo "##### Auto-generated $(date '+%F %T') #####"
    echo "# Model     : $MODEL_BASE"
    echo "# LoRA mode : $USE_LORA"
    echo "# Template  : $template"
    echo ""
    # Paths resolved at RUNTIME (where this script is located = scripts/)
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

  # --------------- Reset eval entries for this model ---------------
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
      # Relative path from $RESULTS_DIR
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

  # --------------- Step 1: Training ---------------
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "##### Step 1: Training (Base + Instruct with LoRA)                        #####" >> "$SCRIPT_FILE"
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "" >> "$SCRIPT_FILE"

  for LR in "${learning_rates[@]}"; do
    generate_train "$B_MODEL" B "$LR"
    generate_train "$I_MODEL" I "$LR"
  done

  # --------------- Step 2: Delta Merge ---------------
  echo "###############################################################################" >> "$SCRIPT_FILE"
  echo "##### Step 2: LoRA Delta Merge                                            #####" >> "$SCRIPT_FILE"
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

  else
    LR="${learning_rates[0]}"
    LR_DEC="$(to_decimal "$LR")"
    LR_TAG="$(lr_tag_dec "$LR")"
    K="$(format_k "${SAMPLES[0]}")"

    # B2I via apply_diff (Shadow-FT for full SFT)
    REL_B_DIR="${REL_ROOT}/B-${K}-sft-${LR_TAG}-${SUFFIX_NAME}"
    REL_MERGED="${REL_B_DIR}/merged-B2I"
    {
      echo "### SFT Merge: B2I (Shadow-FT) ###"
      echo "mkdir -p \"\$RESULTS_DIR/${REL_MERGED}\""
      echo "python3 \"\$WORKSPACE_DIR/src/shadow/apply_diff.py\" \\"
      echo "  --tuned_model \"\$RESULTS_DIR/${REL_B_DIR}\" \\"
      echo "  --target_model \"$I_MODEL\" \\"
      echo "  --base_model \"$B_MODEL\""
      echo ""
    } >> "$SCRIPT_FILE"
    add_eval "$REL_MERGED" 1

    REL_I_DIR="${REL_ROOT}/I-${K}-sft-${LR_TAG}-${SUFFIX_NAME}"
    add_eval "$REL_I_DIR" 1
  fi

  # --------------- Evaluation list ---------------
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

  # --------------- Collect for global eval config ---------------
  for line in "${EVAL_LINES[@]}"; do
    if [[ "$line" == "## "* ]]; then
      ALL_EVAL_BASE+=("$line")
    else
      ALL_EVAL_INSTRUCT+=("$line")
    fi
  done

  echo "INFO  Generated: $SCRIPT_FILE"
done

###############################################################################
##### 3. Generate OpenCompass evaluation config                            #####
###############################################################################
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_generated.py"
cat > "$EVAL_CONFIG" << 'EVAL_HEADER'
# Auto-generated evaluation config for Shadow-FT
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_generated.py -r <TIMESTAMP>

import os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR relative to this config file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'results')

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel

EVAL_HEADER

{
  echo "work_dir = 'outputs/shadow-ft-${TIMESTAMP}/'"
  echo ""
  echo "# ======= Instruct-type models (B2I Shadow-FT, I2I baseline) ======="
  echo "Baseline_settings = ["

  for line in "${ALL_EVAL_INSTRUCT[@]}"; do
    # Strip the leading "# " prefix, keep the tuple
    echo "    ${line#\# }"
  done

  echo "]"
  echo ""
  echo "# ======= Base-type models (B2B, I2B — usually commented out) ======="
  echo "BASE_settings = ["

  for line in "${ALL_EVAL_BASE[@]}"; do
    echo "    ${line#\#\# }"
  done

  echo "]"
  echo ""
} >> "$EVAL_CONFIG"

cat >> "$EVAL_CONFIG" << 'EVAL_FOOTER'
models = []

for abbr, path in Baseline_settings:
    # Resolve $RESULTS_DIR references
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1),
        )
    )

for abbr, path in BASE_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1),
        )
    )
EVAL_FOOTER

echo ""
echo "========================================"
echo "  Generation complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Training scripts: $SCRIPT_OUTPUT_DIR/train_*_${TIMESTAMP}.sh"
echo "  Eval config:      $EVAL_CONFIG"
echo ""
echo "Next steps:"
echo "  1. Run training:  bash $SCRIPT_OUTPUT_DIR/train_<MODEL>_${TIMESTAMP}.sh"
echo "  2. Run eval:      cd opencompass && python3 ./run.py ./eval_generated.py -r <TIMESTAMP>"
echo ""
