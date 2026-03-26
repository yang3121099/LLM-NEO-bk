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
  # --- 2k samples for all 4 datasets ---
  "opus_reasoning_3k|opus3k|2000|1000"
  "openr1_math_220k|openr1|2000|1000"
  "dolci_instruct_mix|dolci_mix|2000|1000"
  "nemotron_if_chat_v1|nemotron_if|2000|1000"
  # --- 200k samples for OpenR1-Math (save ckpt every ~2k samples) ---
  # 2000 / (bs16 * accum16) = ~8 steps per 2k samples
  "openr1_math_220k|openr1|200000|8"
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

# Collect eval entries for 2k experiments only
ALL_EVAL_ENTRIES=()

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

    # Collect eval entries for 2k experiments (not 200k)
    if [[ "$MAX_SAMPLES" -le 10000 ]]; then
      B2I_REL="${REL_ROOT}/B-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}/merged-B2I"
      I2I_REL="${REL_ROOT}/I-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}/merged-I2I"
      B2I_SHORT="${MODEL_BASE}-${SUFFIX}-${K}-B2I"
      I2I_SHORT="${MODEL_BASE}-${SUFFIX}-${K}-I2I"
      ALL_EVAL_ENTRIES+=("    ('${B2I_SHORT}','\$RESULTS_DIR/${B2I_REL}'),")
      ALL_EVAL_ENTRIES+=("    ('${I2I_SHORT}','\$RESULTS_DIR/${I2I_REL}'),")
    fi

    chmod +x "$SCRIPT_FILE"
    echo "INFO  Generated: $SCRIPT_FILE"
  done
done

###############################################################################
##### Generate OpenCompass evaluation config                               #####
###############################################################################
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_generated.py"
cat > "$EVAL_CONFIG" << 'EVAL_HEADER'
# Auto-generated evaluation config for Shadow-FT
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_generated.py -r 20250727200010

import os as _os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR relative to this config file
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
del _os, _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math #########################
    # from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    # from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    # from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets

    ################## Instruction Following ##################
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    ##################### Tool Use (T-Eval) ####################
    from opencompass.configs.datasets.teval.teval_en_gen import teval_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

EVAL_HEADER

{
  echo "work_dir = 'outputs/Rebuttal-0729/shadow-example/'"
  echo ""

  # --- Original Instruct baselines ---
  echo "# ======= Original Instruct baselines (lmdeploy) ======="
  echo "HF_baselines = ["
  for i in "${!MODEL_PAIRS[@]}"; do
    PAIR="${MODEL_PAIRS[$i]}"
    I_MODEL="${PAIR##*||}"
    I_NAME=$(basename "$I_MODEL")
    echo "    ('${I_NAME}-Instruct-hf', '${I_MODEL}'),"
  done
  echo "]"
  echo ""

  # --- Trained models (B2I + I2I from 2k experiments) ---
  echo "# ======= Trained models (B2I Shadow-FT, I2I baseline) ======="
  echo "Baseline_settings = ["
  for entry in "${ALL_EVAL_ENTRIES[@]}"; do
    echo "$entry"
  done
  echo "]"
  echo ""
} >> "$EVAL_CONFIG"

cat >> "$EVAL_CONFIG" << 'EVAL_FOOTER'
models = []

# Original Instruct baselines
for abbr, path in HF_baselines:
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

for abbr, path in Baseline_settings:
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
EVAL_FOOTER

echo "INFO  Generated eval config: $EVAL_CONFIG"

echo ""
echo "========================================"
echo "  Script generation complete!"
echo "========================================"
echo ""
echo "Generated scripts in: $SCRIPT_OUTPUT_DIR/"
echo ""
echo "=== 2k experiments (train + merge + eval) ==="
for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r DATASET SUFFIX MAX_SAMPLES _ <<< "$EXP"
  [[ "$MAX_SAMPLES" -gt 10000 ]] && continue
  K="$(format_k "$MAX_SAMPLES")"
  for M in "${MODEL_NAMES[@]}"; do
    echo "  bash scripts/train_${SUFFIX}_${K}_${M}_${TIMESTAMP}.sh"
  done
done
echo ""
echo "=== 200k experiments (train only, merge on demand) ==="
for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r DATASET SUFFIX MAX_SAMPLES _ <<< "$EXP"
  [[ "$MAX_SAMPLES" -le 10000 ]] && continue
  K="$(format_k "$MAX_SAMPLES")"
  for M in "${MODEL_NAMES[@]}"; do
    echo "  bash scripts/train_${SUFFIX}_${K}_${M}_${TIMESTAMP}.sh"
  done
done
echo ""
echo "=== Evaluation ==="
echo "  cd opencompass && python3 ./run.py ./eval_generated.py -r 20250727200010"
echo ""
