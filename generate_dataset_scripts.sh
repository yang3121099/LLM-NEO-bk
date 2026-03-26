#!/usr/bin/env bash
###############################################################################
# generate_dataset_scripts.sh — Generate Shadow-FT training + eval scripts
#
# Features:
#   - Auto-detect GPU count: bs×acc×gpus = 256
#   - Full-scale experiments: OpenR1-Math-220k, DeepMath-103K
#   - Checkpoint strategy: save every N steps, merge every 10 checkpoints
#   - Timestamped eval configs matching each training experiment
#   - Eval: all math + winogrande + ARC-c (no code benchmarks)
#
# Usage:  bash generate_dataset_scripts.sh
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

# --- Training constants (effective BS = 256, auto-split across GPUs) ---------
USE_LORA=true
LORA_RANK=128
LR=2e-4
EFFECTIVE_BS=256          # invariant: bs × acc × gpus = 256
DEFAULT_PER_GPU_BS=2      # preferred per-GPU batch size (8-gpu scenario)
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
LOGGING_STEPS=1
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

# --- Dataset definitions: name | suffix | samples | save_steps ---------------
#     save_steps is approximate; the script also merges every 10×save_steps
EXPERIMENTS=(
  # --- 2k samples (quick experiments) ---
  "opus_reasoning_3k|opus3k|2000|1000"
  "openr1_math_220k|openr1|2000|1000"
  "dolci_instruct_mix|dolci_mix|2000|1000"
  "nemotron_if_chat_v1|nemotron_if|2000|1000"
  "deepmath_2k_demo|deepmath_demo|2000|1000"
  # --- Full-scale experiments ---
  "openr1_math_220k|openr1|220000|100"
  "deepmath_103k|deepmath|309000|100"
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

# Collect eval entries for 2k experiments
ALL_EVAL_2K_ENTRIES=()
# Collect full-scale experiments for separate eval configs
ALL_EVAL_FULL_ENTRIES=()

for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r DATASET SUFFIX MAX_SAMPLES EXP_SAVE_STEPS <<< "$EXP"
  K="$(format_k "$MAX_SAMPLES")"
  IS_FULL=false
  [[ "$MAX_SAMPLES" -gt 10000 ]] && IS_FULL=true
  # Merge interval: every 10 checkpoints
  MERGE_INTERVAL=$(( EXP_SAVE_STEPS * 10 ))

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
# Effective BS = $EFFECTIVE_BS (auto-detected: bs × acc × gpus)

##### Paths #####
WORKSPACE_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
RESULTS_DIR="\$WORKSPACE_DIR/results"

##### GPU auto-detection #####
NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "\$NUM_GPUS" -eq 0 ]]; then
    echo "WARNING: No GPU detected, defaulting to 1"
    NUM_GPUS=1
fi
echo "INFO: Detected \$NUM_GPUS GPU(s)"

# Compute per-GPU BS and gradient accumulation so that bs*acc*gpus = $EFFECTIVE_BS
PER_GPU_BS=$DEFAULT_PER_GPU_BS
# Ensure effective batch size is exactly $EFFECTIVE_BS
GRAD_ACCUM=\$(( $EFFECTIVE_BS / (PER_GPU_BS * NUM_GPUS) ))
if [[ "\$GRAD_ACCUM" -lt 1 ]]; then
    # Too many GPUs for this per-GPU BS, reduce BS
    PER_GPU_BS=1
    GRAD_ACCUM=\$(( $EFFECTIVE_BS / NUM_GPUS ))
fi
echo "INFO: PER_GPU_BS=\$PER_GPU_BS  GRAD_ACCUM=\$GRAD_ACCUM  NUM_GPUS=\$NUM_GPUS  (effective=$EFFECTIVE_BS)"

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

# Use torchrun for multi-GPU, llamafactory-cli for single-GPU
if [[ "\$NUM_GPUS" -gt 1 ]]; then
    LAUNCH_CMD="torchrun --nproc_per_node=\$NUM_GPUS --master_port=\$(( RANDOM % 10000 + 20000 )) -m llamafactory.train"
else
    LAUNCH_CMD="llamafactory-cli train"
fi

\$LAUNCH_CMD \\
  --model_name_or_path "${M_PATH}" \\
  --stage sft \\
  --do_train true \\
  --finetuning_type lora --lora_rank ${LORA_RANK} \\
  --dataset "${DATASET}" \\
  --template "${template}" \\
  --cutoff_len ${CUTOFF_LEN} \\
  --max_samples ${MAX_SAMPLES} \\
  --output_dir "\$RESULTS_DIR/${REL_OUTDIR}" \\
  --per_device_train_batch_size \$PER_GPU_BS \\
  --gradient_accumulation_steps \$GRAD_ACCUM \\
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
    # For full-scale: merge every MERGE_INTERVAL steps (every 10 ckpts)
    # For small: merge final checkpoint only
    if [[ "$IS_FULL" == "true" ]]; then
      cat >> "$SCRIPT_FILE" << 'MERGE_HEADER'
###############################################################################
##### Merge checkpoints (every 10 save-points to save storage) #####
###############################################################################
MERGE_HEADER

      for TAG_INFO in "B|$B_MODEL|$I_MODEL" "I|$I_MODEL|$I_MODEL"; do
        IFS='|' read -r SRC_TAG SRC_MODEL TGT_MODEL <<< "$TAG_INFO"
        SRC_DIR="${SRC_TAG}-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}"
        REL_ADAP="${REL_ROOT}/${SRC_DIR}"
        if [[ "$SRC_TAG" == "B" ]]; then MERGE_TAG="B2I"; else MERGE_TAG="I2I"; fi

        cat >> "$SCRIPT_FILE" << MERGE_LOOP
echo "=== Merging ${MERGE_TAG} checkpoints from ${SRC_DIR} ==="
ADAPTER_BASE="\$RESULTS_DIR/${REL_ADAP}"
CKPTS=(\$(ls -d "\$ADAPTER_BASE/checkpoint-"* 2>/dev/null | sort -t- -k2 -n))
TOTAL_CKPTS=\${#CKPTS[@]}
echo "Found \$TOTAL_CKPTS checkpoints"

# Merge every ${MERGE_INTERVAL} steps (= every 10 checkpoints)
for (( ci=0; ci<TOTAL_CKPTS; ci++ )); do
    CKPT="\${CKPTS[\$ci]}"
    STEP=\$(basename "\$CKPT" | sed 's/checkpoint-//')
    # Merge at every 10th checkpoint, or the very last one
    if (( (ci + 1) % 10 == 0 )) || (( ci == TOTAL_CKPTS - 1 )); then
        echo "  Merging step \$STEP (${MERGE_TAG}) ..."
        python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
            --adapter_path "\$CKPT" \\
            --target_base "${TGT_MODEL}" \\
            --merge_tag "${MERGE_TAG}" \\
            --template "${template}"
    fi
done

# Also merge from the main output dir (final adapter)
echo "  Merging final adapter (${MERGE_TAG}) ..."
python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
    --adapter_path "\$ADAPTER_BASE" \\
    --target_base "${TGT_MODEL}" \\
    --merge_tag "${MERGE_TAG}" \\
    --template "${template}"

MERGE_LOOP
      done
    else
      # Small experiments: merge final only
      for MERGE in "B|I|" "I|I|" "I|B|# " "B|B|# "; do
        IFS='|' read -r SRC_TAG TGT_TAG PREFIX <<< "$MERGE"
        SRC_DIR="${SRC_TAG}-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}"
        REL_ADAP="${REL_ROOT}/${SRC_DIR}"
        MERGE_TAG="${SRC_TAG}2${TGT_TAG}"
        if [[ "$TGT_TAG" == "I" ]]; then TGT_MODEL="$I_MODEL"; else TGT_MODEL="$B_MODEL"; fi

        cat >> "$SCRIPT_FILE" << MERGE_CMD
### Merge: ${MERGE_TAG} (adapter=${SRC_TAG}, target=${TGT_TAG}) ###
${PREFIX}python3 "\$WORKSPACE_DIR/src/shadow/merge_lora.py" \\
${PREFIX}  --adapter_path "\$RESULTS_DIR/${REL_ADAP}" \\
${PREFIX}  --target_base "${TGT_MODEL}" \\
${PREFIX}  --merge_tag "${MERGE_TAG}" \\
${PREFIX}  --template "${template}"

MERGE_CMD
      done
    fi

    # --- Collect eval entries ---
    B2I_REL="${REL_ROOT}/B-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}/merged-B2I"
    I2I_REL="${REL_ROOT}/I-${K}-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX}/merged-I2I"
    B2I_SHORT="${MODEL_BASE}-${SUFFIX}-${K}-B2I"
    I2I_SHORT="${MODEL_BASE}-${SUFFIX}-${K}-I2I"

    if [[ "$IS_FULL" == "true" ]]; then
      ALL_EVAL_FULL_ENTRIES+=("    ('${B2I_SHORT}','\$RESULTS_DIR/${B2I_REL}'),")
      ALL_EVAL_FULL_ENTRIES+=("    ('${I2I_SHORT}','\$RESULTS_DIR/${I2I_REL}'),")
    else
      ALL_EVAL_2K_ENTRIES+=("    ('${B2I_SHORT}','\$RESULTS_DIR/${B2I_REL}'),")
      ALL_EVAL_2K_ENTRIES+=("    ('${I2I_SHORT}','\$RESULTS_DIR/${I2I_REL}'),")
    fi

    chmod +x "$SCRIPT_FILE"
    echo "INFO  Generated: $SCRIPT_FILE"
  done
done

###############################################################################
##### Generate OpenCompass evaluation configs (one per experiment group)    #####
###############################################################################

generate_eval_config() {
  local CONFIG_FILE="$1"
  local RUN_TAG="$2"
  shift 2
  local -a EVAL_ENTRIES=("$@")

  cat > "$CONFIG_FILE" << 'EVAL_HEADER'
# Auto-generated evaluation config for Shadow-FT
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_generated.py -r <TIMESTAMP>

import os as _os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR relative to this config file
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')

# Auto-detect GPU count for evaluation
_NUM_GPUS = 1
try:
    import subprocess as _sp
    _NUM_GPUS = int(_sp.check_output(
        ['nvidia-smi', '-L'], text=True).strip().count('\n')) + 1
    _NUM_GPUS = max(1, len(
        _sp.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()))
except Exception:
    _NUM_GPUS = 1
del _os, _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

    ################## General (winogrande + ARC-c) ##################
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import winogrande_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

    ################## Instruction Following ##################
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    ##################### Tool Use (T-Eval review only) ####################
    from opencompass.configs.datasets.teval.teval_en_gen_slim import teval_slim_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

EVAL_HEADER

  {
    echo "work_dir = 'outputs/Rebuttal-0729/shadow-${RUN_TAG}/'"
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

    # --- Trained models ---
    echo "# ======= Trained models (B2I Shadow-FT, I2I baseline) ======="
    echo "Baseline_settings = ["
    for entry in "${EVAL_ENTRIES[@]}"; do
      echo "$entry"
    done
    echo "]"
    echo ""
  } >> "$CONFIG_FILE"

  cat >> "$CONFIG_FILE" << 'EVAL_FOOTER'
models = []

# Original Instruct baselines
for abbr, path in HF_baselines:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
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
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )
EVAL_FOOTER
}

# --- Generate 2k eval config ---
EVAL_2K_CONFIG="$WORKSPACE_DIR/opencompass/eval_2k_${TIMESTAMP}.py"
generate_eval_config "$EVAL_2K_CONFIG" "2k-${TIMESTAMP}" "${ALL_EVAL_2K_ENTRIES[@]}"
echo "INFO  Generated 2k eval config: $EVAL_2K_CONFIG"

# --- Generate full-scale eval config ---
if [[ ${#ALL_EVAL_FULL_ENTRIES[@]} -gt 0 ]]; then
  EVAL_FULL_CONFIG="$WORKSPACE_DIR/opencompass/eval_full_${TIMESTAMP}.py"
  generate_eval_config "$EVAL_FULL_CONFIG" "full-${TIMESTAMP}" "${ALL_EVAL_FULL_ENTRIES[@]}"
  echo "INFO  Generated full eval config: $EVAL_FULL_CONFIG"
fi

# --- Also update the default eval_generated.py (for backward compatibility) ---
cp "$EVAL_2K_CONFIG" "$WORKSPACE_DIR/opencompass/eval_generated.py"

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
echo "=== Full-scale experiments (train + selective merge) ==="
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
echo "  # 2k experiments:"
echo "  cd opencompass && python3 ./run.py ./eval_2k_${TIMESTAMP}.py -r ${TIMESTAMP}"
if [[ ${#ALL_EVAL_FULL_ENTRIES[@]} -gt 0 ]]; then
  echo "  # Full-scale experiments:"
  echo "  cd opencompass && python3 ./run.py ./eval_full_${TIMESTAMP}.py -r ${TIMESTAMP}"
fi
echo ""
echo "=== Prerequisites ==="
echo "  # Download eval data (SVAMP, NLTK punkt_tab):"
echo "  bash scripts/download_eval_data.sh"
echo "  # Prepare DeepMath-103K for SFT (expands 3 R1 solutions per question):"
echo "  python3 scripts/prepare_deepmath_103k.py"
echo ""
