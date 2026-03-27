#!/usr/bin/env bash
###############################################################################
# run_weight_similarity_experiment.sh
#
# Weight Similarity × Delta Transferability Experiment
#
# Goal: Train LoRA on 9 Llama-family models using Shadow_2k, then perform
#       cross-model delta merging to build a transferability matrix.
#       Compare with weight similarity σ to validate Shadow-FT's boundary.
#
# Models:
#   Pipeline (Llama-3.1-8B based):
#     1. meta-llama/Llama-3.1-8B                     (Base3.1)
#     2. meta-llama/Llama-3.1-8B-Instruct             (Instruct3.1)
#     3. allenai/Llama-3.1-Tulu-3-8B-SFT              (Tulu3-SFT)
#     4. allenai/Llama-3.1-Tulu-3-8B-DPO              (Tulu3-DPO)
#     5. allenai/Llama-3.1-Tulu-3-8B                   (Tulu3-RLVR)
#   Independent (same architecture, different training):
#     6. allenai/Llama-3.1-Tulu-3.1-8B                (Tulu3.1)
#     7. meta-llama/Llama-3-8B                         (Base3)
#     8. meta-llama/Llama-3-8B-Instruct                (Instruct3)
#     9. deepseek-ai/DeepSeek-R1-Distill-Llama-8B     (R1-Distill)
#
# Usage:
#   bash run_weight_similarity_experiment.sh
#   # Then run the generated script:
#   bash scripts/train_weight_sim_<TIMESTAMP>.sh
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

###############################################################################
# Model definitions
###############################################################################

# Format: SHORT_NAME|HF_PATH|TEMPLATE|IS_CHAT_MODEL
# IS_CHAT_MODEL: 1 = instruct/chat model (can be eval target), 0 = base model
MODELS=(
  "Base3.1|meta-llama/Llama-3.1-8B|llama3|0"
  "Instruct3.1|meta-llama/Llama-3.1-8B-Instruct|llama3|1"
  "Tulu3-SFT|allenai/Llama-3.1-Tulu-3-8B-SFT|llama3|1"
  "Tulu3-DPO|allenai/Llama-3.1-Tulu-3-8B-DPO|llama3|1"
  "Tulu3-RLVR|allenai/Llama-3.1-Tulu-3-8B|llama3|1"
  "Tulu3.1|allenai/Llama-3.1-Tulu-3.1-8B|llama3|1"
  "Base3|meta-llama/Llama-3-8B|llama3|0"
  "Instruct3|meta-llama/Llama-3-8B-Instruct|llama3|1"
  "R1-Distill|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|llama3|1"
)

###############################################################################
# Training hyperparameters
###############################################################################
DATASET="Shadow_2k"
SUFFIX_NAME="shadow2k"
CUTOFF_LEN=16384
MAX_SAMPLES=2000
LORA_RANK=128
LEARNING_RATE="2e-4"
PER_DEVICE_TRAIN_BS=1
GRAD_ACCUM_STEPS=256
NUM_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
BF16=true
VAL_SIZE=0.01
LOGGING_STEPS=1
SAVE_STEPS=1000

###############################################################################
# Generate the experiment script
###############################################################################
TIMESTAMP=$(date +%m%d%H%M%S)
MONTHDAY=$(date +%m%d)
EXP_TAG="weight_sim"
SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${EXP_TAG}_${TIMESTAMP}.sh"
REL_ROOT="${MONTHDAY}/result-weight-similarity-${MONTHDAY}"

: > "$SCRIPT_FILE"

# --- Header ---
cat >> "$SCRIPT_FILE" << HEADER
#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated $(date '+%F %T') #####
# Experiment: Weight Similarity × Delta Transferability
# Dataset: $DATASET ($MAX_SAMPLES samples)
# LoRA rank: $LORA_RANK, LR: $LEARNING_RATE

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

# --- Helper: format learning rate ---
LR_DEC=$(LC_NUMERIC=C printf "%.12f" "$LEARNING_RATE" | sed -E 's/0+$//; s/\.$/.0/')
LR_TAG="lr${LR_DEC}"

###############################################################################
# Step 1: Train LoRA on each model
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 1: Train LoRA on each model using ${DATASET}                   #####"
  echo "###############################################################################"
  echo ""
} >> "$SCRIPT_FILE"

# Collect adapter output dirs for later merging
declare -A ADAPTER_DIRS

for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"

  DIR_NAME="${NAME}-2k-lora-rank${LORA_RANK}-${LR_TAG}-${SUFFIX_NAME}"
  REL_OUTDIR="${REL_ROOT}/${DIR_NAME}"
  ADAPTER_DIRS[$NAME]="$REL_OUTDIR"

  {
    echo "###### Train LoRA on: ${NAME} (${HF_PATH}) ######"
    echo "mkdir -p \"\$RESULTS_DIR/${REL_OUTDIR}\""
    echo 'cd "$WORKSPACE_DIR"'
    echo "llamafactory-cli train \\"
    echo "  --model_name_or_path \"${HF_PATH}\" \\"
    echo "  --stage sft \\"
    echo "  --do_train true \\"
    echo "  --finetuning_type lora --lora_rank ${LORA_RANK} \\"
    echo "  --dataset \"${DATASET}\" \\"
    echo "  --template \"${TEMPLATE}\" \\"
    echo "  --cutoff_len ${CUTOFF_LEN} \\"
    echo "  --max_samples ${MAX_SAMPLES} \\"
    echo "  --output_dir \"\$RESULTS_DIR/${REL_OUTDIR}\" \\"
    echo "  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS} \\"
    echo "  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \\"
    echo "  --learning_rate ${LR_DEC} \\"
    echo "  --num_train_epochs ${NUM_EPOCHS} \\"
    echo "  --logging_steps ${LOGGING_STEPS} \\"
    echo "  --save_steps ${SAVE_STEPS} \\"
    echo "  --save_only_model True \\"
    echo "  --plot_loss true \\"
    echo "  --lr_scheduler_type ${LR_SCHEDULER} \\"
    echo "  --warmup_ratio ${WARMUP_RATIO} \\"
    echo "  --bf16 ${BF16} \\"
    echo "  --val_size ${VAL_SIZE} \\"
    echo "  --per_device_eval_batch_size 1 \\"
    echo "  --eval_strategy steps \\"
    echo "  --eval_steps 10000 \\"
    echo "  --trust_remote_code True \\"
    echo "  --flash_attn fa2 \\"
    echo "  --overwrite_cache false \\"
    echo "  --use_fast_tokenizer True"
    echo ""
  } >> "$SCRIPT_FILE"
done

###############################################################################
# Step 2: Cross-model LoRA delta merge
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 2: Cross-model LoRA delta merge (Shadow-FT transfer matrix)    #####"
  echo "###############################################################################"
  echo ""
  echo "# For each (source, target) pair, merge the LoRA adapter trained on SOURCE"
  echo "# onto the TARGET model. This tests delta transferability across models."
  echo "# Pairs are organized by experimental interest:"
  echo "#   - Pipeline transfers (Base3.1 ↔ SFT ↔ DPO ↔ RLVR ↔ Instruct3.1)"
  echo "#   - Cross-family transfers (Base3.1-trained → R1-Distill, etc.)"
  echo ""
} >> "$SCRIPT_FILE"

# Collect eval entries
EVAL_ENTRIES=()

# --- Generate merge commands ---
# We generate ALL pairwise merges, but comment out less important ones.
# Key merges (uncommented):
#   - Each pipeline model's adapter → Instruct3.1 (main experiment)
#   - Base3.1 adapter → each chat model
#   - Cross-family: Base3 adapter → Instruct3, R1-Distill
#   - Self-merge baselines: adapter → own model (I2I style)

# Define key merge pairs: SOURCE_ADAPTER → TARGET_MODEL
# Format: SRC_NAME|TGT_NAME|COMMENT_PREFIX
# Empty COMMENT_PREFIX = uncommented (will run)

MERGE_PAIRS=(
  # === Pipeline → Instruct3.1 (Shadow-FT core experiment) ===
  "Base3.1|Instruct3.1|"      # B2I — Shadow-FT main
  "Tulu3-SFT|Instruct3.1|"    # SFT2I
  "Tulu3-DPO|Instruct3.1|"    # DPO2I
  "Tulu3-RLVR|Instruct3.1|"   # RLVR2I

  # === Self-merge baselines (I2I) ===
  "Instruct3.1|Instruct3.1|"  # I2I baseline
  "Tulu3-SFT|Tulu3-SFT|"      # SFT2SFT
  "Tulu3-DPO|Tulu3-DPO|"      # DPO2DPO
  "Tulu3-RLVR|Tulu3-RLVR|"    # RLVR2RLVR

  # === Base3.1 adapter → all chat models ===
  "Base3.1|Tulu3-SFT|"        # B2SFT
  "Base3.1|Tulu3-DPO|"        # B2DPO
  "Base3.1|Tulu3-RLVR|"       # B2RLVR
  "Base3.1|Tulu3.1|"           # B2Tulu3.1

  # === Cross-family transfers ===
  "Base3|Instruct3|"           # B3→I3
  "Base3|R1-Distill|"          # B3→R1
  "Base3.1|R1-Distill|"        # B3.1→R1

  # === Independent model baselines ===
  "Instruct3|Instruct3|"      # I3→I3 baseline
  "R1-Distill|R1-Distill|"    # R1→R1 baseline
  "Tulu3.1|Tulu3.1|"           # T3.1→T3.1 baseline

  # === More cross-family (commented out, run if needed) ===
  "Tulu3-SFT|Tulu3-DPO|# "    # SFT→DPO
  "Tulu3-DPO|Tulu3-SFT|# "    # DPO→SFT
  "Tulu3-RLVR|Tulu3-DPO|# "   # RLVR→DPO
  "Instruct3.1|Base3.1|# "     # I2B (reverse)
  "Instruct3|Base3|# "         # I3→B3 (reverse)
  "R1-Distill|Instruct3.1|# " # R1→I3.1
  "R1-Distill|Base3.1|# "      # R1→B3.1
  "Tulu3.1|Instruct3.1|# "    # T3.1→I3.1
  "Base3.1|Instruct3|# "       # B3.1→I3 (cross-version)
  "Base3|Instruct3.1|# "       # B3→I3.1 (cross-version)
)

# Build lookup: NAME → HF_PATH
declare -A MODEL_PATHS
for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"
  MODEL_PATHS[$NAME]="$HF_PATH"
done

for merge_entry in "${MERGE_PAIRS[@]}"; do
  IFS='|' read -r SRC TGT COMMENT <<< "$merge_entry"

  SRC_ADAPTER="${ADAPTER_DIRS[$SRC]}"
  TGT_PATH="${MODEL_PATHS[$TGT]}"
  MERGE_TAG="${SRC}2${TGT}"
  REL_MERGED="${SRC_ADAPTER}/merged-${MERGE_TAG}"

  {
    echo "### Merge: ${MERGE_TAG} (adapter=${SRC}, target=${TGT}) ###"
    echo "${COMMENT}mkdir -p \"\$RESULTS_DIR/${REL_MERGED}\""
    echo "${COMMENT}python3 \"\$WORKSPACE_DIR/src/shadow/merge_lora.py\" \\"
    echo "${COMMENT}  --adapter_path \"\$RESULTS_DIR/${SRC_ADAPTER}\" \\"
    echo "${COMMENT}  --target_base \"${TGT_PATH}\" \\"
    echo "${COMMENT}  --merge_tag \"${MERGE_TAG}\" \\"
    echo "${COMMENT}  --template \"llama3\""
    echo ""
  } >> "$SCRIPT_FILE"

  # Add to eval list
  if [[ -z "$COMMENT" ]]; then
    EVAL_ENTRIES+=("('${MERGE_TAG}','\$RESULTS_DIR/${REL_MERGED}'),")
  else
    EVAL_ENTRIES+=("# ('${MERGE_TAG}','\$RESULTS_DIR/${REL_MERGED}'),")
  fi
done

###############################################################################
# Step 3: Weight Similarity Matrix
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 3: Compute pairwise weight similarity matrix                   #####"
  echo "###############################################################################"
  echo ""
  echo "# Run weight similarity analysis across all models"
  echo "# This generates the σ matrix for comparison with eval performance"
  echo 'cd "$WORKSPACE_DIR"'
  echo "python3 weight_similarity_matrix.py --models \\"
} >> "$SCRIPT_FILE"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"
  echo "  ${NAME}:${HF_PATH} \\" >> "$SCRIPT_FILE"
done

{
  echo "  --output_dir \"\$RESULTS_DIR/${REL_ROOT}/weight_similarity\""
  echo ""
} >> "$SCRIPT_FILE"

###############################################################################
# Step 4: Evaluation entries
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 4: Evaluation — model paths for OpenCompass                    #####"
  echo "###############################################################################"
  echo ""
  echo "# Copy the entries below into eval_weight_similarity.py"
  echo "# or use the auto-generated config directly."
  echo ""

  # Original models (baselines without training)
  echo "# === Original models (no training) ==="
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"
    if [[ "$IS_CHAT" == "1" ]]; then
      echo "# ('${NAME}-orig', '${HF_PATH}'),"
    fi
  done
  echo ""

  echo "# === Merged models ==="
  for line in "${EVAL_ENTRIES[@]}"; do
    echo "$line"
  done
  echo ""
} >> "$SCRIPT_FILE"

chmod +x "$SCRIPT_FILE"

###############################################################################
# Generate OpenCompass evaluation config
###############################################################################
EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_weight_similarity.py"

cat > "$EVAL_CONFIG" << 'PYHEADER'
# Auto-generated evaluation config for Weight Similarity Experiment
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_weight_similarity.py -r <TIMESTAMP>

import os as _os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
del _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 Benchmarks #########################
    # Full MATH benchmark (commented out per request)
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets

    # MATH-500 (subset)
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

    # Minerva MATH
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets

    # GSM8K (8-shot)
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

    # GSM8K (0-shot)
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets

    # AIME 2024
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

    # SVAMP
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate
PYHEADER

# Write the model lists
{
  echo ""
  echo "work_dir = 'outputs/weight-similarity-exp/'"
  echo ""

  # --- Original baselines (no training) ---
  echo "# ======= Original models (baselines without LoRA training) ======="
  echo "original_baselines = ["
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"
    if [[ "$IS_CHAT" == "1" ]]; then
      echo "    ('${NAME}-orig', '${HF_PATH}'),"
    fi
  done
  echo "]"
  echo ""

  # --- Merged models ---
  echo "# ======= Merged models (LoRA delta transfers) ======="
  echo "merged_models = ["
  for line in "${EVAL_ENTRIES[@]}"; do
    echo "    ${line}"
  done
  echo "]"
  echo ""
} >> "$EVAL_CONFIG"

cat >> "$EVAL_CONFIG" << 'PYFOOTER'
models = []

# Original baselines
for abbr, path in original_baselines:
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

# Merged models
for abbr, path in merged_models:
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
PYFOOTER

echo ""
echo "========================================"
echo "  Generation complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Training + merge script: $SCRIPT_FILE"
echo "  Eval config:             $EVAL_CONFIG"
echo ""
echo "Experiment workflow:"
echo "  1. Run training + merge:  bash $SCRIPT_FILE"
echo "  2. Run evaluation:        cd opencompass && python3 ./run.py ./eval_weight_similarity.py"
echo "  3. Analyze results:       python3 weight_similarity_matrix.py --models ..."
echo ""
echo "Models (${#MODELS[@]}):"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME HF_PATH TEMPLATE IS_CHAT <<< "$entry"
  echo "  $NAME -> $HF_PATH"
done
echo ""
echo "Merge pairs: ${#MERGE_PAIRS[@]} total (uncommented: $(grep -c '^[^#]' <<< "$(printf '%s\n' "${EVAL_ENTRIES[@]}")") for eval)"
