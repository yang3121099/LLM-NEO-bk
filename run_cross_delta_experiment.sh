#!/usr/bin/env bash
###############################################################################
# run_cross_delta_experiment.sh
#
# Cross-checkpoint / Cross-model Delta Transfer Experiment
#
# Design:
#   - Train LoRA on 4 source models (Base / SFT / DPO / RLVR)
#   - Also train LoRA on 4 extra targets as direct-FT baselines
#   - Merge each source's delta onto 8 targets (shared-parameter-only)
#   - Evaluate to produce a 4×8 transfer matrix
#
# Output:
#   (1) Absolute score matrix:  A[s,t] = score(merge(source_s → target_t))
#   (2) Gain-over-direct-FT:   G[s,t] = A[s,t] - score(direct_FT(target_t))
#
# Usage:
#   bash run_cross_delta_experiment.sh
#   bash scripts/train_cross_delta_<TS>.sh
#   cd opencompass && python3 ./run.py ./eval_cross_delta.py
#   python3 visualize_transfer_matrix.py --results_dir <path>
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

###############################################################################
# Source models (train LoRA on these 4)
###############################################################################
SRC_NAMES=("Base"      "SFT"                              "DPO"                              "RLVR")
SRC_PATHS=("meta-llama/Llama-3.1-8B" \
           "allenai/Llama-3.1-Tulu-3-8B-SFT" \
           "allenai/Llama-3.1-Tulu-3-8B-DPO" \
           "allenai/Llama-3.1-Tulu-3-8B")

###############################################################################
# Target models (merge delta onto these 8, also direct-FT as baselines)
###############################################################################
TGT_NAMES=("Base"      "SFT"                              "DPO"                              "RLVR" \
           "Instruct"  "Tulu3.1"                          "Llama3-Inst"                      "R1-Distill")
TGT_PATHS=("meta-llama/Llama-3.1-8B" \
           "allenai/Llama-3.1-Tulu-3-8B-SFT" \
           "allenai/Llama-3.1-Tulu-3-8B-DPO" \
           "allenai/Llama-3.1-Tulu-3-8B" \
           "meta-llama/Llama-3.1-8B-Instruct" \
           "allenai/Llama-3.1-Tulu-3.1-8B" \
           "meta-llama/Meta-Llama-3-8B-Instruct" \
           "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

###############################################################################
# Training hyperparameters
###############################################################################
DATASET="Shadow_2k"
SUFFIX="shadow2k"
CUTOFF_LEN=16384
MAX_SAMPLES=2000
LORA_RANK=128
LR="2e-4"
PER_DEVICE_BS=1
GRAD_ACCUM=256
EPOCHS=1
TEMPLATE="llama3"

###############################################################################
# Generate experiment script
###############################################################################
TS=$(date +%m%d%H%M%S)
DAY=$(date +%m%d)
SCRIPT="$SCRIPT_OUTPUT_DIR/train_cross_delta_${TS}.sh"
REL="$DAY/cross-delta-$DAY"

LR_DEC=$(LC_NUMERIC=C printf "%.12f" "$LR" | sed -E 's/0+$//; s/\.$/.0/')

: > "$SCRIPT"

cat >> "$SCRIPT" << HEADER
#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated $(date '+%F %T') #####
# Cross-Delta Transfer Experiment
# Sources: ${SRC_NAMES[*]}
# Targets: ${TGT_NAMES[*]}
# Dataset: $DATASET ($MAX_SAMPLES samples), LoRA rank $LORA_RANK

WORKSPACE_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
RESULTS_DIR="\$WORKSPACE_DIR/results"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

HEADER

# --- Helper: generate training block ---
gen_train() {
  local NAME=$1 HF=$2 TAG=$3
  local OUTDIR="${REL}/${TAG}-${NAME}-lora-r${LORA_RANK}-${SUFFIX}"
  echo "$OUTDIR"  # return value via stdout (caller captures)

  {
    echo "###### Train LoRA: ${TAG}-${NAME} (${HF}) ######"
    echo "mkdir -p \"\$RESULTS_DIR/${OUTDIR}\""
    echo 'cd "$WORKSPACE_DIR"'
    echo "llamafactory-cli train \\"
    echo "  --model_name_or_path \"${HF}\" \\"
    echo "  --stage sft --do_train true \\"
    echo "  --finetuning_type lora --lora_rank ${LORA_RANK} \\"
    echo "  --dataset \"${DATASET}\" --template \"${TEMPLATE}\" \\"
    echo "  --cutoff_len ${CUTOFF_LEN} --max_samples ${MAX_SAMPLES} \\"
    echo "  --output_dir \"\$RESULTS_DIR/${OUTDIR}\" \\"
    echo "  --per_device_train_batch_size ${PER_DEVICE_BS} \\"
    echo "  --gradient_accumulation_steps ${GRAD_ACCUM} \\"
    echo "  --learning_rate ${LR_DEC} --num_train_epochs ${EPOCHS} \\"
    echo "  --logging_steps 1 --save_steps 1000 --save_only_model True \\"
    echo "  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \\"
    echo "  --bf16 true --val_size 0.01 \\"
    echo "  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \\"
    echo "  --trust_remote_code True --flash_attn fa2 \\"
    echo "  --overwrite_cache false --use_fast_tokenizer True"
    echo ""
  } >> "$SCRIPT"
}

###############################################################################
# Step 1: Train LoRA on 4 sources
###############################################################################
echo "###############################################################################" >> "$SCRIPT"
echo "##### Step 1: Train LoRA on 4 source models                              #####" >> "$SCRIPT"
echo "###############################################################################" >> "$SCRIPT"
echo "" >> "$SCRIPT"

declare -A SRC_ADAPTER_DIRS
for i in "${!SRC_NAMES[@]}"; do
  OUTDIR=$(gen_train "${SRC_NAMES[$i]}" "${SRC_PATHS[$i]}" "src")
  SRC_ADAPTER_DIRS[${SRC_NAMES[$i]}]="$OUTDIR"
done

###############################################################################
# Step 2: Train LoRA on extra targets (direct-FT baselines)
#         Skip targets that overlap with sources
###############################################################################
echo "###############################################################################" >> "$SCRIPT"
echo "##### Step 2: Direct-FT baselines on extra targets                       #####" >> "$SCRIPT"
echo "###############################################################################" >> "$SCRIPT"
echo "" >> "$SCRIPT"

declare -A TGT_BASELINE_DIRS
for i in "${!TGT_NAMES[@]}"; do
  NAME="${TGT_NAMES[$i]}"
  HF="${TGT_PATHS[$i]}"

  # Check if this target is already a source (skip double training)
  is_source=false
  for s in "${!SRC_NAMES[@]}"; do
    if [[ "${SRC_PATHS[$s]}" == "$HF" ]]; then
      # Reuse the source adapter as the direct-FT baseline
      TGT_BASELINE_DIRS[$NAME]="${SRC_ADAPTER_DIRS[${SRC_NAMES[$s]}]}"
      is_source=true
      break
    fi
  done

  if ! $is_source; then
    OUTDIR=$(gen_train "$NAME" "$HF" "tgt")
    TGT_BASELINE_DIRS[$NAME]="$OUTDIR"
  fi
done

###############################################################################
# Step 3: Cross-model delta merge (4 sources × 8 targets = 32 merges)
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 3: Cross-model delta merge (4×8 = 32 combinations)           #####"
  echo "###############################################################################"
  echo ""
  echo "# merge_lora.py applies the LoRA delta from source training onto target model."
  echo "# LoRA adapters only touch attention/MLP projections — embed_tokens and"
  echo "# lm_head are NOT modified. Shape mismatches are auto-skipped."
  echo ""
} >> "$SCRIPT"

# Collect all eval entries
declare -a EVAL_MERGED=()
declare -a EVAL_BASELINE=()

for si in "${!SRC_NAMES[@]}"; do
  SRC="${SRC_NAMES[$si]}"
  ADAPTER="${SRC_ADAPTER_DIRS[$SRC]}"

  for ti in "${!TGT_NAMES[@]}"; do
    TGT="${TGT_NAMES[$ti]}"
    TGT_HF="${TGT_PATHS[$ti]}"
    MERGE_TAG="${SRC}2${TGT}"
    REL_MERGED="${ADAPTER}/merged-${MERGE_TAG}"

    {
      echo "### Merge: ${MERGE_TAG} ###"
      echo "mkdir -p \"\$RESULTS_DIR/${REL_MERGED}\""
      echo "python3 \"\$WORKSPACE_DIR/src/shadow/merge_lora.py\" \\"
      echo "  --adapter_path \"\$RESULTS_DIR/${ADAPTER}\" \\"
      echo "  --target_base \"${TGT_HF}\" \\"
      echo "  --merge_tag \"${MERGE_TAG}\" \\"
      echo "  --template \"${TEMPLATE}\""
      echo ""
    } >> "$SCRIPT"

    EVAL_MERGED+=("    ('${MERGE_TAG}', '\$RESULTS_DIR/${REL_MERGED}'),")
  done
done

# Direct-FT baseline merges (each target with its own adapter merged onto itself)
for ti in "${!TGT_NAMES[@]}"; do
  TGT="${TGT_NAMES[$ti]}"
  TGT_HF="${TGT_PATHS[$ti]}"
  ADAPTER="${TGT_BASELINE_DIRS[$TGT]}"
  MERGE_TAG="DirectFT-${TGT}"
  REL_MERGED="${ADAPTER}/merged-${MERGE_TAG}"

  {
    echo "### Baseline: ${MERGE_TAG} ###"
    echo "mkdir -p \"\$RESULTS_DIR/${REL_MERGED}\""
    echo "python3 \"\$WORKSPACE_DIR/src/shadow/merge_lora.py\" \\"
    echo "  --adapter_path \"\$RESULTS_DIR/${ADAPTER}\" \\"
    echo "  --target_base \"${TGT_HF}\" \\"
    echo "  --merge_tag \"${MERGE_TAG}\" \\"
    echo "  --template \"${TEMPLATE}\""
    echo ""
  } >> "$SCRIPT"

  EVAL_BASELINE+=("    ('${MERGE_TAG}', '\$RESULTS_DIR/${REL_MERGED}'),")
done

###############################################################################
# Step 4: Weight similarity matrix
###############################################################################
{
  echo "###############################################################################"
  echo "##### Step 4: Weight similarity matrix (σ)                              #####"
  echo "###############################################################################"
  echo ""
  echo 'cd "$WORKSPACE_DIR"'
  echo "python3 weight_similarity_matrix.py --models \\"
} >> "$SCRIPT"
for ti in "${!TGT_NAMES[@]}"; do
  echo "  ${TGT_NAMES[$ti]}:${TGT_PATHS[$ti]} \\" >> "$SCRIPT"
done
{
  echo "  --output_dir \"\$RESULTS_DIR/${REL}/weight_similarity\""
  echo ""
} >> "$SCRIPT"

chmod +x "$SCRIPT"

###############################################################################
# Generate OpenCompass eval config
###############################################################################
EVAL_CFG="$WORKSPACE_DIR/opencompass/eval_cross_delta.py"

cat > "$EVAL_CFG" << 'PYHEAD'
# Auto-generated: Cross-Delta Transfer Experiment evaluation
# Usage: cd opencompass && python3 ./run.py ./eval_cross_delta.py
import os as _os
from mmengine.config import read_base

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
del _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 (MATH commented out) #########################
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

PYHEAD

{
  echo ""
  echo "work_dir = 'outputs/cross-delta-exp/'"
  echo ""

  # Original target models (no training)
  echo "# === Original target models (no training, reference only) ==="
  echo "original_targets = ["
  for ti in "${!TGT_NAMES[@]}"; do
    echo "    ('${TGT_NAMES[$ti]}-orig', '${TGT_PATHS[$ti]}'),"
  done
  echo "]"
  echo ""

  # Direct-FT baselines
  echo "# === Direct-FT baselines (each target trained on itself) ==="
  echo "direct_ft_baselines = ["
  for line in "${EVAL_BASELINE[@]}"; do
    echo "$line"
  done
  echo "]"
  echo ""

  # Cross-delta merged models (4×8)
  echo "# === Cross-delta merged models (source adapter → target model) ==="
  echo "cross_delta_models = ["
  for line in "${EVAL_MERGED[@]}"; do
    echo "$line"
  done
  echo "]"
  echo ""
} >> "$EVAL_CFG"

cat >> "$EVAL_CFG" << 'PYFOOT'
models = []

def _make_model(abbr, path):
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    return dict(
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

for abbr, path in original_targets:
    models.append(_make_model(abbr, path))

for abbr, path in direct_ft_baselines:
    models.append(_make_model(abbr, path))

for abbr, path in cross_delta_models:
    models.append(_make_model(abbr, path))
PYFOOT

###############################################################################
# Generate visualization helper config
###############################################################################
VIS_CFG="$RESULTS_DIR/${REL}/experiment_config.json"
mkdir -p "$(dirname "$VIS_CFG")"

python3 -c "
import json
cfg = {
    'sources': $(python3 -c "import json; print(json.dumps(dict(zip($(printf "'%s'," "${SRC_NAMES[@]}" | sed 's/,$//' | xargs -I{} echo "[{}]"), $(printf "'%s'," "${SRC_PATHS[@]}" | sed 's/,$//' | xargs -I{} echo "[{}]")))))" 2>/dev/null || echo '{}'),
    'targets': $(python3 -c "import json; print(json.dumps(dict(zip($(printf "'%s'," "${TGT_NAMES[@]}" | sed 's/,$//' | xargs -I{} echo "[{}]"), $(printf "'%s'," "${TGT_PATHS[@]}" | sed 's/,$//' | xargs -I{} echo "[{}]")))))" 2>/dev/null || echo '{}'),
}
json.dump(cfg, open('$VIS_CFG', 'w'), indent=2)
" 2>/dev/null || true

# Write config manually as fallback
cat > "$VIS_CFG" << VISCFG
{
  "sources": ["Base", "SFT", "DPO", "RLVR"],
  "source_paths": [
    "meta-llama/Llama-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "allenai/Llama-3.1-Tulu-3-8B"
  ],
  "targets": ["Base", "SFT", "DPO", "RLVR", "Instruct", "Tulu3.1", "Llama3-Inst", "R1-Distill"],
  "target_paths": [
    "meta-llama/Llama-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "allenai/Llama-3.1-Tulu-3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Llama-3.1-Tulu-3.1-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  ],
  "merge_tag_pattern": "{src}2{tgt}",
  "baseline_tag_pattern": "DirectFT-{tgt}"
}
VISCFG

echo ""
echo "========================================"
echo "  Generation complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Training + merge: $SCRIPT"
echo "  Eval config:      $EVAL_CFG"
echo "  Experiment config: $VIS_CFG"
echo ""
echo "Training breakdown:"
echo "  Sources (LoRA): ${#SRC_NAMES[@]} models"
EXTRA=$((${#TGT_NAMES[@]} - ${#SRC_NAMES[@]}))
echo "  Extra targets (direct-FT baselines): ${EXTRA} models"
echo "  Total training runs: $((${#SRC_NAMES[@]} + EXTRA))"
echo "  Total merges: $((${#SRC_NAMES[@]} * ${#TGT_NAMES[@]} + ${#TGT_NAMES[@]}))"
echo ""
echo "Workflow:"
echo "  1. bash $SCRIPT"
echo "  2. cd opencompass && python3 ./run.py ./eval_cross_delta.py"
echo "  3. python3 visualize_transfer_matrix.py --eval_dir opencompass/outputs/cross-delta-exp/"
