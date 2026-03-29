#!/usr/bin/env bash
###############################################################################
# train_missing_and_merge_0327.sh
#
# 1. Train the 2 missing adapters: Instruct3 and R1-Distill
# 2. Run ALL cross-delta merges (4 sources × 8 targets = 32)
# 3. Run ALL direct-FT baseline merges (8 targets)
# 4. Compute weight similarity matrix
###############################################################################
set -uo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
ADAPTER_ROOT="$RESULTS_DIR/0327/result-weight-similarity-0327"
ADAPTER_SUFFIX="2k-lora-rank128-lr0.0002-shadow2k"

cd "$WORKSPACE_DIR"

###############################################################################
# Step 0: Train missing adapters
###############################################################################
echo "###############################################################################"
echo "##### Step 0: Train missing adapters (Instruct3, R1-Distill)              #####"
echo "###############################################################################"
echo ""

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

train_if_missing() {
  local NAME="$1" HF_PATH="$2"
  local OUTDIR="$ADAPTER_ROOT/${NAME}-${ADAPTER_SUFFIX}"

  if [[ -d "$OUTDIR" ]] && compgen -G "$OUTDIR/adapter_model*" >/dev/null 2>&1; then
    echo "[skip] $NAME already trained: $OUTDIR"
    return 0
  fi

  echo "[train] $NAME ($HF_PATH)"
  mkdir -p "$OUTDIR"
  llamafactory-cli train \
    --model_name_or_path "$HF_PATH" \
    --stage sft --do_train true \
    --finetuning_type lora --lora_rank 128 \
    --dataset "Shadow_2k" --template "llama3" \
    --cutoff_len 16384 --max_samples 2000 \
    --output_dir "$OUTDIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 0.0002 --num_train_epochs 1 \
    --logging_steps 1 --save_steps 1000 --save_only_model True \
    --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
    --bf16 true --val_size 0.01 \
    --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
    --trust_remote_code True --flash_attn fa2 \
    --overwrite_cache false --use_fast_tokenizer True
}

train_if_missing "Instruct3" "meta-llama/Meta-Llama-3-8B-Instruct"
train_if_missing "R1-Distill" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

echo ""

###############################################################################
# Full model registry
###############################################################################
declare -A HF=(
  ["Base3.1"]="meta-llama/Llama-3.1-8B"
  ["Instruct3.1"]="meta-llama/Llama-3.1-8B-Instruct"
  ["Tulu3-SFT"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
  ["Tulu3-DPO"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
  ["Tulu3-RLVR"]="allenai/Llama-3.1-Tulu-3-8B"
  ["Tulu3.1"]="allenai/Llama-3.1-Tulu-3.1-8B"
  ["Instruct3"]="meta-llama/Meta-Llama-3-8B-Instruct"
  ["R1-Distill"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

declare -A SHORT=(
  ["Base3.1"]="Base"  ["Tulu3-SFT"]="SFT"  ["Tulu3-DPO"]="DPO"  ["Tulu3-RLVR"]="RLVR"
  ["Instruct3.1"]="Instruct"  ["Tulu3.1"]="Tulu3.1"
  ["Instruct3"]="Llama3-Inst"  ["R1-Distill"]="R1-Distill"
)

adapter_dir() { echo "$ADAPTER_ROOT/$1-$ADAPTER_SUFFIX"; }

SOURCES=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR")
TARGETS=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR" "Instruct3.1" "Tulu3.1" "Instruct3" "R1-Distill")

###############################################################################
# Helper: merge with skip-if-exists
###############################################################################
do_merge() {
  local SRC_ADAPTER="$1" TGT_HF="$2" MERGE_TAG="$3"
  local MERGED_DIR="${SRC_ADAPTER}/merged-${MERGE_TAG}"

  if [[ -d "$MERGED_DIR" ]] && compgen -G "$MERGED_DIR/*.safetensors" >/dev/null 2>&1; then
    echo "[skip] ${MERGE_TAG}"
    return 0
  fi

  echo "[merge] ${MERGE_TAG}"
  mkdir -p "$MERGED_DIR"
  python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
    --adapter_path "$SRC_ADAPTER" \
    --target_base "$TGT_HF" \
    --merge_tag "$MERGE_TAG" \
    --template "llama3"
}

###############################################################################
# Step 1: Cross-delta merges (4 × 8 = 32)
###############################################################################
echo "###############################################################################"
echo "##### Step 1: Cross-delta merges (4 × 8 = 32)                            #####"
echo "###############################################################################"
echo ""

for SRC in "${SOURCES[@]}"; do
  S="${SHORT[$SRC]}"
  SA="$(adapter_dir "$SRC")"
  for TGT in "${TARGETS[@]}"; do
    T="${SHORT[$TGT]}"
    do_merge "$SA" "${HF[$TGT]}" "${S}2${T}"
  done
done

echo ""

###############################################################################
# Step 2: Direct-FT baselines (8 self-merges)
###############################################################################
echo "###############################################################################"
echo "##### Step 2: Direct-FT baselines (8 self-merges)                         #####"
echo "###############################################################################"
echo ""

for TGT in "${TARGETS[@]}"; do
  T="${SHORT[$TGT]}"
  TA="$(adapter_dir "$TGT")"
  do_merge "$TA" "${HF[$TGT]}" "DirectFT-${T}"
done

echo ""

###############################################################################
# Step 3: Weight similarity matrix
###############################################################################
echo "###############################################################################"
echo "##### Step 3: Weight similarity matrix                                    #####"
echo "###############################################################################"
echo ""

SIGMA_DIR="$ADAPTER_ROOT/weight_similarity"
cd "$WORKSPACE_DIR"

python3 weight_similarity_matrix.py --models \
  Base:meta-llama/Llama-3.1-8B \
  SFT:allenai/Llama-3.1-Tulu-3-8B-SFT \
  DPO:allenai/Llama-3.1-Tulu-3-8B-DPO \
  RLVR:allenai/Llama-3.1-Tulu-3-8B \
  Instruct:meta-llama/Llama-3.1-8B-Instruct \
  Tulu3.1:allenai/Llama-3.1-Tulu-3.1-8B \
  Llama3-Inst:meta-llama/Meta-Llama-3-8B-Instruct \
  R1-Distill:deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --output_dir "$SIGMA_DIR"

echo ""
echo "========================================"
echo "  All done!"
echo "========================================"
echo "Next:"
echo "  cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py"
