#!/usr/bin/env bash
###############################################################################
# merge_cross_delta_0327.sh
#
# Merge-only script for Cross-Delta Transfer Experiment.
# Uses already-trained LoRA adapters from:
#   results/0327/result-weight-similarity-0327/{NAME}-2k-lora-rank128-lr0.0002-shadow2k/
#
# 4 sources × 8 targets = 32 cross-delta merges + 8 baseline self-merges
# No training needed — adapters are already available.
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
ADAPTER_ROOT="$RESULTS_DIR/0327/result-weight-similarity-0327"

###############################################################################
# Adapter name → HF model path mapping
###############################################################################
declare -A HF_PATHS=(
  ["Base3.1"]="meta-llama/Llama-3.1-8B"
  ["Instruct3.1"]="meta-llama/Llama-3.1-8B-Instruct"
  ["Tulu3-SFT"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
  ["Tulu3-DPO"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
  ["Tulu3-RLVR"]="allenai/Llama-3.1-Tulu-3-8B"
  ["Tulu3.1"]="allenai/Llama-3.1-Tulu-3.1-8B"
  ["Base3"]="meta-llama/Llama-3-8B"
  ["Instruct3"]="meta-llama/Meta-Llama-3-8B-Instruct"
  ["R1-Distill"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Adapter directory suffix
ADAPTER_SUFFIX="2k-lora-rank128-lr0.0002-shadow2k"

adapter_dir() {
  echo "$ADAPTER_ROOT/$1-$ADAPTER_SUFFIX"
}

###############################################################################
# Cross-Delta Experiment Design
#
# Sources (4): Base3.1, Tulu3-SFT, Tulu3-DPO, Tulu3-RLVR
# Targets (8): Base3.1, Tulu3-SFT, Tulu3-DPO, Tulu3-RLVR,
#              Instruct3.1, Tulu3.1, Instruct3, R1-Distill
###############################################################################
SOURCES=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR")
TARGETS=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR" "Instruct3.1" "Tulu3.1" "Instruct3" "R1-Distill")

# Short names for the transfer matrix (used in eval abbr and visualization)
declare -A SHORT_NAMES=(
  ["Base3.1"]="Base"
  ["Tulu3-SFT"]="SFT"
  ["Tulu3-DPO"]="DPO"
  ["Tulu3-RLVR"]="RLVR"
  ["Instruct3.1"]="Instruct"
  ["Tulu3.1"]="Tulu3.1"
  ["Instruct3"]="Llama3-Inst"
  ["R1-Distill"]="R1-Distill"
)

echo "========================================"
echo "  Cross-Delta Merge (0327)"
echo "========================================"
echo ""
echo "Adapter root: $ADAPTER_ROOT"
echo "Sources: ${SOURCES[*]}"
echo "Targets: ${TARGETS[*]}"
echo ""

# Verify all adapters exist
echo "Checking adapters..."
ALL_OK=true
ALL_NAMES=()
for name in "${SOURCES[@]}" "${TARGETS[@]}"; do
  # Deduplicate
  [[ " ${ALL_NAMES[*]:-} " =~ " $name " ]] && continue
  ALL_NAMES+=("$name")

  dir="$(adapter_dir "$name")"
  if [[ -d "$dir" ]]; then
    echo "  OK: $name -> $dir"
  else
    echo "  MISSING: $name -> $dir"
    ALL_OK=false
  fi
done

if ! $ALL_OK; then
  echo ""
  echo "ERROR: Some adapters are missing. Please check the paths above."
  exit 1
fi
echo ""

###############################################################################
# Step 1: Cross-delta merges (4 × 8 = 32)
###############################################################################
echo "###############################################################################"
echo "##### Step 1: Cross-delta merges (4 sources × 8 targets = 32)             #####"
echo "###############################################################################"
echo ""

MERGE_COUNT=0
for SRC in "${SOURCES[@]}"; do
  SRC_SHORT="${SHORT_NAMES[$SRC]}"
  SRC_ADAPTER="$(adapter_dir "$SRC")"

  for TGT in "${TARGETS[@]}"; do
    TGT_SHORT="${SHORT_NAMES[$TGT]}"
    TGT_HF="${HF_PATHS[$TGT]}"
    MERGE_TAG="${SRC_SHORT}2${TGT_SHORT}"
    MERGED_DIR="${SRC_ADAPTER}/merged-${MERGE_TAG}"

    # Skip if already merged
    if [[ -d "$MERGED_DIR" ]] && ls "$MERGED_DIR"/*.safetensors &>/dev/null 2>&1; then
      echo "[skip] ${MERGE_TAG} already exists: $MERGED_DIR"
      MERGE_COUNT=$((MERGE_COUNT + 1))
      continue
    fi

    echo "[merge] ${MERGE_TAG}: adapter=${SRC} → target=${TGT}"
    mkdir -p "$MERGED_DIR"
    python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
      --adapter_path "$SRC_ADAPTER" \
      --target_base "$TGT_HF" \
      --merge_tag "$MERGE_TAG" \
      --template "llama3"

    MERGE_COUNT=$((MERGE_COUNT + 1))
  done
done
echo ""
echo "Completed $MERGE_COUNT cross-delta merges."
echo ""

###############################################################################
# Step 2: Direct-FT baseline merges (8 targets, each self-merge)
###############################################################################
echo "###############################################################################"
echo "##### Step 2: Direct-FT baselines (8 self-merges)                         #####"
echo "###############################################################################"
echo ""

for TGT in "${TARGETS[@]}"; do
  TGT_SHORT="${SHORT_NAMES[$TGT]}"
  TGT_HF="${HF_PATHS[$TGT]}"
  TGT_ADAPTER="$(adapter_dir "$TGT")"
  MERGE_TAG="DirectFT-${TGT_SHORT}"
  MERGED_DIR="${TGT_ADAPTER}/merged-${MERGE_TAG}"

  if [[ -d "$MERGED_DIR" ]] && ls "$MERGED_DIR"/*.safetensors &>/dev/null 2>&1; then
    echo "[skip] ${MERGE_TAG} already exists"
    continue
  fi

  echo "[merge] ${MERGE_TAG}: adapter=${TGT} → self"
  mkdir -p "$MERGED_DIR"
  python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
    --adapter_path "$TGT_ADAPTER" \
    --target_base "$TGT_HF" \
    --merge_tag "$MERGE_TAG" \
    --template "llama3"
done

echo ""

###############################################################################
# Step 3: Weight similarity matrix
###############################################################################
echo "###############################################################################"
echo "##### Step 3: Weight similarity matrix                                    #####"
echo "###############################################################################"
echo ""

SIGMA_DIR="$RESULTS_DIR/0327/result-weight-similarity-0327/weight_similarity"
echo "Computing pairwise σ..."
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
echo "  All merges complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Eval:      cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py"
echo "  2. Visualize: python3 visualize_transfer_matrix.py \\"
echo "       --eval_dir opencompass/outputs/cross-delta-0327/ \\"
echo "       --sigma_file $SIGMA_DIR/pairwise_results.json \\"
echo "       --output_dir $RESULTS_DIR/0327/result-weight-similarity-0327/transfer_matrix"
