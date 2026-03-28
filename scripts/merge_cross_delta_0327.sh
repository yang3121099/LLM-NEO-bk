#!/usr/bin/env bash
###############################################################################
# merge_cross_delta_0327.sh
#
# Merge-only script for Cross-Delta Transfer Experiment.
# Uses already-trained LoRA adapters from:
#   results/0327/result-weight-similarity-0327/{NAME}-2k-lora-rank128-lr0.0002-shadow2k/
#
# Automatically skips missing adapters (e.g. if Instruct3/R1-Distill not trained).
###############################################################################
set -uo pipefail  # no -e: we handle errors per-merge

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
ADAPTER_ROOT="$RESULTS_DIR/0327/result-weight-similarity-0327"
ADAPTER_SUFFIX="2k-lora-rank128-lr0.0002-shadow2k"

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

# Short display names for the transfer matrix
declare -A SHORT=(
  ["Base3.1"]="Base"  ["Tulu3-SFT"]="SFT"  ["Tulu3-DPO"]="DPO"  ["Tulu3-RLVR"]="RLVR"
  ["Instruct3.1"]="Instruct"  ["Tulu3.1"]="Tulu3.1"
  ["Instruct3"]="Llama3-Inst"  ["R1-Distill"]="R1-Distill"
)

adapter_dir() { echo "$ADAPTER_ROOT/$1-$ADAPTER_SUFFIX"; }

###############################################################################
# Experiment design
###############################################################################
SOURCES=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR")
TARGETS=("Base3.1" "Tulu3-SFT" "Tulu3-DPO" "Tulu3-RLVR" "Instruct3.1" "Tulu3.1" "Instruct3" "R1-Distill")

echo "========================================"
echo "  Cross-Delta Merge (0327)"
echo "========================================"
echo ""

# Check which adapters exist
echo "Checking adapters..."
declare -A ADAPTER_OK
SEEN=()
for name in "${SOURCES[@]}" "${TARGETS[@]}"; do
  [[ " ${SEEN[*]:-} " =~ " $name " ]] && continue
  SEEN+=("$name")
  dir="$(adapter_dir "$name")"
  if [[ -d "$dir" ]]; then
    ADAPTER_OK[$name]=1
    echo "  OK:      $name -> $dir"
  else
    ADAPTER_OK[$name]=0
    echo "  MISSING: $name (will skip as target, train baseline not available)"
  fi
done
echo ""

# Filter targets to only those with adapters (for baseline merges)
AVAILABLE_TARGETS=()
for tgt in "${TARGETS[@]}"; do
  AVAILABLE_TARGETS+=("$tgt")  # keep all as merge targets (HF models always available)
done

###############################################################################
# Step 1: Cross-delta merges (4 × 8)
###############################################################################
echo "###############################################################################"
echo "##### Step 1: Cross-delta merges                                          #####"
echo "###############################################################################"
echo ""

MERGED=0
SKIPPED=0
FAILED=0

do_merge() {
  local SRC_ADAPTER="$1" TGT_HF="$2" MERGE_TAG="$3"
  local MERGED_DIR="${SRC_ADAPTER}/merged-${MERGE_TAG}"

  # Skip if already merged
  if [[ -d "$MERGED_DIR" ]] && compgen -G "$MERGED_DIR/*.safetensors" >/dev/null 2>&1; then
    echo "[skip] ${MERGE_TAG} — already exists"
    return 0
  fi

  echo "[merge] ${MERGE_TAG}"
  mkdir -p "$MERGED_DIR"
  python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
    --adapter_path "$SRC_ADAPTER" \
    --target_base "$TGT_HF" \
    --merge_tag "$MERGE_TAG" \
    --template "llama3" 2>&1 | tail -5
  return ${PIPESTATUS[0]}
}

for SRC in "${SOURCES[@]}"; do
  SRC_SHORT="${SHORT[$SRC]}"
  SRC_ADAPTER="$(adapter_dir "$SRC")"

  if [[ ! -d "$SRC_ADAPTER" ]]; then
    echo "[WARN] Source adapter missing: $SRC — skipping all its merges"
    SKIPPED=$((SKIPPED + ${#TARGETS[@]}))
    continue
  fi

  for TGT in "${TARGETS[@]}"; do
    TGT_SHORT="${SHORT[$TGT]}"
    TGT_HF="${HF_PATHS[$TGT]}"
    MERGE_TAG="${SRC_SHORT}2${TGT_SHORT}"

    if do_merge "$SRC_ADAPTER" "$TGT_HF" "$MERGE_TAG"; then
      MERGED=$((MERGED + 1))
    else
      echo "[FAIL] ${MERGE_TAG}"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo ""
echo "Cross-delta: merged=$MERGED, skipped=$SKIPPED, failed=$FAILED"
echo ""

###############################################################################
# Step 2: Direct-FT baselines (self-merge for each target that has an adapter)
###############################################################################
echo "###############################################################################"
echo "##### Step 2: Direct-FT baselines                                         #####"
echo "###############################################################################"
echo ""

for TGT in "${TARGETS[@]}"; do
  TGT_SHORT="${SHORT[$TGT]}"
  TGT_HF="${HF_PATHS[$TGT]}"
  TGT_ADAPTER="$(adapter_dir "$TGT")"
  MERGE_TAG="DirectFT-${TGT_SHORT}"

  if [[ "${ADAPTER_OK[$TGT]}" != "1" ]]; then
    echo "[skip] ${MERGE_TAG} — adapter not available for $TGT"
    continue
  fi

  if do_merge "$TGT_ADAPTER" "$TGT_HF" "$MERGE_TAG"; then
    echo "  done"
  else
    echo "[FAIL] ${MERGE_TAG}"
  fi
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
cd "$WORKSPACE_DIR"

# Build model list from available targets only
SIGMA_ARGS=""
for name in "${SEEN[@]}"; do
  hf="${HF_PATHS[$name]}"
  short="${SHORT[$name]}"
  SIGMA_ARGS="$SIGMA_ARGS ${short}:${hf}"
done

echo "Computing pairwise σ..."
python3 weight_similarity_matrix.py --models $SIGMA_ARGS \
  --output_dir "$SIGMA_DIR" || echo "[WARN] Weight similarity computation failed (models may not be locally available)"

echo ""
echo "========================================"
echo "  All merges complete!"
echo "========================================"
echo ""
echo "Next:"
echo "  1. cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py"
echo "  2. python3 visualize_transfer_matrix.py \\"
echo "       --eval_dir opencompass/outputs/cross-delta-0327/ \\"
echo "       --sigma_file $SIGMA_DIR/pairwise_results.json \\"
echo "       --output_dir $RESULTS_DIR/0327/result-weight-similarity-0327/transfer_matrix"
