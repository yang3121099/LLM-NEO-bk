#!/usr/bin/env bash
###############################################################################
# download_eval_data.sh — Download evaluation datasets that need manual setup
#
# Usage: bash scripts/download_eval_data.sh
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${COMPASS_DATA_CACHE:-$HOME/.cache/opencompass}"
DATA_DIR="$CACHE_DIR/data"

mkdir -p "$DATA_DIR"

echo "=== Downloading evaluation datasets ==="
echo "Cache dir: $CACHE_DIR"
echo ""

# --- SVAMP (from HuggingFace: ChilleD/SVAMP) ---
SVAMP_DIR="$DATA_DIR/svamp"
if [ ! -f "$SVAMP_DIR/test.json" ]; then
    echo "[SVAMP] Downloading test.json from HuggingFace ..."
    mkdir -p "$SVAMP_DIR"
    curl -L -o "$SVAMP_DIR/test.json" \
        "https://huggingface.co/datasets/ChilleD/SVAMP/resolve/main/test.json"
    echo "[SVAMP] Done: $SVAMP_DIR/test.json"
else
    echo "[SVAMP] Already exists: $SVAMP_DIR/test.json"
fi

# --- T-Eval (tool-use evaluation, from OpenCompass T-Eval) ---
TEVAL_DIR="$WORKSPACE_DIR/opencompass/data/teval"
if [ ! -d "$TEVAL_DIR/EN" ]; then
    echo "[T-Eval] Downloading T-Eval data ..."
    mkdir -p "$TEVAL_DIR"
    cd "$TEVAL_DIR"
    # T-Eval data is hosted in the OpenCompass data repo
    git clone --depth 1 https://github.com/open-compass/T-Eval.git _teval_tmp 2>/dev/null || {
        echo "[T-Eval] Git clone failed, trying HuggingFace ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('open-compass/T-Eval', repo_type='dataset',
                  local_dir='_teval_tmp', allow_patterns=['data/*'])
" 2>/dev/null || echo "[T-Eval] WARNING: Could not download T-Eval data automatically."
    }
    # Move data files into place
    if [ -d "_teval_tmp/data" ]; then
        cp -r _teval_tmp/data/* . 2>/dev/null || true
        rm -rf _teval_tmp
        echo "[T-Eval] Done: $TEVAL_DIR/"
    elif [ -d "_teval_tmp" ]; then
        # Try alternate layout
        find _teval_tmp -name "*.json" -path "*/EN/*" -exec cp --parents {} . \; 2>/dev/null || true
        rm -rf _teval_tmp
        echo "[T-Eval] Done (partial): $TEVAL_DIR/"
    else
        echo "[T-Eval] WARNING: Download failed. Please manually download T-Eval data to $TEVAL_DIR/EN/"
        echo "         See: https://github.com/open-compass/T-Eval"
    fi
    cd "$WORKSPACE_DIR"
else
    echo "[T-Eval] Already exists: $TEVAL_DIR/EN/"
fi

# --- NLTK punkt_tab (needed by IFEval) ---
echo ""
echo "[NLTK] Ensuring punkt_tab is downloaded ..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || \
    echo "[NLTK] Warning: Could not download punkt_tab, install nltk first"

echo ""
echo "=== Done ==="
echo "AIME/GSM8K/MATH datasets auto-download via OpenCompass."
echo "T-Eval data: $WORKSPACE_DIR/opencompass/data/teval/EN/"
echo "Run copy_files.sh if needed: bash src/copy_files.sh <site-packages>"
