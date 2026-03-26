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

# --- NLTK punkt_tab (needed by IFEval) ---
echo ""
echo "[NLTK] Ensuring punkt_tab is downloaded ..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || \
    echo "[NLTK] Warning: Could not download punkt_tab, install nltk first"

echo ""
echo "=== Done ==="
echo "AIME/GSM8K/MATH datasets auto-download via OpenCompass."
echo "Run copy_files.sh if needed: bash src/copy_files.sh <site-packages>"
