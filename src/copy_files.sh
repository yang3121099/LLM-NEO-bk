#!/bin/bash
# Sync local opencompass source to the pip-installed site-packages.
#
# Instead of chasing individual file mismatches, this script syncs
# entire subdirectories that are known to diverge.
#
# Usage:
#   bash src/copy_files.sh /venv/factory/lib/python3.10/site-packages
#   bash src/copy_files.sh /data/miniconda3/envs/opencompass/lib/python3.10/site-packages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OC_SRC="$REPO_DIR/opencompass/opencompass"

TARGET_PATH="${1:-}"

if [ -z "$TARGET_PATH" ]; then
    echo "Usage: bash $0 <site-packages-path>"
    echo "  e.g. bash $0 /venv/factory/lib/python3.10/site-packages"
    exit 1
fi

OC_DST="$TARGET_PATH/opencompass"

if [ ! -d "$OC_DST" ]; then
    echo "ERROR: $OC_DST does not exist"
    exit 1
fi

echo "Source:  $OC_SRC"
echo "Target:  $OC_DST"
echo ""

# --- Sync entire directories (rsync-style, preserving structure) ---

# tasks/ — extract_role_pred, openicl_eval, etc.
echo "=== tasks/ ==="
cp -v "$OC_SRC/tasks/"*.py "$OC_DST/tasks/"

# openicl/icl_evaluator/ — BaseEvaluator.evaluate(), TEvalEvaluator, etc.
echo ""
echo "=== openicl/icl_evaluator/ ==="
cp -v "$OC_SRC/openicl/icl_evaluator/"*.py "$OC_DST/openicl/icl_evaluator/"

# datasets/ — mbpp, humaneval, IFEval, teval, etc.
echo ""
echo "=== datasets/ (top-level .py) ==="
cp -v "$OC_SRC/datasets/"*.py "$OC_DST/datasets/"

# datasets/IFEval/
if [ -d "$OC_SRC/datasets/IFEval" ]; then
    echo ""
    echo "=== datasets/IFEval/ ==="
    mkdir -p "$OC_DST/datasets/IFEval"
    cp -v "$OC_SRC/datasets/IFEval/"*.py "$OC_DST/datasets/IFEval/"
fi

# models/ — turbomind, etc.
echo ""
echo "=== models/ (top-level .py) ==="
cp -v "$OC_SRC/models/"*.py "$OC_DST/models/"

echo ""
echo "Done. Synced local opencompass to: $TARGET_PATH"
