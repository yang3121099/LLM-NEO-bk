#!/bin/bash
# Sync local opencompass patches to the pip-installed site-packages.
#
# Usage:
#   bash src/copy_files.sh /data/miniconda3/envs/opencompass/lib/python3.10/site-packages
#   bash src/copy_files.sh /venv/factory/lib/python3.10/site-packages

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

# --- datasets ---
cp -v "$OC_SRC/datasets/mbpp.py"      "$OC_DST/datasets/mbpp.py"
cp -v "$OC_SRC/datasets/humaneval.py"  "$OC_DST/datasets/humaneval.py"

# --- models ---
cp -v "$OC_SRC/models/turbomind_with_tf_above_v4_33.py" \
      "$OC_DST/models/turbomind_with_tf_above_v4_33.py"

# --- tasks (extract_role_pred fix) ---
cp -v "$OC_SRC/tasks/base.py"          "$OC_DST/tasks/base.py"
cp -v "$OC_SRC/tasks/openicl_eval.py"  "$OC_DST/tasks/openicl_eval.py"

# --- evaluator (BaseEvaluator.evaluate() fix) ---
cp -v "$OC_SRC/openicl/icl_evaluator/icl_base_evaluator.py" \
      "$OC_DST/openicl/icl_evaluator/icl_base_evaluator.py"

echo ""
echo "Done. Synced local opencompass patches to: $TARGET_PATH"
