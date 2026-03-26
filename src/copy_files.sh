#!/bin/bash
# Sync the ENTIRE local opencompass package over the pip-installed version.
#
# This replaces the installed opencompass with the local source to avoid
# endless version-mismatch errors (JSONToolkit, extract_role_pred, etc.).
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

# Full sync: copy all .py files recursively, preserving directory structure.
# This is the only reliable way to keep local source and installed package in sync.
echo "Syncing entire opencompass package ..."

if command -v rsync &> /dev/null; then
    rsync -av --include='*/' --include='*.py' --exclude='*' \
        "$OC_SRC/" "$OC_DST/"
else
    # Fallback without rsync: find + cp
    cd "$OC_SRC"
    find . -name '*.py' | while read -r f; do
        mkdir -p "$OC_DST/$(dirname "$f")"
        cp "$f" "$OC_DST/$f"
    done
fi

echo ""
echo "Done. Synced entire opencompass package to: $OC_DST"
