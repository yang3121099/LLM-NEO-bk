#!/usr/bin/env bash
###############################################################################
# fast_download.sh — aria2c multi-thread + hf-mirror download
#
# Usage:
#   bash scripts/fast_download.sh Qwen/Qwen2.5-32B
#   bash scripts/fast_download.sh Qwen/Qwen2.5-32B --start 2   # skip shard 1
#   bash scripts/fast_download.sh Qwen/Qwen2.5-32B-Instruct
#   bash scripts/fast_download.sh Qwen/Qwen3-30B-A3B-Base
#
# Env:
#   THREADS=16       Connections per file (default: 16)
#   MIRROR=https://hf-mirror.com   HF mirror URL
###############################################################################
set -euo pipefail

REPO="${1:?Usage: $0 <repo_id> [--start N]}"
START_SHARD="${3:-0}"  # skip shards before this number
if [[ "${2:-}" == "--start" ]]; then
  START_SHARD="${3:-0}"
fi

THREADS="${THREADS:-16}"
MIRROR="${MIRROR:-https://hf-mirror.com}"

# Install aria2 if not present
if ! command -v aria2c &>/dev/null; then
  echo "Installing aria2 ..."
  apt-get update -qq && apt-get install -y -qq aria2 2>/dev/null || \
  conda install -y -c conda-forge aria2 2>/dev/null || \
  pip install aria2p 2>/dev/null
fi

# HF cache directory
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
REPO_CACHE="$HF_CACHE/models--$(echo "$REPO" | tr '/' '--')"
BLOBS_DIR="$REPO_CACHE/blobs"
REFS_DIR="$REPO_CACHE/refs"
SNAPS_DIR="$REPO_CACHE/snapshots"
mkdir -p "$BLOBS_DIR" "$REFS_DIR" "$SNAPS_DIR"

echo "=== Fast Download: $REPO ==="
echo "Mirror:  $MIRROR"
echo "Threads: $THREADS per file"
echo "Cache:   $REPO_CACHE"
echo ""

# Step 1: Get file list from the repo
echo "Fetching file list ..."
FILE_LIST=$(python3 << PYEOF
from huggingface_hub import HfApi, hf_hub_url
import os
os.environ['HF_ENDPOINT'] = '${MIRROR}'
api = HfApi(endpoint='${MIRROR}')
files = api.list_repo_files('${REPO}')
for f in files:
    # Get the download URL
    url = f'${MIRROR}/${REPO}/resolve/main/{f}'
    print(f'{f}|{url}')
PYEOF
)

if [[ -z "$FILE_LIST" ]]; then
  echo "ERROR: Could not fetch file list. Trying direct approach..."
  # Fallback: common files for a typical model
  echo "Please check the repo URL: ${MIRROR}/${REPO}"
  exit 1
fi

TOTAL=$(echo "$FILE_LIST" | wc -l)
echo "Found $TOTAL files"
echo ""

# Step 2: Download each file with aria2c
DOWNLOADED=0
SKIPPED=0

# Create a temp dir for downloads
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

while IFS='|' read -r FILENAME URL; do
  # Skip directories / empty
  [[ -z "$FILENAME" ]] && continue

  # Check if this is a shard file and apply --start filter
  if [[ "$START_SHARD" -gt 0 ]] && [[ "$FILENAME" =~ model-([0-9]+)-of- ]]; then
    SHARD_NUM=$((10#${BASH_REMATCH[1]}))
    if (( SHARD_NUM < START_SHARD )); then
      echo "SKIP (before shard $START_SHARD): $FILENAME"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
  fi

  # Target path in HF cache (simplified: download to snapshot dir)
  SNAP_MAIN="$SNAPS_DIR/main"
  mkdir -p "$SNAP_MAIN/$(dirname "$FILENAME")"
  TARGET="$SNAP_MAIN/$FILENAME"

  # Skip if already exists and has content
  if [[ -f "$TARGET" ]] && [[ -s "$TARGET" ]]; then
    echo "EXIST: $FILENAME ($(du -h "$TARGET" | cut -f1))"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo ""
  echo ">>> [$((DOWNLOADED + SKIPPED + 1))/$TOTAL] $FILENAME"
  echo "    $URL"

  # aria2c: multi-thread download with resume
  aria2c \
    -x "$THREADS" \
    -s "$THREADS" \
    -k 1M \
    --continue=true \
    --max-tries=5 \
    --retry-wait=3 \
    --file-allocation=none \
    --console-log-level=notice \
    -d "$(dirname "$TARGET")" \
    -o "$(basename "$FILENAME")" \
    "$URL" && {
      DOWNLOADED=$((DOWNLOADED + 1))
      echo "<<< OK: $FILENAME"
    } || {
      echo "<<< FAIL: $FILENAME (will retry on next run)"
    }

done <<< "$FILE_LIST"

echo ""
echo "=== Download Complete ==="
echo "Downloaded: $DOWNLOADED"
echo "Skipped:   $SKIPPED"
echo "Location:  $SNAP_MAIN/"
echo ""
echo "To use this model, set:"
echo "  export MODEL_PATH=$SNAP_MAIN"
echo ""
echo "Or create a symlink for HF auto-detection:"
echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('${REPO}', local_dir='$SNAP_MAIN')\""
