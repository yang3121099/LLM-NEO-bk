#!/usr/bin/env bash
###############################################################################
# cleanup_merged.sh — 清理之前合并的全量模型目录，释放磁盘空间
#
# 只删除 merged-* 目录（B2I/I2I/B2B/I2B 的全量模型），保留 LoRA adapter
#
# Usage:
#   bash cleanup_merged.sh              # 预览要删除的目录
#   bash cleanup_merged.sh --delete     # 实际删除
###############################################################################
set -euo pipefail

RESULTS_DIR="${1:-/workspace/LLM-NEO-bk/results}"
DELETE=false

for arg in "$@"; do
  case "$arg" in
    --delete) DELETE=true ;;
    --help|-h)
      echo "Usage: bash cleanup_merged.sh [RESULTS_DIR] [--delete]"
      echo "  Default RESULTS_DIR: /workspace/LLM-NEO-bk/results"
      echo "  Without --delete: dry-run (preview only)"
      exit 0 ;;
  esac
done

echo "Scanning: $RESULTS_DIR"
echo ""

TOTAL_SIZE=0
COUNT=0

while IFS= read -r -d '' dir; do
  SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
  SIZE_BYTES=$(du -sb "$dir" 2>/dev/null | cut -f1)
  echo "  [$SIZE]  $dir"
  TOTAL_SIZE=$((TOTAL_SIZE + SIZE_BYTES))
  COUNT=$((COUNT + 1))
done < <(find "$RESULTS_DIR" -type d -name "merged-*" -print0 2>/dev/null | sort -z)

if [[ $COUNT -eq 0 ]]; then
  echo "No merged-* directories found."
  exit 0
fi

TOTAL_HR=$(numfmt --to=iec-i --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "${TOTAL_SIZE} bytes")
echo ""
echo "Found $COUNT merged directories, total: $TOTAL_HR"
echo ""

if $DELETE; then
  read -p "Confirm delete $COUNT directories? [y/N] " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    find "$RESULTS_DIR" -type d -name "merged-*" -print0 | while IFS= read -r -d '' dir; do
      echo "  rm -rf $dir"
      rm -rf "$dir"
    done
    echo "Done. Freed ~$TOTAL_HR"
  else
    echo "Cancelled."
  fi
else
  echo "Dry run. To actually delete, run:"
  echo "  bash cleanup_merged.sh $RESULTS_DIR --delete"
fi
