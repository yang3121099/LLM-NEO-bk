#!/usr/bin/env bash
###############################################################################
# eval_qwen3_4b_base_bench.sh  —  One-click: env setup → eval
#
# Evaluate Qwen3-4B (Instruct) and Qwen3-4B-Base on 6 base benchmarks:
#   ARC-c (926652), ARC-e (1e0de5), BoolQ (ba58ea),
#   hellaswag (e42710), piqa (1194eb), winogrande (6447e6)
#
# Usage:
#   # Full pipeline (new machine):
#   ./scripts/eval_qwen3_4b_base_bench.sh
#
#   # Skip env setup (already installed):
#   ./scripts/eval_qwen3_4b_base_bench.sh --skip-setup
#
#   # Non-interactive (auto-confirm):
#   ./scripts/eval_qwen3_4b_base_bench.sh --yes
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Parse flags ---
SKIP_SETUP=false
AUTO_YES=false
for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=true ;;
    --yes|-y)     AUTO_YES=true ;;
  esac
done

info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

cat <<'BANNER'

  ╔══════════════════════════════════════════════════╗
  ║   Qwen3-4B  Base Benchmark Evaluation Pipeline   ║
  ╠══════════════════════════════════════════════════╣
  ║  Models:   Qwen3-4B (Instruct) + Qwen3-4B-Base  ║
  ║  Benchmarks:                                     ║
  ║    ARC-c · ARC-e · BoolQ                         ║
  ║    hellaswag · piqa · winogrande                  ║
  ╚══════════════════════════════════════════════════╝

BANNER

if [[ "$AUTO_YES" == "false" ]]; then
  read -rp "Start the pipeline? [Y/n] " ans
  [[ "${ans:-Y}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

###############################################################################
# Stage 1/2: Environment Setup
###############################################################################
info "═══ Stage 1/2: Environment Setup ═══"

if [[ "$SKIP_SETUP" == "true" ]]; then
  info "Skipping setup (--skip-setup)"
else
  bash "$WORKSPACE/setup_env.sh" --skip-conda 2>&1 | tail -5 || \
  bash "$WORKSPACE/setup_env.sh" 2>&1 | tail -20
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate factory 2>/dev/null || error "Cannot activate conda env 'factory'. Run without --skip-setup."

info "Python: $(which python3)"
python3 -c "import torch; print(f'  torch {torch.__version__}  CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}')" 2>/dev/null || warn "torch check failed"
info "Environment ready."

###############################################################################
# Stage 2/2: OpenCompass Evaluation
###############################################################################
echo ""
info "═══ Stage 2/2: OpenCompass Evaluation ═══"
info "Config:    opencompass/eval_qwen3_4b_base_bench.py"
info "Timestamp: $TIMESTAMP"
echo ""

cd "$WORKSPACE/opencompass"

python3 ./run.py ./eval_qwen3_4b_base_bench.py \
    -r "$TIMESTAMP" \
    --dump-eval-details

echo ""
info "═══ Pipeline Complete ═══"
info "Results → $WORKSPACE/opencompass/outputs/eval-qwen3-4b-base-bench/"
echo ""
