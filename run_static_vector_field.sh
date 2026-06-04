#!/usr/bin/env bash
###############################################################################
# run_static_vector_field.sh
#
# Driver for the *static post-training vector-field audit* under
# experiments/static_vector_field/.  Pure static weight analysis of a
# Base -> SFT -> DPO -> RLVR lineage: no data, no training, no evaluation.
#
# Typical first-time setup on a fresh machine:
#   git clone https://github.com/yang3121099/LLM-NEO-bk.git
#   cd LLM-NEO-bk
#   git checkout weight_static_analysis_2606
#   bash setup_env_static.sh     # minimal env: torch/numpy/safetensors/mpl/pyyaml
#   conda activate factory
#   bash run_static_vector_field.sh
#
# Usage:
#   bash run_static_vector_field.sh [options]
#
# Options:
#   --config PATH   Lineage YAML (default: experiments/static_vector_field/
#                   configs/llama31_8b_lineage.yaml)
#   --device cpu|cuda   Override the device in the config for this run.
#   --steps LIST    Comma list of steps to run from {analyze,plot,shadow}
#                   (default: analyze,plot,shadow).
#   --limit N       Debug: process only the first N shared tensors.
#   --skip-deps     Do not pip-install the small extra deps (matplotlib/pyyaml).
#   --env NAME      Conda env to activate (default: factory).
#
# Examples:
#   bash run_static_vector_field.sh --device cuda
#   bash run_static_vector_field.sh --steps analyze --limit 50
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/experiments/static_vector_field"
CONFIG="${EXP_DIR}/configs/llama31_8b_lineage.yaml"
CONDA_ENV="factory"
STEPS="analyze,plot,shadow"
LIMIT=""
DEVICE=""
SKIP_DEPS=false

info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --limit)     LIMIT="$2"; shift 2 ;;
    --env)       CONDA_ENV="$2"; shift 2 ;;
    --skip-deps) SKIP_DEPS=true; shift ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *)           error "Unknown option: $1" ;;
  esac
done

###############################################################################
# 1. Activate conda env (factory)
###############################################################################
eval "$(conda shell.bash hook 2>/dev/null)" || true
if conda activate "$CONDA_ENV" 2>/dev/null; then
  info "Activated conda env: $CONDA_ENV"
else
  warn "Could not 'conda activate $CONDA_ENV' — assuming the right Python is already active."
fi

###############################################################################
# 2. Ensure analysis deps
#    torch / numpy / safetensors come from setup_env.sh; matplotlib & pyyaml
#    are the only extras this experiment needs.
###############################################################################
if [[ "$SKIP_DEPS" == "false" ]]; then
  info "Checking analysis dependencies (matplotlib, pyyaml) ..."
  python - <<'PY' || pip install -q matplotlib pyyaml
import importlib, sys
missing = [m for m in ("matplotlib", "yaml") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
fi

# Report what hardware we're about to use.
python - <<'PY' || true
import torch
print(f"[INFO]  torch {torch.__version__} | CUDA available: "
      f"{torch.cuda.is_available()} ({torch.cuda.device_count()} GPU(s))")
PY

[[ -f "$CONFIG" ]] || error "Config not found: $CONFIG"
info "Config: $CONFIG"
info "Steps:  $STEPS"

# Optional device override -> write into a temp copy of the config.
RUN_CONFIG="$CONFIG"
if [[ -n "$DEVICE" ]]; then
  RUN_CONFIG="$(mktemp --suffix=.yaml)"
  python - "$CONFIG" "$DEVICE" "$RUN_CONFIG" <<'PY'
import sys, yaml
src, device, dst = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"[INFO]  device overridden to '{device}' (temp config: {dst})")
PY
  trap 'rm -f "$RUN_CONFIG"' EXIT
fi

###############################################################################
# 3. Run requested steps
###############################################################################
cd "$EXP_DIR"
has_step() { [[ ",$STEPS," == *",$1,"* ]]; }

if has_step analyze; then
  info "==> analyze_lineage.py"
  LIMIT_ARG=""
  [[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"
  python src/analyze_lineage.py --config "$RUN_CONFIG" $LIMIT_ARG
fi

if has_step plot; then
  info "==> plot_lineage.py"
  python src/plot_lineage.py --config "$RUN_CONFIG"
fi

if has_step shadow; then
  info "==> build_static_shadow.py"
  python src/build_static_shadow.py --config "$RUN_CONFIG"
fi

###############################################################################
# 4. Where to look
###############################################################################
OUT_DIR="$(python - "$RUN_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    print(yaml.safe_load(f).get("output_dir", "outputs/static_vector_field"))
PY
)"
echo ""
info "Done. Outputs under: $EXP_DIR/$OUT_DIR"
echo "    - *.csv            metric tables"
echo "    - summary.md       auto report"
echo "    - figures/*.png    plots"
echo "    - static_shadow/   rollback delta candidate"
