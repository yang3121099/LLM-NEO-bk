#!/usr/bin/env bash
###############################################################################
# setup_env_static.sh — Minimal environment for the static vector-field audit
#
# This sets up ONLY what experiments/static_vector_field/ needs. It deliberately
# does NOT install LlamaFactory, OpenCompass, lmdeploy, vLLM, or any eval/training
# stack. The audit imports just: torch, numpy, safetensors, matplotlib, pyyaml.
#
# (The full Shadow-FT pipeline setup still lives in setup_env.sh — this file does
#  not touch it.)
#
# Usage:
#   bash setup_env_static.sh                 # create conda env `factory` + deps
#   bash setup_env_static.sh --skip-conda    # env already exists, just deps
#   bash setup_env_static.sh --env myenv     # use a different conda env name
#   bash setup_env_static.sh --cpu           # install the CPU-only torch wheel
#   bash setup_env_static.sh --no-conda      # install into the active interpreter
###############################################################################
set -euo pipefail

CONDA_ENV="factory"
PYTHON_VER="3.10"
SKIP_CONDA=false
USE_CONDA=true
TORCH_CPU=false

info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-conda) SKIP_CONDA=true; shift ;;
    --no-conda)   USE_CONDA=false; shift ;;
    --env)        CONDA_ENV="$2"; shift 2 ;;
    --python)     PYTHON_VER="$2"; shift 2 ;;
    --cpu)        TORCH_CPU=true; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *)            error "Unknown option: $1" ;;
  esac
done

###############################################################################
# 1. Conda environment (optional)
###############################################################################
if [[ "$USE_CONDA" == "true" ]]; then
  command -v conda >/dev/null 2>&1 || error "conda not found. Use --no-conda to install into the active Python."
  if [[ "$SKIP_CONDA" == "false" ]]; then
    info "Creating conda env: $CONDA_ENV (python $PYTHON_VER)"
    conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y 2>/dev/null || true
  fi
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate "$CONDA_ENV" 2>/dev/null || warn "conda activate failed; assuming env is already active"
else
  info "Installing into the currently active Python interpreter."
fi

info "Python: $(python -c 'import sys; print(sys.version.split()[0])')  ($(command -v python))"

###############################################################################
# 2. PyTorch
#    Default: the standard wheel (CUDA build) so the GPU path works.
#    --cpu:   the CPU-only wheel (smaller, no CUDA needed).
###############################################################################
if python -c "import torch" 2>/dev/null; then
  info "torch already present: $(python -c 'import torch; print(torch.__version__)')"
else
  if [[ "$TORCH_CPU" == "true" ]]; then
    info "Installing torch (CPU-only wheel) ..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  else
    info "Installing torch (default / CUDA wheel) ..."
    pip install -q torch
  fi
fi

###############################################################################
# 3. The remaining (small) dependencies
###############################################################################
info "Installing numpy, safetensors, matplotlib, pyyaml, huggingface_hub ..."
pip install -q numpy safetensors matplotlib pyyaml "huggingface_hub[cli]"

###############################################################################
# 4. Verify
###############################################################################
info "Verifying ..."
python - <<'PY' || error "Dependency import failed."
import torch, numpy, safetensors, matplotlib, yaml, huggingface_hub
print(f"  torch:           {torch.__version__}")
print(f"  numpy:           {numpy.__version__}")
print(f"  safetensors:     {safetensors.__version__}")
print(f"  matplotlib:      {matplotlib.__version__}")
print(f"  huggingface_hub: {huggingface_hub.__version__}")
print(f"  CUDA:            {torch.cuda.is_available()} ({torch.cuda.device_count()} GPU(s))")
PY

echo ""
info "Minimal environment ready."
echo ""
if [[ "$USE_CONDA" == "true" ]]; then
  echo "  Activate:  conda activate $CONDA_ENV"
fi
echo "  HF auth:   huggingface-cli login          # paste token at the prompt (not on the CLI)"
echo "             # or:  export HF_TOKEN=<your-token>   (set in your shell, never commit it)"
echo "  Run:       bash run_static_vector_field.sh            # CPU"
echo "             bash run_static_vector_field.sh --device cuda"
echo ""
