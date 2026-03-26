#!/usr/bin/env bash
###############################################################################
# setup_env.sh — One-shot environment setup for Shadow-FT pipeline
#
# Handles: conda env, LlamaFactory, OpenCompass, lmdeploy, eval deps,
#          dataset downloads, and source sync.
#
# Usage:
#   # Full setup (new machine):
#   bash setup_env.sh
#
#   # Skip conda creation (env already exists):
#   bash setup_env.sh --skip-conda
#
#   # Only sync source files (after git pull):
#   bash setup_env.sh --sync-only
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${SCRIPT_DIR}"
CONDA_ENV="factory"
PYTHON_VER="3.10"

# --- Parse flags ---
SKIP_CONDA=false
SYNC_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --skip-conda) SKIP_CONDA=true ;;
    --sync-only)  SYNC_ONLY=true ;;
  esac
done

###############################################################################
# 0. Helper
###############################################################################
info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# Detect site-packages path
get_site_packages() {
  python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null
}

###############################################################################
# 1. Conda environment
###############################################################################
if [[ "$SYNC_ONLY" == "true" ]]; then
  info "Sync-only mode, skipping install steps ..."
else

if [[ "$SKIP_CONDA" == "false" ]]; then
  info "Creating conda environment: $CONDA_ENV (python $PYTHON_VER)"
  conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y 2>/dev/null || true
fi

# Activate (works in both bash and script context)
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate "$CONDA_ENV" 2>/dev/null || {
  warn "conda activate failed, assuming env is already active"
}

###############################################################################
# 2. Install LlamaFactory (editable)
###############################################################################
info "Installing LlamaFactory ..."
cd "$WORKSPACE"
pip install -e ".[torch,metrics]" -q

###############################################################################
# 3. Core dependencies (pinned versions)
###############################################################################
info "Installing core dependencies ..."
pip install -q \
  torch==2.6.0 \
  transformers==4.51.2 \
  torchvision \
  deepspeed \
  peft==0.15.2 \
  importlib_metadata \
  omegaconf

###############################################################################
# 4. OpenCompass + eval backends
###############################################################################
info "Installing OpenCompass ..."
cd "$WORKSPACE/opencompass"
pip install -q ".[vllm]"

info "Installing lmdeploy + eval tools ..."
pip install -q \
  lmdeploy \
  pynvml \
  evalplus==0.3.1 \
  human-eval \
  latex2sympy2_extended \
  math_verify \
  prettytable \
  jieba \
  rouge_chinese \
  rank_bm25 \
  gradio_client \
  tree_sitter_languages \
  fuzzywuzzy \
  h5py

# lm-evaluation-harness (for additional evals)
pip install -q "git+https://github.com/EleutherAI/lm-evaluation-harness.git"

# AQLM support
pip install -q "aqlm[gpu,cpu]" 2>/dev/null || warn "AQLM install failed (optional)"

# HuggingFace CLI
pip install -q "huggingface_hub[cli]==0.*"

###############################################################################
# 5. NLTK data (needed by IFEval)
###############################################################################
info "Downloading NLTK punkt_tab ..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

fi  # end of non-sync-only block

###############################################################################
# 6. Sync local OpenCompass source → installed site-packages
###############################################################################
SITE_PACKAGES="$(get_site_packages)"
if [[ -d "$SITE_PACKAGES/opencompass" ]]; then
  info "Syncing local opencompass source → $SITE_PACKAGES"
  bash "$WORKSPACE/src/copy_files.sh" "$SITE_PACKAGES"
else
  warn "No installed opencompass found at $SITE_PACKAGES, skipping sync"
fi

###############################################################################
# 7. Download eval datasets (SVAMP, etc.)
###############################################################################
info "Downloading eval datasets ..."
bash "$WORKSPACE/scripts/download_eval_data.sh"

###############################################################################
# 8. Verify
###############################################################################
info "Verifying installation ..."
cd "$WORKSPACE"

python3 -c "
import torch, transformers, peft, lmdeploy
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  lmdeploy:     {lmdeploy.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
" 2>/dev/null || warn "Some imports failed, check versions"

python3 -c "
from llamafactory.train import run_exp
print('  llamafactory: OK')
" 2>/dev/null || warn "LlamaFactory import failed"

echo ""
info "Setup complete!"
echo ""
echo "  Activate:  conda activate $CONDA_ENV"
echo "  Generate:  bash generate_dataset_scripts.sh"
echo "  Prepare:   python3 scripts/prepare_deepmath_103k.py  (if using DeepMath)"
echo "  Train:     bash scripts/train_<exp>_<ts>.sh"
echo "  Eval:      cd opencompass && python3 ./run.py ./eval_2k_<ts>.py -r <ts>"
echo ""
echo "  After git pull, re-sync with:"
echo "    bash setup_env.sh --sync-only"
echo ""
