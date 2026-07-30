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
#
# The CUDA stack is selected from the GPU generation found on the machine (see
# scripts/gpu_profile.sh). Hopper and older keep the pins the existing results
# were produced with; Blackwell (B200 sm_100, B300 sm_103) gets a newer torch,
# because no torch below 2.7/2.9 has kernels for those devices at all.
#
#   FORCE_GPU_CC=10.3 bash setup_env.sh    # build a B300 env from a CPU node
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

# --- Hardware profile (exports GPU_*, TORCH_*, ATTN_IMPL, VLLM_SPEC) ---
# shellcheck source=scripts/gpu_profile.sh
source "${SCRIPT_DIR}/scripts/gpu_profile.sh"

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
# 3. Core dependencies (per-generation)
###############################################################################
info "Detected hardware:"
gpu_profile_refresh_attn   # now that the conda env's python is on PATH
gpu_profile_summary | sed 's/^/         /'

if gpu_is_blackwell; then
  # torch 2.6.0 has no sm_100/sm_103 cubins, so the Hopper pin below is not an
  # option here. torchvision must come from the SAME index as torch: pip will
  # otherwise take it from PyPI, and a CUDA major mismatch between the two
  # surfaces much later as a bogus "Could not import module 'Qwen2ForCausalLM'".
  info "Installing torch for ${GPU_LABEL}: ${TORCH_SPEC} from ${TORCH_INDEX}"
  pip install -q --index-url "$TORCH_INDEX" "$TORCH_SPEC" torchvision \
    || error "torch install failed. Override the wheel index if this machine
       needs a different CUDA, e.g.
         TORCH_INDEX=https://download.pytorch.org/whl/cu129 bash setup_env.sh"

  # deepspeed JIT-compiles its ops against the installed torch; without an arch
  # list it builds for the arch of whatever GPU it happens to see, or for none.
  export TORCH_CUDA_ARCH_LIST
  info "Installing the rest (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) ..."
  pip install -q \
    transformers==4.52.1 \
    deepspeed \
    peft==0.15.2 \
    importlib_metadata \
    omegaconf
else
  info "Installing core dependencies (legacy ${GPU_FAMILY} stack) ..."
  pip install -q \
    torch==2.6.0 \
    transformers==4.51.2 \
    torchvision \
    deepspeed \
    peft==0.15.2 \
    importlib_metadata \
    omegaconf
fi

###############################################################################
# 4. OpenCompass + eval backends
###############################################################################
info "Installing OpenCompass ..."
cd "$WORKSPACE/opencompass"
if gpu_is_blackwell; then
  # OpenCompass's [vllm] extra asks for a bare `vllm`, which resolves a torch
  # from PyPI — and the default PyPI torch build is not compiled for sm_103.
  # Install OpenCompass without the extra, then vllm with the CUDA-matched index
  # visible so the resolver can keep the torch we just installed.
  pip install -q "."
  info "Installing ${VLLM_SPEC} with ${TORCH_INDEX} as an extra index ..."
  pip install -q --extra-index-url "$TORCH_INDEX" "$VLLM_SPEC" || {
    warn "vllm install failed on ${GPU_LABEL}."
    warn "Blackwell needs a vllm wheel built against CUDA >= 12.9. If PyPI has"
    warn "none, use the NGC container (docker/docker-cuda, BASE_IMAGE pinned to"
    warn "a Blackwell-ready NGC release) or vllm's own wheel index."
  }
else
  pip install -q ".[vllm]"
fi

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

# The failure mode this catches: a torch that imports fine, reports the GPU, and
# then dies on the first matmul with "no kernel image is available for execution
# on the device" because the wheel carries no cubin for this compute capability.
if ! python3 "$WORKSPACE/shadow_rl/check_env.py" >/tmp/setup_env_check.log 2>&1; then
  warn "Environment check reported problems:"
  sed 's/^/         /' /tmp/setup_env_check.log >&2
  warn "Full output: python3 shadow_rl/check_env.py --full"
else
  info "Environment check passed (python3 shadow_rl/check_env.py)"
fi

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
