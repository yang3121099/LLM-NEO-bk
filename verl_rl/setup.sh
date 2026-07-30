#!/usr/bin/env bash
###############################################################################
# setup.sh — install verl and its RL stack, for the GPU actually in this box.
#
#   ./verl_rl/setup.sh                    # verl + ray + deps
#   ./verl_rl/setup.sh --with-flash-attn   # also build flash-attn for this arch
#
# Assumes torch is already installed and usable (setup_env.sh does that, and
# picks the right one per GPU generation). This checks that before installing
# anything on top: verl pulls vLLM, vLLM pins a torch, and letting pip resolve
# that against a torch with no kernels for this device produces an environment
# that installs cleanly and dies at the first rollout.
###############################################################################
set -euo pipefail

WITH_FLASH_ATTN=0
[[ "${1:-}" == "--with-flash-attn" ]] && WITH_FLASH_ATTN=1

# shellcheck source=config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

info() { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn ]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[fail ]\033[0m $*" >&2; exit 1; }

PYBIN="${PYTHON:-python3}"

info "hardware:"
gpu_profile_summary | sed 's/^/         /'

###############################################################################
# 1. torch has to work here before anything is layered on it
###############################################################################
if ! $PYBIN -c 'import torch' 2>/dev/null; then
    die "torch is not installed. Run: bash setup_env.sh"
fi
if ! $PYBIN "$REPO_ROOT/shadow_rl/check_env.py" --arch-only; then
    die "torch cannot run kernels on this GPU (output above).
       Fix that first -- bash setup_env.sh reinstalls the matching build."
fi

###############################################################################
# 2. verl
###############################################################################
VERL_SPEC="${VERL_SPEC:-verl}"
if $PYBIN -c 'import verl' 2>/dev/null; then
    info "verl already installed: $($PYBIN -c 'import verl; print(getattr(verl,"__version__","?"))')"
else
    PIP_ARGS=()
    # On Blackwell the resolver has to be able to see the CUDA-matched torch, or
    # it drags in the PyPI default build, which is not compiled for sm_103.
    gpu_is_blackwell && PIP_ARGS+=(--extra-index-url "$TORCH_INDEX")
    info "installing $VERL_SPEC"
    $PYBIN -m pip install "${PIP_ARGS[@]+"${PIP_ARGS[@]}"}" "$VERL_SPEC" \
        || die "verl install failed. From source instead:
       git clone https://github.com/volcengine/verl && pip install -e ./verl"
fi

info "installing the rest (ray, parquet, math checking)"
$PYBIN -m pip install -q ray pandas pyarrow "datasets" \
    latex2sympy2_extended math_verify \
    || warn "some optional deps failed; math_verify only affects reward precision"

###############################################################################
# 3. vLLM, which verl uses for rollout
###############################################################################
if $PYBIN -c 'import vllm' 2>/dev/null; then
    info "vllm: $($PYBIN -c 'import vllm; print(vllm.__version__)')"
else
    PIP_ARGS=()
    gpu_is_blackwell && PIP_ARGS+=(--extra-index-url "$TORCH_INDEX")
    info "installing $VLLM_SPEC"
    $PYBIN -m pip install "${PIP_ARGS[@]+"${PIP_ARGS[@]}"}" "$VLLM_SPEC" \
        || die "vllm install failed. On Blackwell you need a wheel built against
       CUDA >= 12.9; see docs/B300_PRODUCTION.md."
fi

###############################################################################
# 4. flash-attn (optional, and on Blackwell it has to be built here)
###############################################################################
# verl's sequence-packing path calls flash-attn's varlen kernels directly, so a
# usable flash-attn is worth real throughput. A prebuilt wheel for sm_103 may not
# exist; building takes a while and needs nvcc to know the arch.
if [[ $WITH_FLASH_ATTN -eq 1 ]]; then
    if [[ "$CUDA_TOOLKIT_OK" == "no" ]]; then
        die "nvcc here cannot target sm_${GPU_SM} (needs CUDA >= 12.9), so this
       build would produce a wheel that does not run. Upgrade the toolkit or
       use the NGC container (docker/docker-cuda)."
    fi
    info "building flash-attn for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (slow)"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="${MAX_JOBS:-8}" \
        $PYBIN -m pip install flash-attn --no-build-isolation \
        || warn "flash-attn build failed; training will run with sdpa and no packing"
fi

###############################################################################
# 5. verify
###############################################################################
info "verifying"
$PYBIN -c "
import verl, torch
print('  verl:  ', getattr(verl, '__version__', '?'))
print('  torch: ', torch.__version__)
print('  GPUs:  ', torch.cuda.device_count())
" || die "verl does not import after install"

$PYBIN "$VERL_RL_DIR/reward_math.py" --self-test >/dev/null \
    && info "reward function self-test passed" \
    || die "reward function self-test failed"

gpu_profile_refresh_attn
info "attention for training: $ATTN_IMPL"
if [[ "$ATTN_IMPL" != "fa2" ]]; then
    warn "no usable flash-attn: verl will run without sequence packing."
    warn "Re-run with --with-flash-attn to build one for this arch."
fi

cat <<EOF

$(info "setup complete")

Next:

  # 1. data (add --demo for a 200-row synthetic set that needs no download)
  python3 verl_rl/prepare_data.py --out "$DATA_DIR"

  # 2. the whole thing
  ./verl_rl/run_all.sh

  # or one stage at a time
  ./verl_rl/train_grpo.sh base
  ./verl_rl/train_grpo.sh instruct
  python3 verl_rl/export_hf.py --role base
  python3 verl_rl/export_hf.py --role instruct
  ./verl_rl/merge_shadow.sh
EOF
