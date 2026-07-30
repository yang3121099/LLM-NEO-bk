#!/usr/bin/env bash
###############################################################################
# gpu_profile.sh — detect the GPU generation and export the stack that matches.
#
# The pipeline used to be pinned to one machine shape: H100 (sm_90), torch
# 2.6.0 from the cu126-era wheels, `--flash_attn fa2`. Every one of those three
# is wrong on a B300 (Blackwell Ultra, sm_103):
#
#   * torch 2.6.0 ships no sm_100/sm_103 cubins at all, so the first matmul dies
#     with "no kernel image is available for execution on the device";
#   * cu126/cu128 wheels predate sm_103 support in the CUDA toolkit (12.9+);
#   * FlashAttention-2 wheels built for Hopper have no Blackwell kernels, and
#     anything built arch-conditionally for sm_100a does NOT run on sm_103.
#
# Rather than fork the scripts per machine, everything sources this file and
# reads the exported variables.
#
# Usage:
#   source scripts/gpu_profile.sh          # export GPU_* and the stack vars
#   bash   scripts/gpu_profile.sh --print  # human-readable summary
#   bash   scripts/gpu_profile.sh --json   # machine-readable summary
#   bash   scripts/gpu_profile.sh --var TORCH_INDEX   # one value, for $(...)
#
# Overrides (useful on a CPU login node, or to pin a stack by hand):
#   FORCE_GPU_CC=10.3            pretend the device is a B300
#   FORCE_GPU_NAME="NVIDIA B300" name-based detection input
#   FORCE_GPU_COUNT=8            skip nvidia-smi for the device count
#   FORCE_GPU_MEM_GB=279         skip nvidia-smi for the per-GPU memory
#   TORCH_INDEX / TORCH_CUDA_ARCH_LIST / ATTN_IMPL — respected if already set
#
# Exported:
#   GPU_NAME GPU_COUNT GPU_MEM_GB GPU_CC GPU_SM GPU_FAMILY GPU_LABEL
#   TORCH_INDEX TORCH_SPEC TORCH_CUDA_ARCH_LIST ATTN_IMPL
#   VLLM_SPEC CUDA_TOOLKIT_OK MICRO_BS_HINT
#
# This file is sourced by scripts that run under `set -euo pipefail`, so it must
# never exit non-zero and must never dereference an unset variable.
###############################################################################

# ---------------------------------------------------------------------------- #
# 0. detection
# ---------------------------------------------------------------------------- #

# Compute capability of device 0, as reported by the driver. `compute_cap` needs
# a reasonably recent driver; when it is missing we fall back to the model name,
# which is the case that matters on clusters running an older driver branch.
_gp_query() {
    # Callers run under `set -euo pipefail`, where a missing nvidia-smi would
    # otherwise take the whole pipeline down with 127 instead of leaving the
    # field empty. Every failure here has to become "unknown", not an exit.
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null || true; } \
        | head -1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

_gp_cc_from_name() {
    # Only the families this repo is actually run on. Unknown names stay empty
    # so the caller reports "unknown" instead of silently picking a wrong stack.
    local name_lc
    name_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$name_lc" in
        *b300*|*gb300*)          echo "10.3" ;;
        *b200*|*gb200*|*b100*)   echo "10.0" ;;
        *rtx*pro*|*rtx\ 50*|*5090*) echo "12.0" ;;
        *h100*|*h200*|*gh200*|*h800*) echo "9.0" ;;
        *l40*|*l4*|*4090*)       echo "8.9" ;;
        *a100*|*a800*)           echo "8.0" ;;
        *)                       echo "" ;;
    esac
}

GPU_NAME="${FORCE_GPU_NAME:-$(_gp_query name)}"
GPU_CC="${FORCE_GPU_CC:-$(_gp_query compute_cap)}"
GPU_MEM_GB=""
GPU_COUNT="${FORCE_GPU_COUNT:-}"

if [[ -z "$GPU_COUNT" ]]; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
fi
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || GPU_COUNT=0

if [[ -z "$GPU_CC" && -n "$GPU_NAME" ]]; then
    GPU_CC="$(_gp_cc_from_name "$GPU_NAME")"
fi

if [[ -n "${FORCE_GPU_MEM_GB:-}" ]]; then
    GPU_MEM_GB="$FORCE_GPU_MEM_GB"
else
    _gp_mem_mb="$(_gp_query memory.total)"
    if [[ "$_gp_mem_mb" =~ ^[0-9]+$ ]]; then
        GPU_MEM_GB=$(( _gp_mem_mb / 1024 ))
    fi
    unset _gp_mem_mb
fi

# sm number as the toolchain spells it: 9.0 -> 90, 10.3 -> 103.
if [[ "$GPU_CC" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    GPU_SM="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
else
    GPU_SM=""
fi

case "$GPU_CC" in
    10.3) GPU_FAMILY="blackwell-ultra" ; GPU_LABEL="Blackwell Ultra (B300/GB300, sm_103)" ;;
    10.0) GPU_FAMILY="blackwell"       ; GPU_LABEL="Blackwell (B200/GB200, sm_100)" ;;
    12.0) GPU_FAMILY="blackwell-rtx"   ; GPU_LABEL="Blackwell workstation (sm_120)" ;;
    9.0)  GPU_FAMILY="hopper"          ; GPU_LABEL="Hopper (H100/H200, sm_90)" ;;
    8.9)  GPU_FAMILY="ada"             ; GPU_LABEL="Ada Lovelace (sm_89)" ;;
    8.0|8.6) GPU_FAMILY="ampere"       ; GPU_LABEL="Ampere (A100, sm_${GPU_SM})" ;;
    *)    GPU_FAMILY="unknown"         ; GPU_LABEL="unknown GPU${GPU_CC:+ (cc $GPU_CC)}" ;;
esac

# Blackwell and later: the whole point of this file.
gpu_is_blackwell() { [[ -n "$GPU_SM" && "$GPU_SM" -ge 100 ]]; }

# ---------------------------------------------------------------------------- #
# 1. CUDA toolkit capability
# ---------------------------------------------------------------------------- #
# sm_103 codegen needs CUDA >= 12.9. Building flash-attn or any CUTLASS-based
# kernel with an older nvcc silently produces a wheel that cannot run here, so
# check up front rather than after a 40-minute build.
CUDA_TOOLKIT_OK="unknown"
_gp_nvcc_knows() {
    command -v nvcc >/dev/null 2>&1 || return 1
    nvcc --list-gpu-arch 2>/dev/null | grep -qx "compute_$1"
}
if command -v nvcc >/dev/null 2>&1; then
    if [[ -z "$GPU_SM" ]] || _gp_nvcc_knows "$GPU_SM"; then
        CUDA_TOOLKIT_OK="yes"
    else
        CUDA_TOOLKIT_OK="no"
    fi
fi

# ---------------------------------------------------------------------------- #
# 2. the stack
# ---------------------------------------------------------------------------- #
# TORCH_INDEX  wheel index to install torch and its companions from. They must
#              all come from the same index; a torch/torchvision CUDA major
#              mismatch surfaces much later as a bogus "Could not import module
#              'Qwen2ForCausalLM'" (see shadow_rl/check_env.py).
# TORCH_SPEC   pip requirement. Legacy archs keep the exact pin the H100 runs
#              were done with, so those results stay reproducible. Blackwell
#              gets a floor instead: no released torch below these supports the
#              arch, and pinning an exact version would rot immediately.
case "$GPU_FAMILY" in
    blackwell-ultra)
        _gp_index="https://download.pytorch.org/whl/cu130"
        _gp_torch="torch>=2.9.0"
        _gp_arch="10.3a"
        _gp_vllm="vllm>=0.11.0"
        ;;
    blackwell|blackwell-rtx)
        _gp_index="https://download.pytorch.org/whl/cu128"
        _gp_torch="torch>=2.7.0"
        _gp_arch="$( [[ "$GPU_FAMILY" == blackwell-rtx ]] && echo "12.0a" || echo "10.0a" )"
        _gp_vllm="vllm>=0.9.0"
        ;;
    *)
        # Hopper and older: the stack the existing results were produced with.
        _gp_index="https://download.pytorch.org/whl/cu126"
        _gp_torch="torch==2.6.0"
        _gp_arch="9.0"
        [[ "$GPU_FAMILY" == ampere ]] && _gp_arch="8.0"
        [[ "$GPU_FAMILY" == ada ]] && _gp_arch="8.9"
        # Unconstrained, as before: every released vllm has Hopper kernels.
        _gp_vllm="vllm"
        ;;
esac

# An sm_103 device with a CUDA 12.8-or-older toolkit cannot get arch-conditional
# sm_103a kernels. Plain sm_100 cubins still run (minor-version compatibility
# inside the 10.x family), so fall back to those and say why: the FP8/NVFP4
# CUTLASS paths that require `a` kernels will be missing.
if [[ "$GPU_FAMILY" == "blackwell-ultra" && "$CUDA_TOOLKIT_OK" == "no" ]]; then
    _gp_arch="10.0"
fi

TORCH_INDEX="${TORCH_INDEX:-$_gp_index}"
TORCH_SPEC="${TORCH_SPEC:-$_gp_torch}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$_gp_arch}"
VLLM_SPEC="${VLLM_SPEC:-$_gp_vllm}"
unset _gp_index _gp_torch _gp_arch _gp_vllm

# ---------------------------------------------------------------------------- #
# 3. attention implementation
# ---------------------------------------------------------------------------- #
# `--flash_attn fa2` is right on Hopper and a coin flip on Blackwell: it needs
# flash-attn >= 2.8 (first release with sm_100 kernels) AND that wheel to carry
# kernels for this exact sm. Import-and-call is the only honest test, so ask the
# installed package instead of guessing from a version number alone. When in
# doubt use sdpa — on Blackwell that lands on the cuDNN attention backend, which
# is fast and always present.
_gp_probe_flash_attn() {
    local py="${PYTHON:-python3}"
    command -v "$py" >/dev/null 2>&1 || return 1
    "$py" - <<'PY' >/dev/null 2>&1
import sys
try:
    import flash_attn
    from packaging.version import Version
except Exception:
    sys.exit(1)
if Version(flash_attn.__version__.split("+")[0]) < Version("2.8.0"):
    sys.exit(1)
try:
    import torch
    from flash_attn import flash_attn_func
except Exception:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
# The kernels are selected at call time; a missing cubin only shows up here.
q = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16, device="cuda")
try:
    flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
except Exception:
    sys.exit(1)
PY
}

_gp_pick_attn() {
    if ! gpu_is_blackwell; then
        echo "fa2"
    elif _gp_probe_flash_attn; then
        echo "fa2"
    else
        echo "sdpa"
    fi
}

[[ -n "${ATTN_IMPL:-}" ]] || ATTN_IMPL="$(_gp_pick_attn)"

# The probe above asks the *current* interpreter. Callers that source this file
# before activating their env (setup_env.sh does, to read TORCH_INDEX) should
# call this once the right python is on PATH.
gpu_profile_refresh_attn() {
    ATTN_IMPL="$(_gp_pick_attn)"
    export ATTN_IMPL
}

# ---------------------------------------------------------------------------- #
# 4. micro-batch hint
# ---------------------------------------------------------------------------- #
# A B300 carries ~279 GB of HBM3e against an H100's 80 GB, so the
# per_device_train_batch_size=1 / grad_accum=256 shape the H100 runs used leaves
# most of the card idle. Callers that use this must scale grad_accum down by the
# same factor to keep the effective batch size — and therefore the optimisation
# trajectory — unchanged.
if [[ -n "$GPU_MEM_GB" ]] && gpu_is_blackwell; then
    if   (( GPU_MEM_GB >= 240 )); then MICRO_BS_HINT=4
    elif (( GPU_MEM_GB >= 160 )); then MICRO_BS_HINT=2
    else                               MICRO_BS_HINT=1
    fi
else
    MICRO_BS_HINT=1
fi

export GPU_NAME GPU_COUNT GPU_MEM_GB GPU_CC GPU_SM GPU_FAMILY GPU_LABEL
export TORCH_INDEX TORCH_SPEC TORCH_CUDA_ARCH_LIST ATTN_IMPL VLLM_SPEC
export CUDA_TOOLKIT_OK MICRO_BS_HINT

# ---------------------------------------------------------------------------- #
# 5. reporting
# ---------------------------------------------------------------------------- #
gpu_profile_summary() {
    echo "GPU profile"
    echo "  device        : ${GPU_NAME:-<none detected>}"
    echo "  generation    : $GPU_LABEL"
    echo "  count         : $GPU_COUNT"
    echo "  memory/GPU    : ${GPU_MEM_GB:-?} GB"
    echo "  torch         : $TORCH_SPEC  from $TORCH_INDEX"
    echo "  arch list     : $TORCH_CUDA_ARCH_LIST"
    echo "  attention     : $ATTN_IMPL"
    echo "  vllm          : $VLLM_SPEC"
    echo "  nvcc knows sm_${GPU_SM:-?} : $CUDA_TOOLKIT_OK"
    echo "  micro-batch   : $MICRO_BS_HINT"
    if [[ "$GPU_FAMILY" == "unknown" ]]; then
        echo
        echo "  WARNING: GPU generation not recognised; falling back to the legacy"
        echo "           (Hopper) stack. Set FORCE_GPU_CC=<major.minor> to pick one."
    fi
    if [[ "$GPU_FAMILY" == "blackwell-ultra" && "$CUDA_TOOLKIT_OK" == "no" ]]; then
        echo
        echo "  WARNING: nvcc here cannot target sm_103 (needs CUDA >= 12.9)."
        echo "           Using sm_100 kernels, which run but lose the sm_103a"
        echo "           FP8/NVFP4 CUTLASS paths. Upgrade the toolkit to build"
        echo "           flash-attn or any other kernel from source."
    fi
}

gpu_profile_json() {
    printf '{'
    printf '"name":"%s",'        "$GPU_NAME"
    printf '"cc":"%s",'          "$GPU_CC"
    printf '"sm":"%s",'          "$GPU_SM"
    printf '"family":"%s",'      "$GPU_FAMILY"
    printf '"count":%s,'         "${GPU_COUNT:-0}"
    printf '"memory_gb":%s,'     "${GPU_MEM_GB:-0}"
    printf '"torch_spec":"%s",'  "$TORCH_SPEC"
    printf '"torch_index":"%s",' "$TORCH_INDEX"
    printf '"arch_list":"%s",'   "$TORCH_CUDA_ARCH_LIST"
    printf '"attn_impl":"%s",'   "$ATTN_IMPL"
    printf '"vllm_spec":"%s",'   "$VLLM_SPEC"
    printf '"toolkit_ok":"%s",'  "$CUDA_TOOLKIT_OK"
    printf '"micro_bs":%s'       "$MICRO_BS_HINT"
    printf '}\n'
}

# Executed rather than sourced: print and exit.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:---print}" in
        --print) gpu_profile_summary ;;
        --json)  gpu_profile_json ;;
        --var)   printf '%s\n' "${!2:-}" ;;
        *)       echo "usage: $0 [--print|--json|--var NAME]" >&2; exit 2 ;;
    esac
fi
