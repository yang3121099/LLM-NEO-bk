#!/usr/bin/env bash
# One-time environment setup for the Shadow-FT RL grafting experiment.
#
#   ./shadow_rl/setup.sh              # core: python deps + Search-R1 checkout
#   ./shadow_rl/setup.sh --with-bm25  # also install pyserini + JDK (search pairs)
#
# Safe to re-run: every step is skipped if already satisfied.
#
# Environment:
#   SEARCH_R1_ROOT  where to clone the harness   (default $HOME/Search-R1)
#   VENV            virtualenv to create/use     (default $HOME/shadow-rl-venv, "" to skip)
#   TORCH_INDEX     torch wheel index            (default CUDA 12.8, right for H200)
set -euo pipefail

SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-$HOME/Search-R1}"
VENV="${VENV-$HOME/shadow-rl-venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
WITH_BM25=0
[[ "${1:-}" == "--with-bm25" ]] && WITH_BM25=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. GPU ---------------------------------------------------------------- #
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPUs detected:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader | sed 's/^/         /'
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    log "$NGPU GPU(s) available"
else
    warn "nvidia-smi not found. The merge and the similarity pass run on CPU,"
    warn "but evaluation needs a GPU."
fi

# ---- 2. python ------------------------------------------------------------- #
PYBIN="${PYTHON:-python3}"
command -v "$PYBIN" >/dev/null || die "$PYBIN not found"
PYVER=$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
log "python $PYVER ($PYBIN)"
"$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)' \
    || die "python >= 3.9 required, found $PYVER"

if [[ -n "$VENV" ]]; then
    if [[ ! -d "$VENV" ]]; then
        log "creating virtualenv at $VENV"
        "$PYBIN" -m venv "$VENV"
    else
        log "reusing virtualenv at $VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PYBIN=python
fi

log "upgrading pip"
$PYBIN -m pip install --quiet --upgrade pip setuptools wheel

# ---- 3. torch -------------------------------------------------------------- #
if $PYBIN -c 'import torch' 2>/dev/null; then
    log "torch already installed: $($PYBIN -c 'import torch; print(torch.__version__)')"
else
    log "installing torch from $TORCH_INDEX"
    $PYBIN -m pip install --quiet torch --index-url "$TORCH_INDEX" \
        || die "torch install failed. Pick the wheel matching your CUDA and retry:
       TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./shadow_rl/setup.sh"
fi

# ---- 4. the rest ----------------------------------------------------------- #
log "installing python dependencies"
$PYBIN -m pip install --quiet \
    safetensors huggingface_hub transformers datasets accelerate requests pandas

# vllm pins its own torch; install it last so it can resolve a consistent pair.
if $PYBIN -c 'import vllm' 2>/dev/null; then
    log "vllm already installed: $($PYBIN -c 'import vllm; print(vllm.__version__)')"
else
    log "installing vllm (this pulls a large wheel, be patient)"
    $PYBIN -m pip install --quiet vllm \
        || die "vllm install failed. See https://docs.vllm.ai/en/latest/getting_started/installation.html"
fi

# ---- 5. Search-R1 harness -------------------------------------------------- #
if [[ -d "$SEARCH_R1_ROOT/.git" ]]; then
    log "Search-R1 already at $SEARCH_R1_ROOT"
else
    log "cloning Search-R1 -> $SEARCH_R1_ROOT"
    git clone --depth 1 https://github.com/PeterGriffinJin/Search-R1.git "$SEARCH_R1_ROOT" \
        || die "clone failed"
fi
[[ -f "$SEARCH_R1_ROOT/verl/utils/reward_score/qa_em.py" ]] \
    || die "qa_em.py missing under $SEARCH_R1_ROOT -- is that really a Search-R1 checkout?"
log "official EM scorer found"

# ---- 6. BM25 (optional, only for the SearchR1-* pairs) --------------------- #
if [[ $WITH_BM25 -eq 1 ]]; then
    if $PYBIN -c 'import pyserini' 2>/dev/null; then
        log "pyserini already installed"
    else
        log "installing pyserini (needs a JDK)"
        if ! command -v java >/dev/null 2>&1; then
            if command -v conda >/dev/null 2>&1; then
                log "installing OpenJDK 21 via conda"
                conda install -y -c conda-forge openjdk=21 maven >/dev/null
            else
                warn "no java and no conda. Install a JDK 21 yourself, then:"
                warn "  pip install pyserini"
            fi
        fi
        $PYBIN -m pip install --quiet pyserini || warn "pyserini install failed; BM25 will not work"
    fi
    $PYBIN -m pip install --quiet fastapi uvicorn || true
fi

# ---- 7. verify ------------------------------------------------------------- #
log "running the CPU test suites"
FAILED=0
for t in test_merge test_similarity; do
    if $PYBIN "shadow_rl/tests/$t.py" >/tmp/shadow_$t.log 2>&1; then
        log "  $t: $(tail -1 /tmp/shadow_$t.log)"
    else
        warn "  $t FAILED -- see /tmp/shadow_$t.log"; FAILED=1
    fi
done
if $PYBIN shadow_rl/tests/test_evaluate.py --search-r1-root "$SEARCH_R1_ROOT" \
        >/tmp/shadow_test_evaluate.log 2>&1; then
    log "  test_evaluate: $(tail -1 /tmp/shadow_test_evaluate.log)"
else
    warn "  test_evaluate FAILED -- see /tmp/shadow_test_evaluate.log"; FAILED=1
fi
[[ $FAILED -eq 0 ]] || die "verification failed; fix the above before running the experiment"

# ---- done ------------------------------------------------------------------ #
cat <<EOF

$(log "setup complete")

  Search-R1 : $SEARCH_R1_ROOT
$( [[ -n "$VENV" ]] && echo "  venv      : $VENV  (source $VENV/bin/activate)" )

Next:

  export SEARCH_R1_ROOT=$SEARCH_R1_ROOT
$( [[ -n "$VENV" ]] && echo "  source $VENV/bin/activate" )

  # start here: the no-search pairs need no retrieval server
  ./shadow_rl/run_all.sh --pairs nosearch

  # everything (needs BM25 + a lot of disk)
  ./shadow_rl/run_all.sh --pairs all --cleanup
EOF
