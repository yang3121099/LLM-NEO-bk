#!/usr/bin/env bash
# Install verl into the working tree — not $HOME, not /root.
#
#   ./shadow_rl/verl_math/setup_verl.sh            # clone + install + verify
#   ./shadow_rl/verl_math/setup_verl.sh --check    # verify only, install nothing
#   VERL_ROOT=/data/verl ./shadow_rl/verl_math/setup_verl.sh
#
# The checkout lands at third_party/verl inside the repo (gitignored) and is
# pip-installed editable with --no-deps. --no-deps is deliberate: verl's
# requirements.txt pins `transformers>=5.5.3,<5.11` and pulls `ray`, `torchdata`
# and friends transitively, and letting pip resolve that set will happily
# reinstall torch from PyPI — the default CUDA build — on top of the cu130 wheel
# vllm is working against. So we install verl's code, then add by hand only the
# packages it actually imports for FSDP + vLLM GRPO.
#
# Nothing here needs VERL_ROOT afterwards: once installed, `import verl` and
# `python -m verl.model_merger` work from any directory. The path is only used
# for the clone itself.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/third_party/verl}"
VERL_REPO="${VERL_REPO:-https://github.com/volcengine/verl.git}"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

log()  { printf '\033[1;34m[verl]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- what verl imports at runtime ------------------------------------------ #
# Derived by importing verl.trainer.main_ppo with --no-deps and adding whatever
# it complained about, not by copying requirements.txt. Deliberately absent:
#   torch, transformers, vllm  — already installed, and the ones that break
#   wandb                      — we log to console (see train_grpo.sh)
#   liger-kernel, peft, flash-attn — optional, not on this recipe's path
DEPS=(
    "ray[default]"      # verl's process model; every worker is a ray actor
    tensordict          # DataProto's storage
    omegaconf
    hydra-core          # the whole CLI is hydra overrides
    codetiming          # marked_timer, imported unconditionally
    dill                # ray serialises the reward function with it
    torchdata           # StatefulDataLoader, for resumable training
    pylatexenc          # math answer normalisation
    datasets            # RLHFDataset is built on it
    pillow              # verl.utils.dataset imports PIL unconditionally
    accelerate          # init_empty_weights, in the checkpoint merger
    pyarrow
    pandas
    tqdm
)

py() { python3 "$@"; }

verify() {
    # Three things have to work, and they fail in different places:
    #   import verl              -> package installed
    #   verl.trainer.main_ppo    -> the runtime deps are all present
    #   verl.model_merger        -> export_hf.py can convert the checkpoint
    # stderr is discarded on purpose: verl logs accelerator warnings there on a
    # CPU-only box ("Platform 'nvidia' is registered but not available"), and
    # they arrive *before* the verdict, so anything that reads the first line of
    # a merged stream calls a healthy install broken. The python below reports
    # both outcomes on stdout, so nothing is lost.
    local out
    out="$(py - 2>/dev/null <<'PY'
import sys
try:
    import verl
    from verl.trainer import main_ppo                        # noqa: F401
    # base_model_merger, not the package __init__: the package imports fine
    # with accelerate missing, and only the CLI that export_hf.py calls
    # actually breaks. Check the thing that has to work.
    from verl.model_merger import base_model_merger          # noqa: F401
except Exception as exc:                       # noqa: BLE001
    name = getattr(exc, "name", None)
    print(f"MISSING\t{name or ''}\t{type(exc).__name__}: {exc}")
    sys.exit(1)
print(f"OK\t{getattr(verl, '__version__', '?')}\t{verl.__file__}")
PY
)"
    out="$(grep -m1 -E '^(OK|MISSING)	' <<< "$out")"
    if [[ "$out" == OK* ]]; then
        ok "verl $(cut -f2 <<< "$out") — $(cut -f3 <<< "$out")"
        return 0
    fi
    VERIFY_DETAIL="$(cut -f3- <<< "$out")"
    VERIFY_MODULE="$(cut -f2 <<< "$out")"
    return 1
}

if [[ $CHECK_ONLY -eq 1 ]]; then
    verify && exit 0
    warn "verl is not importable: ${VERIFY_DETAIL:-unknown}"
    echo "  fix: ./shadow_rl/verl_math/setup_verl.sh" >&2
    exit 1
fi

# ---- 1. checkout ----------------------------------------------------------- #
if [[ -d "$VERL_ROOT/.git" ]]; then
    ok "checkout already at $VERL_ROOT"
elif [[ -e "$VERL_ROOT" ]]; then
    die "$VERL_ROOT exists but is not a git checkout; move it aside or set VERL_ROOT"
else
    log "cloning verl -> $VERL_ROOT"
    mkdir -p "$(dirname "$VERL_ROOT")"
    git clone --depth 1 "$VERL_REPO" "$VERL_ROOT" || die "clone failed"
fi

# ---- 2. the package, without its dependency resolution --------------------- #
log "pip install -e $VERL_ROOT --no-deps"
py -m pip install -e "$VERL_ROOT" --no-deps || die "editable install failed"

# ---- 3. the dependencies it actually needs --------------------------------- #
# --no-deps skipped pip's resolver, which also means it skipped verl's version
# ranges. Several of them matter: tensordict is pinned to <=0.10.0, and a bare
# `pip install -U tensordict` happily installs 0.13. So read the specifiers back
# off the freshly installed distribution and apply them by hand. Packages verl
# does not constrain come through unpinned.
mapfile -t PINNED < <(py - "${DEPS[@]}" <<'PY'
import sys
from importlib.metadata import requires

from packaging.requirements import Requirement

declared = {}
for line in requires("verl") or []:
    try:
        req = Requirement(line)
    except Exception:  # noqa: BLE001 - a malformed line is not worth dying over
        continue
    if req.marker is None or req.marker.evaluate():
        declared[req.name.lower().replace("_", "-")] = str(req.specifier)

for dep in sys.argv[1:]:
    name = dep.split("[")[0].lower().replace("_", "-")
    print(f"{dep}{declared.get(name, '')}")
PY
)
if [[ ${#PINNED[@]} -ne ${#DEPS[@]} ]]; then
    warn "could not read verl's version ranges; installing unpinned"
    PINNED=("${DEPS[@]}")
fi

log "installing runtime dependencies:"
printf '         %s\n' "${PINNED[@]}"
if ! py -m pip install -U "${PINNED[@]}"; then
    # Distro-packaged wheels have no RECORD, so pip refuses to replace them:
    #   ERROR: Cannot uninstall packaging 24.0, RECORD file not found.
    # Only reachable outside a venv; --ignore-installed leaves the distro copy
    # on disk and shadows it, which is what a venv would have done anyway.
    warn "install failed; retrying with --ignore-installed for distro-owned packages"
    py -m pip install -U --ignore-installed packaging "${PINNED[@]}" \
        || die "dependency install failed"
fi

# ---- 4. verify ------------------------------------------------------------- #
if verify; then
    cat <<EOF

$(ok "verl is ready — no VERL_ROOT export needed")
  checkout   : $VERL_ROOT   (gitignored; only used for the clone)
  next       : ./shadow_rl/verl_math/run_all_math.sh --demo --yes

pip will have printed lines like "verl requires wandb, which is not installed"
and "requires transformers<5.11, but you have ...". Both are expected and were
checked above: those packages are not on this recipe's code path, and the import
test that just passed is the thing that decides whether it runs.
EOF
    exit 0
fi

warn "verl still does not import: ${VERIFY_DETAIL:-unknown}"
if [[ -n "${VERIFY_MODULE:-}" ]]; then
    echo "  the missing module is '$VERIFY_MODULE'; add it to DEPS in $0 and rerun," >&2
    echo "  or just: python3 -m pip install $VERIFY_MODULE" >&2
fi
exit 1
