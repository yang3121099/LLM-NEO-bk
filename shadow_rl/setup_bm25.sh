#!/usr/bin/env bash
# Install and verify everything BM25 retrieval needs: a JDK, faiss, pyserini,
# and a JAVA_HOME that survives the next shell.
#
#   ./shadow_rl/setup_bm25.sh              # install what is missing, verify, persist
#   ./shadow_rl/setup_bm25.sh --check      # verify only, change nothing
#   ./shadow_rl/setup_bm25.sh --no-persist # do not touch the environment's activate hook
#
# Only the SearchR1-* pairs need this. The R1-* (no-search) pairs do not.
#
# Three things go wrong here, over and over, and each fails far from its cause:
#
#   1. faiss-cpu and pyserini get installed into a different interpreter than
#      the one that launches the retrieval server, which then dies on
#      "No module named 'faiss'" despite pip having said Successfully installed.
#      Fixed by installing with $PYBIN -m pip, and by reporting which python
#      that is.
#
#   2. JAVA_HOME points at a directory with no libjvm.so in it. pyserini loads
#      Lucene through jnius, which dlopens $JAVA_HOME/lib/server/libjvm.so.
#      conda's openjdk lives at $CONDA_PREFIX/lib/jvm, so that path gets tried
#      even in an environment where the JDK actually came from apt. The symptom
#      is "JVM failed to start: -1" or a dlopen error, an hour into a run.
#      Fixed by choosing a JAVA_HOME that contains libjvm.so, not one that
#      merely exists.
#
#   3. JAVA_HOME is exported by a setup script and then lost, so the next shell
#      is back to the broken one. Fixed by writing it into the environment's own
#      activation hook.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/shadow_rl/paths.sh"

PYBIN="${PYTHON:-python3}"
CHECK_ONLY=0
PERSIST=1
for arg in "$@"; do
    case "$arg" in
        --check)      CHECK_ONLY=1 ;;
        --no-persist) PERSIST=0 ;;
        *) echo "usage: $0 [--check] [--no-persist]" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;34m[bm25]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

log "python: $($PYBIN -c 'import sys; print(sys.executable)')"

# ---- 1. a JDK with a real libjvm.so ---------------------------------------- #
if ! JAVA_HOME_FOUND="$(shadow_rl_java_home)"; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
        die "no JDK with a usable libjvm.so. Run without --check to install one."
    fi
    if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
        log "installing openjdk-21-jdk-headless (apt)"
        apt-get update -qq && apt-get install -y -qq openjdk-21-jdk-headless \
            || warn "apt install failed"
    elif command -v conda >/dev/null 2>&1; then
        log "installing openjdk 21 (conda-forge)"
        conda install -y -c conda-forge openjdk=21 maven >/dev/null \
            || warn "conda install failed"
    else
        die "no JDK, and neither root apt nor conda is available. Install one:
       apt-get install -y openjdk-21-jdk-headless
       conda install -y -c conda-forge openjdk=21"
    fi
    JAVA_HOME_FOUND="$(shadow_rl_java_home)" \
        || die "still no libjvm.so after installing. Check the install log above."
fi

export JAVA_HOME="$JAVA_HOME_FOUND"
ok "JAVA_HOME=$JAVA_HOME"
ok "  libjvm: $(ls "$JAVA_HOME"/lib/server/libjvm.so "$JAVA_HOME"/lib/*/server/libjvm.so 2>/dev/null | head -1)"
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib/jvm" && "$JAVA_HOME" != "$CONDA_PREFIX"* ]]; then
    # This is the trap: the directory jnius reaches for first is present but
    # useless, so say plainly that it is being passed over.
    warn "$CONDA_PREFIX/lib/jvm exists but has no libjvm.so — ignoring it."
    warn "That directory is what jnius would have picked by default; it is the"
    warn "cause of 'dlopen(.../lib/jvm/lib/server/libjvm.so): No such file'."
fi

# ---- 2. faiss + pyserini, in *this* interpreter ----------------------------- #
missing=$($PYBIN - <<'PY'
import importlib
out = []
for m in ("faiss", "pyserini", "jnius"):
    try:
        importlib.import_module(m)
    except Exception:  # noqa: BLE001
        out.append(m)
print(" ".join(out))
PY
)
if [[ -n "$missing" ]]; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
        die "missing in $($PYBIN -c 'import sys; print(sys.executable)'): $missing
       $PYBIN -m pip install -U faiss-cpu pyserini"
    fi
    log "installing faiss-cpu + pyserini (missing: $missing)"
    # Same interpreter, always. `pip install` from a shell whose PATH points at
    # another python is the single most common way to get here.
    $PYBIN -m pip install -U faiss-cpu pyserini fastapi uvicorn \
        || $PYBIN -m pip install -U --ignore-installed packaging \
               faiss-cpu pyserini fastapi uvicorn \
        || die "install failed"
fi

# ---- 3. verify: boot the JVM, do not just import ---------------------------- #
# Importing pyserini succeeds without a JVM; the JVM starts on first use, which
# is where the run would otherwise die.
$PYBIN - <<'PY' || die "verification failed"
import os
import sys

import faiss  # noqa: F401
import pyserini  # noqa: F401

try:
    from jnius import autoclass

    version = autoclass("java.lang.System").getProperty("java.version")
except Exception as exc:  # noqa: BLE001
    print(f"[fail] the JVM would not start: {exc}", file=sys.stderr)
    print(f"       JAVA_HOME={os.environ.get('JAVA_HOME', '(unset)')}", file=sys.stderr)
    sys.exit(1)

print(f"[ok]   JVM starts, java {version}")
print(f"[ok]   faiss + pyserini import in {sys.executable}")
PY

# ---- 4. make JAVA_HOME survive the next shell ------------------------------- #
# Exporting it here dies with this process. Without persisting it, the next
# shell inherits conda's broken default again and the whole problem returns.
if [[ $PERSIST -eq 1 && $CHECK_ONLY -eq 0 ]]; then
    HOOK=""
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        HOOK="$CONDA_PREFIX/etc/conda/activate.d/zz-shadow-rl-java.sh"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        HOOK="$VIRTUAL_ENV/bin/activate"
    fi
    if [[ -n "$HOOK" ]]; then
        mkdir -p "$(dirname "$HOOK")"
        if grep -q "shadow-rl JAVA_HOME" "$HOOK" 2>/dev/null; then
            log "activation hook already set: $HOOK"
        else
            {
                echo ""
                echo "# shadow-rl JAVA_HOME: pyserini dlopens \$JAVA_HOME/lib/server/libjvm.so,"
                echo "# and the default here has none. Remove this block to undo."
                echo "export JAVA_HOME=\"$JAVA_HOME\""
            } >> "$HOOK"
            ok "JAVA_HOME persisted in $HOOK"
            log "  (delete that block to undo; --no-persist skips this step)"
        fi
    else
        warn "no conda env or virtualenv active, so JAVA_HOME was not persisted."
        warn "Add this to your shell profile:"
        warn "  export JAVA_HOME=$JAVA_HOME"
    fi
fi

cat <<EOF

$(ok "BM25 dependencies are ready")
  java     : $JAVA_HOME
  python   : $($PYBIN -c 'import sys; print(sys.executable)')
  next     : ./shadow_rl/launch_bm25_retriever.sh      # downloads ~70GB, then serves :8000
             ./shadow_rl/run_all.sh --pairs search --auto-retriever --yes
EOF
