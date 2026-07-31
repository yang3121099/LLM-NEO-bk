#!/usr/bin/env bash
# Download the 2018 Wikipedia corpus + an index, and serve retrieval on :8000.
# Only the SearchR1-* pairs need this; the R1-* (no-search) pairs do not.
#
#   ./shadow_rl/launch_retriever.sh                     # auto: dense, no Java
#   ./shadow_rl/launch_retriever.sh --retriever e5-hnsw # e5 on CPU, no faiss-gpu
#   ./shadow_rl/launch_retriever.sh --retriever bm25    # sparse, needs a JVM
#   ./shadow_rl/launch_retriever.sh --check             # dependencies only
#
#   ./shadow_rl/launch_retriever.sh --daemon            # background, survives logout
#   ./shadow_rl/launch_retriever.sh --status            # is it up? is it answering?
#   ./shadow_rl/launch_retriever.sh --stop
#   ./shadow_rl/launch_retriever.sh --daemon --port 8123    # a specific port
#
# The port defaults to `auto`: the first free one from 8000 upwards. 8000 is a
# popular default -- vLLM's own OpenAI server uses it, as do plenty of dev tools
# -- so binding it blindly either fails or, worse, succeeds against a port some
# other process is about to claim. The chosen URL is written to
# logs/retriever.url, and run_all.sh reads it, so nothing has to be kept in sync
# by hand.
#
# Without --daemon the server runs in the foreground and dies with the shell --
# Ctrl-C, closing the terminal or an SSH drop all take it with them, and the
# next command then says "Connection refused". --daemon detaches it with setsid,
# writes logs/retriever.{pid,log}, and waits until it actually answers before
# returning, so a successful exit means a working retriever.
#
# Which one to use
# ---------------------------------------------------------------------------
#   e5        exact dense match. What the Search-R1 paper used, so the absolute
#             EM numbers are comparable to the published ones. No Java, no
#             pyserini, no JVM. Wants faiss-gpu for the flat index; on faiss-cpu
#             an exhaustive scan of 21M passages per query is impractically slow,
#             which is what e5-hnsw exists for.
#   e5-hnsw   the same encoder against an approximate (HNSW64) index. Runs fine
#             on CPU with plain faiss-cpu. Slightly less accurate than flat,
#             mostly at small topk. Still no Java.
#   bm25      sparse. No GPU and no embedding pass, but it goes through pyserini
#             -> Lucene -> a JVM, which is the single most fragile dependency
#             here. Absolute EM sits below the published numbers.
#   auto      the default: e5 when faiss-gpu is installed and a GPU is visible,
#             e5-hnsw otherwise. Never bm25 -- Java is opt-in now, not a
#             fallback you end up in by accident.
#
# Whichever is chosen, every model role is served by the same index, so the
# comparison the experiment actually measures stays fair. Do not switch
# retrievers half way through a results.csv.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/shadow_rl/paths.sh"
SEARCH_R1_ROOT="$(shadow_rl_search_r1_root)"
# The corpus and index are tens of GB; keep them under the working tree rather
# than in $HOME, where they tend to fill a small root volume.
CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus}"
PORT="${PORT:-auto}"
PORT_BASE="${PORT_BASE:-8000}"
TOPK="${TOPK:-3}"
RETRIEVER="${RETRIEVER:-auto}"
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"
CHECK_ONLY=0
FAISS_GPU="auto"
MODE="run"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/shadow_rl/logs}"
PIDFILE="$LOG_DIR/retriever.pid"
URLFILE="$LOG_DIR/retriever.url"
LOGFILE="$LOG_DIR/retriever.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retriever) RETRIEVER="$2"; shift 2 ;;
        --check)     CHECK_ONLY=1;   shift ;;
        --daemon|-d) MODE="daemon";  shift ;;
        --status)    MODE="status";  shift ;;
        --stop)      MODE="stop";    shift ;;
        --faiss-gpu) FAISS_GPU=1;    shift ;;
        --no-faiss-gpu) FAISS_GPU=0; shift ;;
        --port)      PORT="$2";      shift 2 ;;
        --topk)      TOPK="$2";      shift 2 ;;
        -h|--help)   sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$RETRIEVER" in
    auto|e5|e5-hnsw|bm25) ;;
    *) echo "[fail] --retriever must be auto, e5, e5-hnsw or bm25 (got '$RETRIEVER')" >&2
       exit 2 ;;
esac

log()  { printf '\033[1;34m[retriever]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$LOG_DIR"

port_free() {
    python3 -c '
import socket, sys
s = socket.socket()
free = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0
s.close()
sys.exit(0 if free else 1)' "$1" 2>/dev/null
}

whats_on_port() {   # best effort; ss is not always present or permitted
    command -v ss >/dev/null 2>&1 || return 0
    ss -ltnp 2>/dev/null | awk -v p=":$1\$" '$4 ~ p {print "         " $0}'
}

# --- resolve the port ------------------------------------------------------- #
# status/stop must find the server that is already running, so they read back
# the port it chose rather than guessing.
if [[ "$PORT" == auto && -f "$URLFILE" ]] && [[ "$MODE" == status || "$MODE" == stop ]]; then
    PORT="$(sed 's#.*:\([0-9]*\)/.*#\1#' "$URLFILE")"
fi
if [[ "$PORT" == auto ]]; then
    if [[ "$MODE" == status || "$MODE" == stop ]]; then
        PORT="$PORT_BASE"
    else
        for CANDIDATE in $(seq "$PORT_BASE" $((PORT_BASE + 99))); do
            if port_free "$CANDIDATE"; then PORT="$CANDIDATE"; break; fi
        done
        [[ "$PORT" == auto ]] && die "no free port in $PORT_BASE-$((PORT_BASE + 99))"
        [[ "$PORT" != "$PORT_BASE" ]] \
            && log "port $PORT_BASE is taken; using $PORT instead"
    fi
fi
URL="http://127.0.0.1:$PORT/retrieve"

running_pid() {
    # A pidfile alone is not evidence: the process may have died, or the pid may
    # have been reused by something unrelated. Check that it is alive and that
    # it is the retrieval server.
    [[ -f "$PIDFILE" ]] || return 1
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    grep -q retrieval_server "/proc/$pid/cmdline" 2>/dev/null || return 1
    echo "$pid"
}

answering() {
    curl -fsS --max-time 10 -X POST "$URL" -H 'Content-Type: application/json' \
        -d '{"queries":["ping"],"topk":1,"return_scores":false}' >/dev/null 2>&1
}

case "$MODE" in
    status)
        PID="$(running_pid)" && ok "running, pid $PID" || warn "no retriever process"
        if answering; then
            ok "answering at $URL"
            exit 0
        fi
        warn "not answering at $URL"
        [[ -f "$LOGFILE" ]] && { echo "--- last 15 lines of $LOGFILE ---" >&2
                                 tail -15 "$LOGFILE" >&2; }
        exit 1
        ;;
    stop)
        if PID="$(running_pid)"; then
            log "stopping pid $PID"
            kill "$PID" 2>/dev/null
            for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
            kill -0 "$PID" 2>/dev/null && { warn "still alive, sending KILL"; kill -9 "$PID"; }
            rm -f "$PIDFILE"
            ok "stopped"
        else
            warn "nothing to stop"
        fi
        exit 0
        ;;
esac

if [[ "$MODE" == "daemon" ]] && PID="$(running_pid)"; then
    if answering; then
        ok "already running and answering (pid $PID)"
        exit 0
    fi
    die "a retriever process is running (pid $PID) but not answering.
       ./shadow_rl/launch_retriever.sh --stop, then start it again."
fi

# An explicit --port that something else already owns: fail here with the
# culprit rather than inside uvicorn, or -- worse -- appear to work while the
# evaluation talks to a completely different service.
if [[ $CHECK_ONLY -eq 0 ]] && ! port_free "$PORT"; then
    if answering; then
        ok "a retriever is already answering at $URL"
        echo "$URL" > "$URLFILE"
        exit 0
    fi
    die "port $PORT is in use by something that is not a retriever:
$(whats_on_port "$PORT")
       Pick another:  --port 8123      (or --port auto to choose one)"
fi

[[ -d "$SEARCH_R1_ROOT" ]] \
    || die "Search-R1 not found at $SEARCH_R1_ROOT
       ./shadow_rl/setup_search_r1.sh"

log "retriever: $RETRIEVER"

# ---- dependencies ---------------------------------------------------------- #
# Checked in the interpreter that will run the server, not whichever pip
# happened to be on $PATH -- "I already installed faiss-cpu" is almost always
# true and almost always about a different python.
if [[ "$RETRIEVER" == bm25 ]]; then
    # Only this path needs Java. retrieval_server.py imports pyserini inside
    # BM25Retriever.__init__, so the dense retrievers never touch the JVM.
    if JH="$(shadow_rl_java_home)"; then
        export JAVA_HOME="$JH"
        ok "JAVA_HOME=$JAVA_HOME"
    else
        die "no JDK with a usable libjvm.so.
       ./shadow_rl/setup_bm25.sh
       or avoid Java entirely:  --retriever e5-hnsw"
    fi
    NEEDED="faiss pyserini"
else
    NEEDED="faiss torch transformers datasets"
fi

FAISS_GPU_AVAILABLE=0
python3 - "$RETRIEVER" $NEEDED <<'PY' || die "missing dependencies; see above"
import importlib
import sys

retriever, *modules = sys.argv[1:]
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        missing.append((module, exc))

if missing:
    print(f"[fail] interpreter: {sys.executable}", file=sys.stderr)
    for module, exc in missing:
        print(f"       {module}: {exc}", file=sys.stderr)
    print("       Install them with that same interpreter -- a plain `pip install`\n"
          "       may well be a different python:\n"
          f"         {sys.executable} -m pip install faiss-cpu", file=sys.stderr)
    if retriever == "bm25":
        print(f"         {sys.executable} -m pip install pyserini    "
              "# and a JDK; see ./shadow_rl/setup_bm25.sh", file=sys.stderr)
    sys.exit(1)

if retriever == "bm25":
    # Importing pyserini is not enough: the JVM starts on first use, which is
    # where "JVM failed to start: -1" appears.
    import os

    try:
        from jnius import autoclass

        autoclass("java.lang.System").getProperty("java.version")
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] the JVM would not start: {exc}", file=sys.stderr)
        print(f"       JAVA_HOME={os.environ.get('JAVA_HOME', '(unset)')}", file=sys.stderr)
        print("       ./shadow_rl/setup_bm25.sh, or switch: --retriever e5-hnsw",
              file=sys.stderr)
        sys.exit(1)

print(f"[ok]   dependencies present ({sys.executable})")
PY

if [[ "$RETRIEVER" != bm25 ]]; then
    # Does this faiss actually do GPU? Ask faiss, not the module namespace.
    #
    # hasattr(faiss, "index_cpu_to_all_gpus") is NOT a valid test: faiss-cpu
    # defines it anyway, as a pure-python wrapper in gpu_wrappers.py, so the
    # attribute is present on a build with no GPU support at all. The server
    # then dies on the first genuinely GPU-only symbol it touches:
    #   AttributeError: module 'faiss' has no attribute 'GpuMultipleClonerOptions'
    # get_num_gpus() returns 0 on faiss-cpu and is the honest answer; the two
    # symbols below are the ones retrieval_server.py actually calls.
    if python3 -c '
import sys
import faiss
ok = (getattr(faiss, "get_num_gpus", lambda: 0)() > 0
      and hasattr(faiss, "GpuMultipleClonerOptions")
      and hasattr(faiss, "index_cpu_to_all_gpus"))
sys.exit(0 if ok else 1)' 2>/dev/null; then
        FAISS_GPU_AVAILABLE=1
    fi
    if [[ "$FAISS_GPU" == "auto" ]]; then
        FAISS_GPU=$FAISS_GPU_AVAILABLE
    elif [[ "$FAISS_GPU" == "1" && $FAISS_GPU_AVAILABLE -eq 0 ]]; then
        die "--faiss-gpu asked for, but this faiss has no GPU support
       (faiss.get_num_gpus() == 0). faiss-cpu cannot do it, whatever the module
       namespace suggests. Either install a GPU build -- both provide the same
       'faiss' module, so remove the CPU one first --
         pip uninstall -y faiss-cpu && pip install faiss-gpu
         # older CUDA runtimes: pip install faiss-gpu-cu12
       or use the CPU index:  --retriever e5-hnsw"
    fi
    # Resolve `auto` now that the faiss capability is known: exact flat search
    # if it can run on the GPU, the approximate CPU index otherwise. Picking
    # flat without faiss-gpu would be technically correct and practically
    # unusable, which is not a useful default.
    if [[ "$RETRIEVER" == auto ]]; then
        if [[ "$FAISS_GPU" == "1" ]]; then
            RETRIEVER=e5
            log "auto -> e5 (flat, exact, on GPU)"
        else
            RETRIEVER=e5-hnsw
            log "auto -> e5-hnsw (approximate, CPU; no faiss-gpu here)"
        fi
    fi
    if [[ "$RETRIEVER" == "e5" && "$FAISS_GPU" != "1" ]]; then
        warn "flat e5 index without faiss-gpu: every query scans 21M passages on"
        warn "the CPU. Expect this to be far too slow for a full evaluation."
        warn "Either install a GPU build of faiss and keep this index --"
        warn "  pip uninstall -y faiss-cpu && pip install faiss-gpu"
        warn "or use --retriever e5-hnsw, which is built for CPU search."
    fi
    [[ "$FAISS_GPU" == "1" ]] && ok "faiss GPU support available"
fi

[[ $CHECK_ONLY -eq 1 ]] && { ok "checks passed; nothing downloaded"; exit 0; }

mkdir -p "$CORPUS_DIR"

# ---- corpus (shared by every retriever) ------------------------------------ #
if [[ ! -f "$CORPUS_DIR/wiki-18.jsonl" ]]; then
    log "downloading PeterJinGo/wiki-18-corpus (~10GB unpacked)"
    python3 - "$CORPUS_DIR" <<'PY' || die "corpus download failed"
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="PeterJinGo/wiki-18-corpus", filename="wiki-18.jsonl.gz",
                repo_type="dataset", local_dir=sys.argv[1])
PY
    log "decompressing"
    gzip -dk "$CORPUS_DIR/wiki-18.jsonl.gz" || die "decompression failed"
else
    ok "corpus already at $CORPUS_DIR/wiki-18.jsonl"
fi

# ---- index ----------------------------------------------------------------- #
# The dense releases are split into part_* files because a single 60GB+ file is
# awkward on the Hub; they are concatenated back into one .index.
assemble_parts() {   # <repo> <output file>
    local repo="$1" out="$2"
    if [[ -f "$out" ]]; then
        ok "index already at $out"
        return 0
    fi
    log "downloading $repo (this is the big one; tens of GB)"
    python3 - "$repo" "$CORPUS_DIR" <<'PY' || return 1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], repo_type="dataset",
                        local_dir=sys.argv[2]))
PY
    local parts=("$CORPUS_DIR"/part_*)
    if [[ ! -e "${parts[0]}" ]]; then
        # Some revisions ship the whole file directly.
        local single
        single=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.index' | head -1)
        [[ -n "$single" ]] || return 1
        [[ "$single" == "$out" ]] || mv "$single" "$out"
        return 0
    fi
    log "concatenating $(( ${#parts[@]} )) part(s) -> $(basename "$out")"
    cat "${parts[@]}" > "$out" || return 1
    rm -f "${parts[@]}"
    return 0
}

case "$RETRIEVER" in
    e5)
        INDEX="$CORPUS_DIR/e5_Flat.index"
        assemble_parts "PeterJinGo/wiki-18-e5-index" "$INDEX" \
            || die "could not assemble the e5 flat index under $CORPUS_DIR"
        ;;
    e5-hnsw)
        INDEX="$CORPUS_DIR/e5_HNSW64.index"
        assemble_parts "PeterJinGo/wiki-18-e5-index-HNSW64" "$INDEX" \
            || die "could not assemble the e5 HNSW index under $CORPUS_DIR"
        ;;
    bm25)
        INDEX="$CORPUS_DIR/bm25"
        if [[ ! -d "$INDEX" ]]; then
            log "downloading PeterJinGo/wiki-18-bm25-index"
            python3 - "$CORPUS_DIR" <<'PY' || die "index download failed"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="PeterJinGo/wiki-18-bm25-index", repo_type="dataset",
                  local_dir=sys.argv[1])
PY
            # Shipped as a tarball in some revisions, a plain directory in others.
            if [[ ! -d "$INDEX" ]]; then
                TARBALL=$(find "$CORPUS_DIR" -maxdepth 1 -name 'bm25*.tar*' | head -1)
                [[ -n "$TARBALL" ]] && { log "extracting $TARBALL"; tar -xf "$TARBALL" -C "$CORPUS_DIR"; }
            fi
            [[ -d "$INDEX" ]] || die "no bm25 index directory under $CORPUS_DIR after download"
        else
            ok "index already at $INDEX"
        fi
        ;;
esac

# ---- serve ----------------------------------------------------------------- #
ARGS=(--index_path "$INDEX"
      --corpus_path "$CORPUS_DIR/wiki-18.jsonl"
      --topk "$TOPK")
# retrieval_server.py hardcodes uvicorn's port at 8000, so a different one is
# only honoured if its argparse grew a --port. Check rather than assume.
if grep -q '"--port"' "$SEARCH_R1_ROOT/search_r1/search/retrieval_server.py" 2>/dev/null; then
    ARGS+=(--port "$PORT")
elif [[ "$PORT" != 8000 ]]; then
    SERVE_PORT_OVERRIDE=1
fi
if [[ "$RETRIEVER" == bm25 ]]; then
    ARGS+=(--retriever_name bm25)
else
    ARGS+=(--retriever_name e5 --retriever_model "$E5_MODEL")
    [[ "$FAISS_GPU" == "1" ]] && ARGS+=(--faiss_gpu)
fi

SERVER=(python3 "$SEARCH_R1_ROOT/search_r1/search/retrieval_server.py" "${ARGS[@]}")

# This Search-R1 has no --port, so uvicorn would bind 8000 whatever we asked
# for. Wrap it: the shim imports the server module, then serves its `app` on the
# port we actually chose. Same code, same index, different socket.
if [[ -n "${SERVE_PORT_OVERRIDE:-}" ]]; then
    # The shim works by intercepting uvicorn.run. A server that starts serving
    # some other way would simply block inside the import and never come up, so
    # check before committing to it rather than hanging for the timeout.
    grep -q 'uvicorn\.run' "$SEARCH_R1_ROOT/search_r1/search/retrieval_server.py" \
        || die "this retrieval_server.py neither accepts --port nor calls
       uvicorn.run, so it cannot be moved off port 8000 from the outside.
       Run it on 8000 (--port 8000), or add a --port to your checkout."
    SHIM="$LOG_DIR/serve_on_port.py"
    cat > "$SHIM" <<'PYSHIM'
"""Run Search-R1's retrieval server on a port of our choosing.

    python serve_on_port.py <port> <retrieval_server.py> [server args...]

retrieval_server.py ends with a hardcoded

    uvicorn.run(app, host="0.0.0.0", port=8000)

and its argparse has no --port. Rather than patch a third-party checkout,
replace uvicorn.run for the duration of the import: the module still parses its
arguments, loads the index and builds its FastAPI app, it just does not start
serving. Then serve that same app on the port we were given.
"""
import runpy
import sys

import uvicorn

port = int(sys.argv.pop(1))
server = sys.argv.pop(1)
# argparse reads sys.argv[1:], so what is left must be exactly the server's own
# arguments -- with argv[0] naming the server, for sane --help and error output.
sys.argv[0] = server

real_run = uvicorn.run
captured = {}


def capture(app, *args, **kwargs):
    captured["app"] = app


uvicorn.run = capture
try:
    namespace = runpy.run_path(server, run_name="__main__")
finally:
    uvicorn.run = real_run

app = captured.get("app") or namespace.get("app")
if app is None:
    sys.exit("[fail] retrieval_server.py did not expose a FastAPI app; its layout "
             "may have changed. Run it on port 8000 without this shim.")
print(f"[shim] serving retrieval_server.py's app on port {port}", flush=True)
real_run(app, host="0.0.0.0", port=port)
PYSHIM
    SERVER=(python3 "$SHIM" "$PORT"
            "$SEARCH_R1_ROOT/search_r1/search/retrieval_server.py" "${ARGS[@]}")
    log "serving on port $PORT via a shim (this Search-R1 hardcodes 8000)"
fi

if [[ "$MODE" != "daemon" ]]; then
    log "serving $RETRIEVER on port $PORT (topk=$TOPK)"
    warn "foreground: this dies when the shell does. --daemon to detach."
    cd "$SEARCH_R1_ROOT"
    exec "${SERVER[@]}"
fi

# setsid detaches from the controlling terminal, so an SSH drop or a closed
# terminal does not take the server with it.
log "starting $RETRIEVER in the background (log: $LOGFILE)"
cd "$SEARCH_R1_ROOT"
setsid nohup "${SERVER[@]}" >"$LOGFILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
cd "$REPO_ROOT"

# Loading a 60GB index takes minutes; wait for it rather than returning into a
# race, and fail loudly with the log if the process dies on the way up.
log "waiting for it to answer (loading the index takes a few minutes)"
for i in $(seq 1 180); do
    if answering; then
        echo "$URL" > "$URLFILE"
        ok "retriever up: pid $SERVER_PID, $URL"
        log "  stop it with: ./shadow_rl/launch_retriever.sh --stop"
        log "  verify:       python3 shadow_rl/check_retriever.py"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        rm -f "$PIDFILE"
        echo "--- last 25 lines of $LOGFILE ---" >&2
        tail -25 "$LOGFILE" >&2
        die "the retriever exited while starting up"
    fi
    sleep 10
    [[ $((i % 6)) -eq 0 ]] && log "  still loading ($((i / 6))m)..."
done
die "not answering after 30m. See $LOGFILE"
