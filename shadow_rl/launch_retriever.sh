#!/usr/bin/env bash
# Download the 2018 Wikipedia corpus + an index, and serve retrieval on :8000.
# Only the SearchR1-* pairs need this; the R1-* (no-search) pairs do not.
#
#   ./shadow_rl/launch_retriever.sh                     # auto: dense, no Java
#   ./shadow_rl/launch_retriever.sh --retriever e5-hnsw # e5 on CPU, no faiss-gpu
#   ./shadow_rl/launch_retriever.sh --retriever bm25    # sparse, needs a JVM
#   ./shadow_rl/launch_retriever.sh --check             # dependencies only
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
PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"
RETRIEVER="${RETRIEVER:-auto}"
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"
CHECK_ONLY=0
FAISS_GPU="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retriever) RETRIEVER="$2"; shift 2 ;;
        --check)     CHECK_ONLY=1;   shift ;;
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
    # --faiss_gpu calls faiss.index_cpu_to_all_gpus, which only exists in
    # faiss-gpu. Passing it with faiss-cpu installed is an AttributeError at
    # startup, so decide here rather than let the server discover it.
    if python3 -c 'import faiss, sys; sys.exit(0 if hasattr(faiss, "index_cpu_to_all_gpus") else 1)' 2>/dev/null \
       && command -v nvidia-smi >/dev/null 2>&1; then
        FAISS_GPU_AVAILABLE=1
    fi
    if [[ "$FAISS_GPU" == "auto" ]]; then
        FAISS_GPU=$FAISS_GPU_AVAILABLE
    elif [[ "$FAISS_GPU" == "1" && $FAISS_GPU_AVAILABLE -eq 0 ]]; then
        die "--faiss-gpu asked for, but this faiss has no index_cpu_to_all_gpus.
       That comes from faiss-gpu; faiss-cpu cannot do it.
       Either install faiss-gpu, or use --retriever e5-hnsw (CPU, approximate)."
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
        warn "Use --retriever e5-hnsw instead, or install faiss-gpu."
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
if [[ "$RETRIEVER" == bm25 ]]; then
    ARGS+=(--retriever_name bm25)
else
    ARGS+=(--retriever_name e5 --retriever_model "$E5_MODEL")
    [[ "$FAISS_GPU" == "1" ]] && ARGS+=(--faiss_gpu)
fi

log "serving $RETRIEVER on port $PORT (topk=$TOPK)"
cd "$SEARCH_R1_ROOT"
exec python3 search_r1/search/retrieval_server.py "${ARGS[@]}"
