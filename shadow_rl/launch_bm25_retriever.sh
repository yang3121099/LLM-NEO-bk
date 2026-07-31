#!/usr/bin/env bash
# Download the 2018 Wikipedia corpus + prebuilt BM25 index and serve them on :8000.
# Only needed for the SearchR1-* pairs; the R1-* (no-search) pairs do not use this.
#
#   ./shadow_rl/launch_bm25_retriever.sh
#
# BM25 is chosen over the paper's dense E5 index for speed: it needs no GPU and no
# embedding pass. Absolute EM on the search pairs therefore sits below the published
# numbers. All four roles are served by the same index, so the comparison between
# them -- which is what the experiment measures -- stays fair.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The corpus and index are ~70GB; keep them under the working tree rather than
# in $HOME, where they tend to fill a small root volume.
CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus}"
source "$REPO_ROOT/shadow_rl/paths.sh"
SEARCH_R1_ROOT="$(shadow_rl_search_r1_root)"
PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"

if [[ ! -d "$SEARCH_R1_ROOT" ]]; then
    echo "[fail] Search-R1 not found at $SEARCH_R1_ROOT" >&2
    echo "       ./shadow_rl/setup_search_r1.sh" >&2
    exit 1
fi

# ---- JVM ------------------------------------------------------------------- #
# shadow_rl_java_home (paths.sh) returns a JAVA_HOME that actually contains
# libjvm.so, which is what jnius dlopens; see the comment there for why the
# obvious candidate is often the wrong one.
if JH="$(shadow_rl_java_home)"; then
    export JAVA_HOME="$JH"
    echo "[ok]   JAVA_HOME=$JAVA_HOME"
else
    echo "[fail] no usable JDK found: nothing with lib/server/libjvm.so under" >&2
    echo "       \$JAVA_HOME, \$CONDA_PREFIX, \$(which java) or /usr/lib/jvm/*." >&2
    echo "       apt-get install -y openjdk-21-jdk-headless" >&2
    echo "       # or, without root:  conda install -y -c conda-forge openjdk=21" >&2
    exit 1
fi

# ---- python dependencies, in the interpreter that will run the server ------ #
# "but I installed faiss-cpu" is almost always true and almost always in a
# different interpreter, so check with the one that matters and say which it is.
python3 - <<'PY' || exit 1
import importlib
import os
import sys

missing = []
for module, install in (("faiss", "pip install faiss-cpu"),
                        ("pyserini", "pip install pyserini")):
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        missing.append((module, install, exc))

if missing:
    print(f"[fail] interpreter: {sys.executable}", file=sys.stderr)
    for module, install, exc in missing:
        print(f"       {module}: {exc}", file=sys.stderr)
        print(f"       fix: {install}", file=sys.stderr)
    print("       If you have already installed these, they went to a different\n"
          "       interpreter than the one above -- activate that environment,\n"
          "       or install with the same python:\n"
          f"         {sys.executable} -m pip install faiss-cpu pyserini", file=sys.stderr)
    sys.exit(1)

# Importing pyserini is not enough: the JVM only starts on first use, which is
# where "JVM failed to start: -1" and the dlopen error appear. Boot it now, via
# jnius directly -- pyserini's own search modules drag in unrelated optional
# dependencies whose failures would be misreported here as a Java problem.
try:
    from jnius import autoclass

    autoclass("java.lang.String")
except Exception as exc:  # noqa: BLE001
    text = f"{exc}".lower()
    print(f"[fail] the JVM would not start: {exc}", file=sys.stderr)
    print(f"       JAVA_HOME={os.environ.get('JAVA_HOME', '(unset)')}", file=sys.stderr)
    if "jvm" in text or "dlopen" in text or "libjvm" in text:
        print("       That JAVA_HOME has no working libjvm.so. Install a JDK and\n"
              "       let this script pick it up, or point JAVA_HOME at one:\n"
              "         apt-get install -y openjdk-21-jdk-headless\n"
              "         conda install -y -c conda-forge openjdk=21", file=sys.stderr)
    sys.exit(1)

print(f"[ok]   faiss + pyserini + JVM ready ({sys.executable})")
PY

# --check stops here: everything past this point downloads or serves.
[[ "${1:-}" == "--check" ]] && exit 0

mkdir -p "$CORPUS_DIR"

# ---- corpus --------------------------------------------------------------- #
if [[ ! -f "$CORPUS_DIR/wiki-18.jsonl" ]]; then
    echo "[step] downloading PeterJinGo/wiki-18-corpus (~10GB unpacked)"
    python3 - "$CORPUS_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="PeterJinGo/wiki-18-corpus", filename="wiki-18.jsonl.gz",
                repo_type="dataset", local_dir=sys.argv[1])
PY
    echo "[step] decompressing"
    gzip -dk "$CORPUS_DIR/wiki-18.jsonl.gz"
else
    echo "[skip] corpus already at $CORPUS_DIR/wiki-18.jsonl"
fi

# ---- index ---------------------------------------------------------------- #
if [[ ! -d "$CORPUS_DIR/bm25" ]]; then
    echo "[step] downloading PeterJinGo/wiki-18-bm25-index"
    python3 - "$CORPUS_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="PeterJinGo/wiki-18-bm25-index", repo_type="dataset",
                  local_dir=sys.argv[1])
PY
    # The release ships the index as a tarball in some revisions and as a plain
    # directory in others; normalise both to $CORPUS_DIR/bm25.
    if [[ ! -d "$CORPUS_DIR/bm25" ]]; then
        TARBALL=$(find "$CORPUS_DIR" -maxdepth 1 -name 'bm25*.tar*' | head -1)
        if [[ -n "$TARBALL" ]]; then
            echo "[step] extracting $TARBALL"
            tar -xf "$TARBALL" -C "$CORPUS_DIR"
        fi
    fi
    [[ -d "$CORPUS_DIR/bm25" ]] || {
        echo "[fail] no bm25 index directory under $CORPUS_DIR after download." >&2
        echo "       Inspect the repo layout and point --index_path at it manually." >&2
        exit 1
    }
else
    echo "[skip] index already at $CORPUS_DIR/bm25"
fi

echo "[step] serving BM25 on port $PORT (topk=$TOPK)"
cd "$SEARCH_R1_ROOT"
exec python3 search_r1/search/retrieval_server.py \
    --index_path "$CORPUS_DIR/bm25" \
    --corpus_path "$CORPUS_DIR/wiki-18.jsonl" \
    --topk "$TOPK" \
    --retriever_name bm25
