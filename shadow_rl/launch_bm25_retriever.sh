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

CORPUS_DIR="${CORPUS_DIR:-$HOME/search_r1_corpus}"
SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-$HOME/Search-R1}"
PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"

if [[ ! -d "$SEARCH_R1_ROOT" ]]; then
    echo "[fail] SEARCH_R1_ROOT=$SEARCH_R1_ROOT does not exist." >&2
    echo "       git clone https://github.com/PeterGriffinJin/Search-R1.git $SEARCH_R1_ROOT" >&2
    exit 1
fi

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

# BM25 retrieval in Search-R1 goes through pyserini, which needs a JVM.
python3 -c "import pyserini" 2>/dev/null || {
    echo "[warn] pyserini not importable. Install it and a JDK first:" >&2
    echo "         conda install -c conda-forge openjdk=21 maven -y && pip install pyserini" >&2
}

echo "[step] serving BM25 on port $PORT (topk=$TOPK)"
cd "$SEARCH_R1_ROOT"
exec python3 search_r1/search/retrieval_server.py \
    --index_path "$CORPUS_DIR/bm25" \
    --corpus_path "$CORPUS_DIR/wiki-18.jsonl" \
    --topk "$TOPK" \
    --retriever_name bm25
