#!/usr/bin/env bash
# Compatibility shim. The retriever launcher now handles the dense (e5) indexes
# as well, so it is called launch_retriever.sh; this name still works and means
# --retriever bm25.
#
# BM25 is the one option that needs a JVM, via pyserini -> Lucene. If that has
# been giving you trouble, the dense retrievers skip it entirely and are what
# the Search-R1 paper actually used:
#
#   ./shadow_rl/launch_retriever.sh --retriever e5-hnsw   # CPU, faiss-cpu, no Java
#   ./shadow_rl/launch_retriever.sh --retriever e5        # exact, wants faiss-gpu
exec "$(dirname "${BASH_SOURCE[0]}")/launch_retriever.sh" --retriever bm25 "$@"
