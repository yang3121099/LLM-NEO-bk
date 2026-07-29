#!/usr/bin/env bash
# Merge and evaluate one pair end to end: the four model roles over the seven QA sets.
#
#   ./shadow_rl/run_pair.sh <pair_id> [extra args passed to evaluate.py]
#
# Environment:
#   SEARCH_R1_ROOT   path to a Search-R1 checkout            (default ~/Search-R1)
#   MERGED_DIR       where merged models are written         (default shadow_rl/merged)
#   RESULTS          results.csv path                        (default shadow_rl/results.csv)
#   TP               tensor parallel size                    (default 1)
#   RETRIEVER_URL    retrieval endpoint for the search pairs (default http://127.0.0.1:8000/retrieve)
#   ROLES            roles to evaluate                       (default all four)
#
# Run `python shadow_rl/pairs.py` for the list of pair ids.
set -euo pipefail

PAIR="${1:-}"
if [[ -z "$PAIR" ]]; then
    echo "usage: $0 <pair_id> [evaluate.py args...]" >&2
    python3 shadow_rl/pairs.py >&2
    exit 1
fi
shift || true

SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-$HOME/Search-R1}"
MERGED_DIR="${MERGED_DIR:-shadow_rl/merged}"
RESULTS="${RESULTS:-shadow_rl/results.csv}"
TP="${TP:-1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
ROLES="${ROLES:-instruct_baseline rl_on_instruct rl_on_base shadow}"

if [[ ! -d "$SEARCH_R1_ROOT" ]]; then
    echo "[fail] SEARCH_R1_ROOT=$SEARCH_R1_ROOT does not exist." >&2
    echo "       git clone https://github.com/PeterGriffinJin/Search-R1.git $SEARCH_R1_ROOT" >&2
    exit 1
fi

# Pull the pair's checkpoint ids out of the manifest so they live in one place.
read -r BASE INSTRUCT RL_BASE RL_INSTRUCT < <(
    python3 - "$PAIR" <<'PY'
import sys, os
sys.path.insert(0, os.path.join("shadow_rl"))
from pairs import PAIRS_BY_ID
p = PAIRS_BY_ID[sys.argv[1]]
print(p.base, p.instruct, p.repo("rl_on_base"), p.repo("rl_on_instruct"))
PY
)

SHADOW_DIR="$MERGED_DIR/$PAIR"

echo "=============================================================="
echo " pair      : $PAIR"
echo " base      : $BASE"
echo " instruct  : $INSTRUCT"
echo " RL(base)  : $RL_BASE"
echo " RL(instr) : $RL_INSTRUCT"
echo " shadow    : $SHADOW_DIR"
echo "=============================================================="

# ---- 1. merge (skipped if already present) -------------------------------- #
if [[ -f "$SHADOW_DIR/config.json" ]]; then
    echo "[skip] merged model already at $SHADOW_DIR"
else
    echo "[step] merging W_shadow = W_I + (RL(W_B) - W_B)"
    python3 shadow_rl/merge.py \
        --base "$BASE" \
        --instruct "$INSTRUCT" \
        --rl-base "$RL_BASE" \
        --rl-instruct "$RL_INSTRUCT" \
        --out "$SHADOW_DIR"
fi

# ---- 2. smoke test: does it load and generate coherent text? --------------- #
echo "[step] smoke-testing the merged model"
python3 shadow_rl/smoke_test.py --model "$SHADOW_DIR"

# ---- 3. evaluate the four roles ------------------------------------------- #
for ROLE in $ROLES; do
    echo
    echo "[step] evaluating role=$ROLE"
    EXTRA=()
    [[ "$ROLE" == "shadow" ]] && EXTRA=(--model-path "$SHADOW_DIR")
    python3 shadow_rl/evaluate.py \
        --search-r1-root "$SEARCH_R1_ROOT" \
        --pair "$PAIR" \
        --role "$ROLE" \
        --out "$RESULTS" \
        --retriever-url "$RETRIEVER_URL" \
        --tensor-parallel-size "$TP" \
        "${EXTRA[@]}" "$@"
done

# ---- 4. refresh FINDINGS.md ----------------------------------------------- #
python3 shadow_rl/aggregate.py --results "$RESULTS" --out shadow_rl/FINDINGS.md
echo
echo "[done] $PAIR -> $RESULTS, shadow_rl/FINDINGS.md"
