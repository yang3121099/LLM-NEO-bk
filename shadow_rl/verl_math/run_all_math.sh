#!/usr/bin/env bash
# Shadow-FT on RL we train ourselves: Qwen3-4B, GRPO on DAPO-Math.
#
#   ./shadow_rl/verl_math/run_all_math.sh            # everything, in order
#   ./shadow_rl/verl_math/run_all_math.sh --stages merge,eval
#
# Stages
#   data   prepare DAPO-Math train + AIME24/AIME25/AMC23 validation parquets
#   train  GRPO on Qwen3-4B-Base and Qwen3-4B-Instruct, identical recipe
#   export convert both verl checkpoints to HuggingFace format
#   merge  W_shadow = W_I + (RL(W_B) - W_B)
#   eval   all five roles on AIME24 / AIME25 / AMC23
#
# The two training runs are the expensive part (8 GPUs, hours each) and are
# skipped if their checkpoint directory already holds a global_step_*.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
HERE="shadow_rl/verl_math"

STAGES="data,train,export,merge,eval"
K=4
N_GPUS="${N_GPUS:-8}"
TP="${TP:-1}"
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --k)      K="$2";      shift 2 ;;
        --tp)     TP="$2";     shift 2 ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/$HERE/ckpt}"
HF_DIR="$REPO_ROOT/$HERE/hf"
MERGED="$REPO_ROOT/$HERE/merged/qwen3-4b-shadow"
RESULTS="$REPO_ROOT/$HERE/math_results.csv"
BASE="Qwen/Qwen3-4B-Base"
INSTRUCT="Qwen/Qwen3-4B-Instruct"

log()  { printf '\033[1;34m[math]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }
has()  { [[ ",$STAGES," == *",$1,"* ]]; }

trained() { compgen -G "$CKPT_DIR/$1/global_step_*" >/dev/null 2>&1; }

cat <<EOF

==============================================================
 Shadow-FT on self-trained RL — Qwen3-4B, GRPO, DAPO-Math
 stages   : $STAGES
 base     : $BASE
 instruct : $INSTRUCT
 shadow   : $MERGED
 results  : $RESULTS
 gpus     : $N_GPUS (train), tp=$TP (eval)
 avg@k    : $K
==============================================================
EOF
if [[ $ASSUME_YES -eq 0 && -t 0 ]]; then
    read -r -p "proceed? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { log "aborted"; exit 0; }
fi

# ---- data ------------------------------------------------------------------ #
if has data; then
    log "stage: data"
    if [[ -f "datasets/math_val/val.parquet" && -f "datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet" ]]; then
        ok "parquets already present, skipping"
    else
        python3 "$HERE/prepare_data.py" --out datasets || die "data preparation failed"
    fi
fi

# ---- train ----------------------------------------------------------------- #
if has train; then
    for side in base instruct; do
        exp="qwen3-4b-${side}-dapo-math-grpo"
        if trained "$exp"; then
            ok "train: $exp already has checkpoints, skipping"
        else
            log "stage: train ($side) — this is the long one"
            N_GPUS="$N_GPUS" CKPT_DIR="$CKPT_DIR" "$HERE/train_grpo.sh" "$side" \
                || die "training failed for $side"
        fi
    done
fi

# ---- export ---------------------------------------------------------------- #
if has export; then
    for side in base instruct; do
        exp="qwen3-4b-${side}-dapo-math-grpo"
        src="$CKPT_DIR/$exp"
        dst="$HF_DIR/qwen3-4b-${side}-rl"
        orig=$([[ "$side" == base ]] && echo "$BASE" || echo "$INSTRUCT")
        if [[ -f "$dst/config.json" ]]; then
            ok "export: $dst already present, skipping"
        elif [[ -d "$src" ]]; then
            log "stage: export ($side)"
            python3 "$HERE/export_hf.py" --ckpt "$src" --out "$dst" --base "$orig" \
                || die "export failed for $side"
        else
            die "no checkpoint at $src — run the train stage first"
        fi
    done
fi

# ---- merge ----------------------------------------------------------------- #
if has merge; then
    if [[ -f "$MERGED/shadow_merge_stats.json" ]]; then
        ok "merge: $MERGED already complete, skipping"
    else
        log "stage: merge — W_shadow = W_I + (RL(W_B) - W_B)"
        python3 shadow_rl/merge.py \
            --base "$BASE" --instruct "$INSTRUCT" \
            --rl-base "$HF_DIR/qwen3-4b-base-rl" \
            --rl-instruct "$HF_DIR/qwen3-4b-instruct-rl" \
            --out "$MERGED" || die "merge failed"
        python3 shadow_rl/smoke_test.py --model "$MERGED" \
            || die "merged model failed the smoke test; not evaluating it"
    fi
fi

# ---- eval ------------------------------------------------------------------ #
if has eval; then
    log "stage: eval (avg@$K on AIME24 / AIME25 / AMC23)"
    declare -A MODELS=(
        [base_baseline]="$BASE"
        [instruct_baseline]="$INSTRUCT"
        [rl_on_base]="$HF_DIR/qwen3-4b-base-rl"
        [rl_on_instruct]="$HF_DIR/qwen3-4b-instruct-rl"
        [shadow]="$MERGED"
    )
    # Same order as the search track, so the two reports read alike.
    for role in base_baseline instruct_baseline rl_on_instruct rl_on_base shadow; do
        model="${MODELS[$role]}"
        if grep -q "^$role," "$RESULTS" 2>/dev/null; then
            ok "eval: $role already in $RESULTS, skipping"
            continue
        fi
        log "  $role  <- $model"
        python3 "$HERE/eval_math.py" --model "$model" --role "$role" \
            --out "$RESULTS" --k "$K" --tensor-parallel-size "$TP" \
            || warn "eval failed for $role; continuing"
    done
    python3 "$HERE/report_math.py" --results "$RESULTS"
fi

ok "done — $RESULTS"
