#!/usr/bin/env bash
# Shadow-FT on RL we train ourselves: Qwen3-4B, GRPO on DAPO-Math.
#
#   ./shadow_rl/verl_math/run_all_math.sh            # everything, in order
#   ./shadow_rl/verl_math/run_all_math.sh --demo --yes   # tiny end-to-end check
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
DEMO=0
detect_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
    else
        echo 0
    fi
}
N_GPUS="${N_GPUS:-$(detect_gpus)}"
# A 4B model fits on one GPU, so tensor parallelism buys nothing for eval; the
# GPUs are better spent on training width.
TP="${TP:-1}"
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --k)      K="$2";      shift 2 ;;
        --tp)     TP="$2";     shift 2 ;;
        --demo)   DEMO=1;      shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# --demo: enough steps and data to exercise every stage end to end in well under
# an hour. The resulting numbers are meaningless -- 8 optimiser steps will not
# move a 4B model -- but every moving part runs, including the merge and the eval.
LIMIT_TRAIN=""
LIMIT_VAL=""
TRAIN_EXTRA=()
LOW_GPU_NOTE=""
if [[ $DEMO -eq 1 ]]; then
    LIMIT_TRAIN=512
    LIMIT_VAL=8
    TRAIN_EXTRA=(trainer.total_training_steps=8
                 trainer.save_freq=4
                 trainer.test_freq=4
                 data.max_response_length=1024
                 actor_rollout_ref.rollout.n=4)
    [[ "$K" == "4" ]] && K=1
fi

# On few GPUs a 4B actor plus Adam states plus a vLLM engine will not fit
# without help: bf16 weights are ~8GB but fp32 optimiser state is ~32GB, and the
# rollout engine wants its own pool. Offload the parameters and optimiser to
# host memory and leave vLLM a smaller share. Costs throughput, not correctness,
# and it applies to both sides equally so the comparison is unaffected.
if [[ "$N_GPUS" -le 2 && "$N_GPUS" -ge 1 && ",$STAGES," == *",train,"* ]]; then
    TRAIN_EXTRA+=(actor_rollout_ref.actor.fsdp_config.param_offload=True
                  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
                  actor_rollout_ref.ref.fsdp_config.param_offload=True
                  actor_rollout_ref.rollout.gpu_memory_utilization=0.4)
    LOW_GPU_NOTE="offloading params+optimiser to host RAM ($N_GPUS GPU)"
fi

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

if has train && [[ "$N_GPUS" -lt 1 ]]; then
    die "no GPU visible; the train stage needs at least one"
fi
if has eval && [[ "$N_GPUS" -lt 1 ]]; then
    die "no GPU visible; the eval stage needs at least one"
fi
# Check verl up front rather than after the data stage: preparing the parquets
# takes a while, and discovering there that the trainer will not import is a
# waste of it. Only train and export need verl; merge and eval do not.
if has train || has export; then
    if ! "$HERE/setup_verl.sh" --check >/dev/null 2>&1; then
        # Keep the diagnosis, drop setup_verl.sh's own "fix:" line — we print a
        # better one right below it.
        VERL_MSG="$("$HERE/setup_verl.sh" --check 2>&1 | grep -m1 'not importable')"
        die "verl is needed by the '$STAGES' stages but is not usable.
       ${VERL_MSG:-import failed}
       Install it into the working tree with:
         ./shadow_rl/verl_math/setup_verl.sh"
    fi
fi

cat <<EOF

==============================================================
 Shadow-FT on self-trained RL — Qwen3-4B, GRPO, DAPO-Math
 stages   : $STAGES
 base     : $BASE
 instruct : $INSTRUCT
 shadow   : $MERGED
 results  : $RESULTS
 gpus     : $N_GPUS visible — $N_GPUS for training, tp=$TP for eval
 avg@k    : $K
 memory   : ${LOW_GPU_NOTE:-default (no offload)}
 mode     : $( [[ $DEMO -eq 1 ]] && echo "DEMO (8 steps, 512 prompts, 8 problems/benchmark — numbers are not meaningful)" || echo "full" )
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
        PREP=(--out datasets)
        [[ -n "$LIMIT_TRAIN" ]] && PREP+=(--limit-train "$LIMIT_TRAIN")
        [[ -n "$LIMIT_VAL" ]] && PREP+=(--limit-val "$LIMIT_VAL")
        python3 "$HERE/prepare_data.py" "${PREP[@]}" || die "data preparation failed"
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
            N_GPUS="$N_GPUS" CKPT_DIR="$CKPT_DIR" \
                "$HERE/train_grpo.sh" "$side" "${TRAIN_EXTRA[@]}" \
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
        EV=(--model "$model" --role "$role" --out "$RESULTS" --k "$K"
            --tensor-parallel-size "$TP")
        [[ $DEMO -eq 1 ]] && EV+=(--limit 8 --max-tokens 2048 --max-model-len 4096)
        python3 "$HERE/eval_math.py" "${EV[@]}" \
            || warn "eval failed for $role; continuing"
    done
    python3 "$HERE/report_math.py" --results "$RESULTS"
fi

ok "done — $RESULTS"
