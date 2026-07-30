#!/usr/bin/env bash
###############################################################################
# run_all.sh — data -> GRPO on both sides -> export -> graft -> evaluate.
#
#   ./verl_rl/run_all.sh                     # everything
#   ./verl_rl/run_all.sh --demo              # 200 synthetic rows, 5 steps
#   ./verl_rl/run_all.sh --dry-run           # print the plan and stop
#   ./verl_rl/run_all.sh --stages merge,eval # resume from a stage
#   ./verl_rl/run_all.sh --skip-eval         # stop after the merge
#
# Resumable: a stage whose output already exists is skipped unless --force.
###############################################################################
set -euo pipefail

STAGES="data,train,export,merge,eval"
DRY_RUN=0
FORCE=0
DEMO=0
YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)    STAGES="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --force)     FORCE=1; shift ;;
    --demo)      DEMO=1; shift ;;
    --skip-eval) STAGES="data,train,export,merge"; shift ;;
    --yes|-y)    YES=1; shift ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *)           echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

info() { echo -e "\033[1;32m[run  ]\033[0m $*"; }
step() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }
warn() { echo -e "\033[1;33m[warn ]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[fail ]\033[0m $*" >&2; exit 1; }

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

if [[ $DEMO -eq 1 ]]; then
  # Smallest thing that exercises every stage end to end. Not a result:
  # 5 steps on synthetic arithmetic tells you the plumbing works, nothing more.
  TOTAL_STEPS="${TOTAL_STEPS_DEMO:-5}"
  TRAIN_BATCH_SIZE=16
  PPO_MINI_BATCH_SIZE=8
  ROLLOUT_N=4
  SAVE_FREQ=5
  TEST_FREQ=5
  MAX_RESPONSE_LEN=512
  EVAL_SUITE=fast
fi

mkdir -p "$WORK_DIR" "$LOG_DIR"

###############################################################################
# Plan
###############################################################################
cat <<EOF

  pair        : $PAIR_NAME
  W_B         : $BASE_MODEL
  W_I         : $INSTRUCT_MODEL
  hardware    : $GPU_LABEL x$N_GPUS${GPU_MEM_GB:+, ${GPU_MEM_GB}GB/GPU}
  attention   : ${ATTN_IMPL:-fa2}
  data        : $( [[ $DEMO -eq 1 ]] && echo "demo (200 synthetic rows)" || echo "$RL_DATASET" )
  GRPO        : ${TRAIN_BATCH_SIZE} prompts x ${ROLLOUT_N} samples, lr=$LEARNING_RATE, kl=$KL_LOSS_COEF
              : $( [[ "${TOTAL_STEPS:-0}" -gt 0 ]] && echo "${TOTAL_STEPS} steps" || echo "${TOTAL_EPOCHS} epoch(s)" )
  eval        : suite=$EVAL_SUITE backend=$EVAL_BACKEND
  stages      : $STAGES
  work dir    : $WORK_DIR

  Two GRPO runs on the identical recipe (base-init and instruct-init), then
  W_shadow = W_I + (RL(W_B) - W_B), then all five roles on the same benchmarks.

EOF

if [[ $DRY_RUN -eq 1 ]]; then
  info "dry run: nothing executed"
  exit 0
fi

if [[ $YES -eq 0 && -t 0 ]]; then
  read -r -p "  proceed? [y/N] " reply
  [[ "${reply,,}" == "y" ]] || { info "aborted"; exit 0; }
fi

START=$(date +%s)

###############################################################################
# 1. data
###############################################################################
if has_stage data; then
  step "data"
  if [[ -f "$DATA_DIR/train.parquet" && $FORCE -eq 0 ]]; then
    info "reusing $DATA_DIR/train.parquet"
  else
    ARGS=(--out "$DATA_DIR" --dataset "$RL_DATASET" --val-size "$VAL_SIZE")
    [[ "${TRAIN_SIZE:-0}" -gt 0 ]] && ARGS+=(--train-size "$TRAIN_SIZE")
    [[ $DEMO -eq 1 ]] && ARGS+=(--demo)
    python3 "$VERL_RL_DIR/prepare_data.py" "${ARGS[@]}"
  fi
fi

###############################################################################
# 2. train — both sides, same recipe
###############################################################################
if has_stage train; then
  for ROLE in base instruct; do
    step "GRPO on $ROLE"
    if [[ -d "$CKPT_DIR/$ROLE" ]] && compgen -G "$CKPT_DIR/$ROLE/global_step_*" >/dev/null \
       && [[ $FORCE -eq 0 ]]; then
      info "checkpoints already exist for $ROLE (use --force to retrain)"
      continue
    fi
    TOTAL_STEPS="$TOTAL_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" ROLLOUT_N="$ROLLOUT_N" \
    SAVE_FREQ="$SAVE_FREQ" TEST_FREQ="$TEST_FREQ" MAX_RESPONSE_LEN="$MAX_RESPONSE_LEN" \
      "$VERL_RL_DIR/train_grpo.sh" "$ROLE" \
      || die "GRPO failed on $ROLE -- see $LOG_DIR/train_${ROLE}.log"
  done
fi

###############################################################################
# 3. export
###############################################################################
if has_stage export; then
  for ROLE in base instruct; do
    step "export $ROLE"
    ARGS=(--role "$ROLE")
    [[ $FORCE -eq 1 ]] && ARGS+=(--force)
    python3 "$VERL_RL_DIR/export_hf.py" "${ARGS[@]}" \
      || die "export failed for $ROLE"
  done
fi

###############################################################################
# 4. merge — the Shadow-FT graft
###############################################################################
if has_stage merge; then
  step "Shadow-FT graft"
  if [[ -f "$MERGED_DIR/shadow_merge_stats.json" && $FORCE -eq 0 ]]; then
    info "reusing $MERGED_DIR (merge.py writes its stats last, so this is complete)"
  else
    "$VERL_RL_DIR/merge_shadow.sh" || die "merge failed"
  fi
fi

###############################################################################
# 5. evaluate
###############################################################################
if has_stage eval; then
  step "evaluation"
  EVAL_CFG="$REPO_ROOT/opencompass/eval_${PAIR_NAME//-/_}.py"
  EVAL_SUITE="$EVAL_SUITE" python3 "$VERL_RL_DIR/make_eval_config.py" \
      --out "$EVAL_CFG" --suite "$EVAL_SUITE" --backend "$EVAL_BACKEND" \
    || die "could not write the eval config"

  info "running OpenCompass (this is the long part)"
  ( cd "$REPO_ROOT/opencompass" && python3 ./run.py "./$(basename "$EVAL_CFG")" ) \
      2>&1 | tee "$LOG_DIR/eval.log" \
    || die "evaluation failed -- see $LOG_DIR/eval.log"
fi

###############################################################################
ELAPSED=$(( $(date +%s) - START ))
step "done in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
cat <<EOF

  RL(W_B)   $EXPORT_DIR/base
  RL(W_I)   $EXPORT_DIR/instruct
  W_shadow  $MERGED_DIR
  results   $REPO_ROOT/opencompass/outputs/$PAIR_NAME/

  The number that answers the question is W_shadow vs RL(W_I): does grafting the
  base-side GRPO update onto the instruct backbone beat running the same GRPO on
  the instruct model directly?
EOF
