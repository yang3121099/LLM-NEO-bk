#!/usr/bin/env bash
###############################################################################
# train_grpo.sh — one GRPO run through verl, for one side of the pair.
#
#   ./verl_rl/train_grpo.sh base        # RL on Qwen3-4B-Base   -> RL(W_B)
#   ./verl_rl/train_grpo.sh instruct    # RL on Qwen3-4B        -> RL(W_I)
#
# Both sides run the identical recipe -- same data, same seed, same batch sizes,
# same number of steps. That is the whole premise of the comparison: if the two
# runs differed in anything but the starting weights, neither the RL(W_B) vs
# RL(W_I) result nor the graft would mean anything.
#
# Every setting lives in config.sh. Hydra overrides can be appended per-run:
#   ./verl_rl/train_grpo.sh base actor_rollout_ref.actor.optim.lr=5e-7
###############################################################################
set -euo pipefail

ROLE="${1:-}"
shift || true
case "$ROLE" in
  base|instruct) ;;
  *) echo "usage: $0 <base|instruct> [extra hydra overrides...]" >&2; exit 2 ;;
esac

# shellcheck source=config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

info() { echo -e "\033[1;32m[train]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn ]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[fail ]\033[0m $*" >&2; exit 1; }

if [[ "$ROLE" == "base" ]]; then
  MODEL_PATH="$BASE_MODEL"
else
  MODEL_PATH="$INSTRUCT_MODEL"
fi

RUN_NAME="${PAIR_NAME}-${ROLE}"
ROLE_CKPT_DIR="$CKPT_DIR/$ROLE"
mkdir -p "$ROLE_CKPT_DIR" "$LOG_DIR"

TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/val.parquet"
[[ -f "$TRAIN_FILE" ]] || die "no training data at $TRAIN_FILE -- run verl_rl/prepare_data.py first"
[[ -f "$REWARD_FN_PATH" ]] || die "reward function not found at $REWARD_FN_PATH"

python3 -c 'import verl' 2>/dev/null \
  || die "verl is not installed in this environment -- run ./verl_rl/setup.sh"

###############################################################################
# Attention backend
###############################################################################
# verl's `use_remove_padding` path packs a variable-length batch and calls
# flash-attn's varlen kernels directly, so it is not optional-with-fallback: no
# usable flash-attn means that path has to be off. ATTN_IMPL comes from
# scripts/gpu_profile.sh, which probes by *calling* flash_attn on this GPU --
# on a B300 a wheel built for sm_100a will import happily and then have no
# kernels.
ATTN_ARGS=()
if [[ "${ATTN_IMPL:-fa2}" == "fa2" ]]; then
  ATTN_ARGS+=("actor_rollout_ref.model.use_remove_padding=True")
else
  warn "flash-attn is not usable on ${GPU_LABEL}; falling back to sdpa."
  warn "Sequence packing is disabled with it, which costs throughput on a"
  warn "batch of mixed lengths. Build flash-attn for sm_${GPU_SM} to get it back:"
  warn "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST pip install flash-attn --no-build-isolation"
  ATTN_ARGS+=("actor_rollout_ref.model.use_remove_padding=False")
  # Hydra `+` adds a key that may not exist in this verl version's schema. If
  # your verl rejects it, drop it here and set the model's attn_implementation
  # in its config.json instead.
  ATTN_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=sdpa")
fi

###############################################################################
# Step budget
###############################################################################
STEP_ARGS=()
if [[ "${TOTAL_STEPS:-0}" -gt 0 ]]; then
  STEP_ARGS+=("trainer.total_training_steps=$TOTAL_STEPS")
fi

info "role         : $ROLE"
info "model        : $MODEL_PATH"
info "hardware     : $GPU_LABEL x$N_GPUS"
info "attention    : ${ATTN_IMPL:-fa2}"
info "data         : $TRAIN_FILE"
info "checkpoints  : $ROLE_CKPT_DIR"
info "batch        : $TRAIN_BATCH_SIZE prompts x $ROLLOUT_N samples, mini=$PPO_MINI_BATCH_SIZE"

set -x
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESPONSE_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type="$KL_LOSS_TYPE" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEF" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_UTIL" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="$REWARD_FN_PATH" \
  custom_reward_function.name="$REWARD_FN_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.project_name="$PAIR_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$ROLE_CKPT_DIR" \
  trainer.logger='[console]' \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "${STEP_ARGS[@]+"${STEP_ARGS[@]}"}" \
  "${ATTN_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_DIR/train_${ROLE}.log"
set +x

info "done: $ROLE"
info "export it with:  python3 verl_rl/export_hf.py --role $ROLE"
