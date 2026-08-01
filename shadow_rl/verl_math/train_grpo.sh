#!/usr/bin/env bash
# verl GRPO on DAPO-Math-17k-Processed, for one side of a Shadow-FT pair.
#
#   ./shadow_rl/verl_math/train_grpo.sh base       # Qwen3-4B-Base
#   ./shadow_rl/verl_math/train_grpo.sh instruct   # Qwen3-4B-Instruct
#
# Both sides must be trained with an identical recipe -- that is the whole
# premise of the comparison. Everything below is shared; only the model path and
# the experiment name differ, so do not tune one side without the other.
#
# verl is used as an installed package, so there is no checkout path to export;
# ./shadow_rl/verl_math/setup_verl.sh puts it under third_party/ and pip-installs
# it editable. Everything runs from the repo root, which is what makes the
# relative parquet and reward-function paths below work.
#
# Environment:
#   DATA_DIR      prepared parquet files       (default $PWD/datasets)
#   CKPT_DIR      where checkpoints are written(default $PWD/shadow_rl/verl_math/ckpt)
#   N_GPUS        GPUs per node                (default: all visible)
set -euo pipefail

SIDE="${1:-}"
shift || true
# Anything after the side is appended verbatim to the verl command line, so a
# demo run can cap the step count without a separate script:
#   ./train_grpo.sh base trainer.total_training_steps=4
EXTRA=("$@")
case "$SIDE" in
    base)     MODEL="Qwen/Qwen3-4B-Base" ;;
    instruct) MODEL="Qwen/Qwen3-4B-Instruct" ;;
    *) echo "usage: $0 {base|instruct}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/shadow_rl/verl_math/ckpt}"
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
[[ "$N_GPUS" -lt 1 ]] && { echo "[fail] no GPU visible; training needs at least one" >&2; exit 1; }
REWARD_FN="$REPO_ROOT/shadow_rl/verl_math/math_reward.py"
EXPERIMENT="qwen3-4b-${SIDE}-dapo-math-grpo"
# verl defaults to ["console","wandb"], which needs wandb installed and logged
# in; a run that is otherwise fine then dies at step 0 on an auth prompt. Console
# only unless asked. LOGGER='[console,wandb]' to opt back in.
LOGGER="${LOGGER:-[console]}"

TRAIN="$DATA_DIR/DAPO-Math-17k-Processed/DAPO-Math.parquet"
VAL="$DATA_DIR/math_val/val.parquet"          # AIME24 + AIME25 + AMC23, concatenated

./shadow_rl/verl_math/setup_verl.sh --check >/dev/null 2>&1 \
    || { echo "[fail] verl is not installed. Run: ./shadow_rl/verl_math/setup_verl.sh" >&2; exit 1; }
[[ -f "$TRAIN" ]] || { echo "[fail] $TRAIN missing. Run: python shadow_rl/verl_math/prepare_data.py" >&2; exit 1; }
[[ -f "$VAL" ]]   || { echo "[fail] $VAL missing. Run: python shadow_rl/verl_math/prepare_data.py" >&2; exit 1; }
[[ -f "$REWARD_FN" ]] || { echo "[fail] $REWARD_FN missing" >&2; exit 1; }

# verl 0.9 moved the reward configuration under a `reward` group. The old
# top-level `custom_reward_function` and `reward_model` keys still exist, but on
# the main_ppo path nothing reads them any more -- verl/trainer/ppo/reward.py
# looks only at config.reward.custom_reward_function. Passing the legacy key is
# accepted without complaint and then silently ignored, so training scores with
# the default reward manager instead of math_reward.py. Nothing errors; the
# reward curve is just wrong, which is far worse. Ask the installed verl which
# schema it has rather than guessing.
mapfile -t REWARD_ARGS < <(python3 - "$REWARD_FN" <<'PY'
import os
import sys

import verl

cfg = os.path.join(os.path.dirname(verl.__file__),
                   "trainer", "config", "reward", "reward.yaml")
prefix = "reward." if os.path.exists(cfg) else ""
print(f"{prefix}custom_reward_function.path={sys.argv[1]}")
print(f"{prefix}custom_reward_function.name=compute_score")
print(f"{prefix}reward_model.enable=False")
PY
)
[[ ${#REWARD_ARGS[@]} -eq 3 ]] \
    || { echo "[fail] could not determine verl's reward config schema" >&2; exit 1; }

mkdir -p "$CKPT_DIR/$EXPERIMENT"
echo "=============================================================="
echo " side       : $SIDE"
echo " model      : $MODEL"
echo " experiment : $EXPERIMENT"
echo " checkpoints: $CKPT_DIR/$EXPERIMENT"
echo " gpus       : $N_GPUS"
[[ ${#EXTRA[@]} -gt 0 ]] && echo " overrides  : ${EXTRA[*]}"
echo "=============================================================="

# Run from the repo root, not from the verl checkout: hydra resolves
# config_path relative to main_ppo.py inside the installed package, so cwd is
# free, and keeping it here is what makes the relative parquet / reward-function
# paths above resolve.
#
# On the validation length: verl has no separate budget for it — in-training
# validation reuses data.max_response_length, so the 31744-token allowance for
# AIME lives in eval_math.py (--max-tokens), which is the number that ends up in
# the results table. The mid-training test_freq passes are a progress signal, not
# the measurement.
ARGS=(
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    "${REWARD_ARGS[@]}" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=20 \
    trainer.test_freq=2000 \
    "trainer.logger=$LOGGER" \
    trainer.project_name=shadow-ft-math \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$CKPT_DIR/$EXPERIMENT" \
    "${EXTRA[@]}"
)

# DRY_RUN=1 prints the fully composed config and exits without touching a GPU.
# The fastest way to check that an override actually landed where you think it
# did -- hydra accepts a well-formed key that no code reads, so "it started" is
# not evidence that the setting took effect.
if [[ -n "${DRY_RUN:-}" ]]; then
    exec python3 -m verl.trainer.main_ppo --cfg job "${ARGS[@]}"
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo "${ARGS[@]}" \
    2>&1 | tee "$CKPT_DIR/$EXPERIMENT.log"

echo
echo "[done] $EXPERIMENT -> $CKPT_DIR/$EXPERIMENT"
echo "Convert the FSDP checkpoint to HuggingFace format before merging:"
echo "  python shadow_rl/verl_math/export_hf.py --ckpt $CKPT_DIR/$EXPERIMENT --out <hf_dir>"
