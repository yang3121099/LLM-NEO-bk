#!/usr/bin/env bash
###############################################################################
# config.sh — every setting for the Qwen3-4B GRPO + Shadow-FT run, in one place.
#
# Sourced by setup.sh, train_grpo.sh, merge_shadow.sh and run_all.sh. Nothing
# else in verl_rl/ hardcodes a path, a model id or a hyperparameter.
#
# Override anything from the environment:
#   BASE_MODEL=Qwen/Qwen3-8B-Base ./verl_rl/run_all.sh
#   MAX_RESPONSE_LEN=8192 ROLLOUT_N=16 ./verl_rl/run_all.sh
###############################################################################

VERL_RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$VERL_RL_DIR/.." && pwd)"

# --- The pair -----------------------------------------------------------------
# Shadow-FT needs the two checkpoints of the *same* model: the delta learned on
# the base backbone is grafted onto the instruct one, so they must share
# architecture, vocabulary and tensor shapes. merge.py verifies that.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Base}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-Qwen/Qwen3-4B}"
PAIR_NAME="${PAIR_NAME:-qwen3-4b-grpo}"

# --- Where everything lands ---------------------------------------------------
WORK_DIR="${WORK_DIR:-$REPO_ROOT/results/verl_rl/$PAIR_NAME}"
DATA_DIR="${DATA_DIR:-$WORK_DIR/data}"
CKPT_DIR="${CKPT_DIR:-$WORK_DIR/checkpoints}"   # verl's sharded output
EXPORT_DIR="${EXPORT_DIR:-$WORK_DIR/hf}"        # HF-format exports
MERGED_DIR="${MERGED_DIR:-$WORK_DIR/merged}"    # W_shadow
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"

# --- RL data ------------------------------------------------------------------
# DAPO-Math-17k, per the published recipe. Accepts either a local parquet (the
# recipe names datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet) or a HF id;
# prepare_data.py passes a verl-shaped parquet through untouched and maps field
# names otherwise.
RL_DATASET="${RL_DATASET:-DAPO-Math-17k-Processed}"
TRAIN_SIZE="${TRAIN_SIZE:-0}"      # 0 = all

# Validation is the three competition sets the recipe names, not a slice of the
# training data. They are small and hard, which is why the validation response
# budget below is four times the training one.
VAL_DATASETS="${VAL_DATASETS:-AIME24,AIME25,AMC23}"
VAL_SIZE="${VAL_SIZE:-0}"          # 0 = the whole of each validation set

# --- GRPO ---------------------------------------------------------------------
# These follow the published recipe. The ones that define the update are
# ROLLOUT_N (the GRPO group size), PPO_MINI_BATCH_SIZE, LEARNING_RATE, and the
# fact that the KL term is off; MICRO_BATCH_PER_GPU is accumulation chunking.
#
# TRAIN_BATCH_SIZE is the one number the recipe does NOT state. 512 is our
# choice, not theirs -- see the note in README.md before treating a run as an
# exact reproduction.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

# KL disabled. verl only builds the reference-policy worker when a KL term needs
# it, so this also frees a whole model's worth of memory per GPU.
USE_KL_LOSS="${USE_KL_LOSS:-false}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"   # unused while USE_KL_LOSS=false
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# token-mean: every token in the mini-batch weighs the same, so a long rollout
# contributes proportionally more than a short one. With responses up to 7168
# tokens this is not a detail -- seq-mean-token-sum would weight the two very
# differently.
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"    # 0 = derive from epochs
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
SEED="${SEED:-1}"

# 1024 + 7168 = 8192 for training; 1024 + 31744 = 32768 for validation. The
# rollout engine is sized for the larger of the two.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-7168}"
VAL_RESPONSE_LEN="${VAL_RESPONSE_LEN:-31744}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# --- Rollout engine -----------------------------------------------------------
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"                   # 4B fits one B300 comfortably
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.75}"

# --- Reward -------------------------------------------------------------------
# A custom scorer rather than verl's data_source registry: the registry maps a
# fixed set of dataset names to scorers, and a name it does not know raises at
# the first batch. reward_math.py is self-contained and takes the same
# (data_source, solution_str, ground_truth) contract.
REWARD_FN_PATH="${REWARD_FN_PATH:-$VERL_RL_DIR/reward_math.py}"
REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score}"

# --- Evaluation ---------------------------------------------------------------
# The final scoring of all five roles, separate from the in-training validation
# above. `recipe` is the recipe's own validation sets, so the numbers line up
# with what the run reports during training.
#   EVAL_SUITE=recipe  AIME24, AIME25 (+ AMC23 when a config exists for it)
#   EVAL_SUITE=math    the above plus MATH-500, GSM8K, OlympiadBench, GPQA-D
#   EVAL_SUITE=shadow  the repo's existing Shadow-FT set (comparable to the SFT runs)
EVAL_SUITE="${EVAL_SUITE:-recipe}"
EVAL_DATASETS="${EVAL_DATASETS:-}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"            # vllm | turbomind
# Matches the recipe's validation budget, so eval-time truncation cannot make a
# model look worse than it was during training.
EVAL_MAX_OUT_LEN="${EVAL_MAX_OUT_LEN:-31744}"
EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-32768}"

# --- Hardware (from the GPU actually present) ---------------------------------
# shellcheck source=../scripts/gpu_profile.sh
source "$REPO_ROOT/scripts/gpu_profile.sh"
N_GPUS="${N_GPUS:-${GPU_COUNT:-8}}"
[[ "$N_GPUS" -ge 1 ]] 2>/dev/null || N_GPUS=1
NNODES="${NNODES:-1}"

export VERL_RL_DIR REPO_ROOT BASE_MODEL INSTRUCT_MODEL PAIR_NAME
export WORK_DIR DATA_DIR CKPT_DIR EXPORT_DIR MERGED_DIR LOG_DIR
export N_GPUS NNODES
