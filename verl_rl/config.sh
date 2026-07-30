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
# DeepMath-103K: every row carries a verifiable final answer, which is what a
# rule-based reward needs. scripts/prepare_deepmath_103k.py already handles this
# dataset for SFT; prepare_data.py writes the verl parquet form.
RL_DATASET="${RL_DATASET:-zwhe99/DeepMath-103K}"
TRAIN_SIZE="${TRAIN_SIZE:-0}"      # 0 = all
VAL_SIZE="${VAL_SIZE:-500}"

# --- GRPO ---------------------------------------------------------------------
# Sized for 8x B300 (279 GB/GPU) and a 4B actor. The two batch sizes that matter
# for reproducibility are TRAIN_BATCH_SIZE (prompts per step) and ROLLOUT_N
# (samples per prompt, i.e. the GRPO group size); the micro-batch is a memory
# knob only and does not change the update.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"    # 0 = derive from epochs
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
SEED="${SEED:-1}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

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
# Swap the whole suite with EVAL_SUITE, or name datasets explicitly:
#   EVAL_SUITE=math   AIME24, AIME25, MATH-500, GSM8K, OlympiadBench, GPQA-D
#   EVAL_SUITE=shadow the repo's existing Shadow-FT set (comparable to the SFT runs)
#   EVAL_DATASETS="aime2024,math500"  explicit
EVAL_SUITE="${EVAL_SUITE:-math}"
EVAL_DATASETS="${EVAL_DATASETS:-}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"            # vllm | turbomind
EVAL_MAX_OUT_LEN="${EVAL_MAX_OUT_LEN:-8192}"
EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-16384}"

# --- Hardware (from the GPU actually present) ---------------------------------
# shellcheck source=../scripts/gpu_profile.sh
source "$REPO_ROOT/scripts/gpu_profile.sh"
N_GPUS="${N_GPUS:-${GPU_COUNT:-8}}"
[[ "$N_GPUS" -ge 1 ]] 2>/dev/null || N_GPUS=1
NNODES="${NNODES:-1}"

export VERL_RL_DIR REPO_ROOT BASE_MODEL INSTRUCT_MODEL PAIR_NAME
export WORK_DIR DATA_DIR CKPT_DIR EXPORT_DIR MERGED_DIR LOG_DIR
export N_GPUS NNODES
