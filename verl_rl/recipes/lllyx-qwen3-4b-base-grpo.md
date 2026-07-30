<!--
The published recipe for huggingface.co/lllyx/Qwen3-4B-Base-GRPO, as supplied by
the user. This environment cannot reach huggingface.co (egress policy), so this
file is the reference config.sh is checked against:

    python verl_rl/compare_recipe.py --card verl_rl/recipes/lllyx-qwen3-4b-base-grpo.md

Transcribed from the model card, not scraped -- if the page changes, this does
not.
-->

Base model:        Qwen3-4B-Base
Framework:         verl
Stage:             RL (GRPO), full-parameter actor update
Rollout engine:    vLLM
Precision:         BF16

# Algorithm
algorithm:                grpo
grpo outcome weight:      1.0
reward_model.enable:      false
reward:                   custom rule-based math reward fn
kl_loss:                  disabled
format reward:            disabled
loss aggregation:         token-mean

# Data
train:                    DAPO-Math-17k-Processed
train file:               datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
val:                      AIME24, AIME25, AMC23

# Lengths
prompt length:            1024
response length:          7168
val response length:      31744
max model length:         32768

# Rollout
n responses per prompt:   8
temperature:              1.0
repetition penalty:       1.0

# Optim
lr:                       1e-6
ppo mini-batch size:      64
ppo micro-batch/gpu:      1
epochs:                   1

# Infra
n gpus:                   8
tensor parallel size:     1
save freq:                every 20 steps
test freq:                every 20 steps
