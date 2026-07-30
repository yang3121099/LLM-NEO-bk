# Shadow-FT on RL we train ourselves — Qwen3-4B, GRPO on DAPO-Math

The Search-R1 track reuses someone else's matched checkpoints. This track trains
both sides itself, so the recipe is under our control and the pair is matched by
construction rather than by trust.

```
train GRPO on Qwen3-4B-Base      -> RL(W_B)
train GRPO on Qwen3-4B-Instruct  -> RL(W_I)      identical recipe
W_shadow = W_I + (RL(W_B) - W_B)
```

Then all five models are scored on AIME24 / AIME25 / AMC23.

## Demo first

Everything runs, on a fraction of the data — 8 optimiser steps, 512 prompts, 8
problems per benchmark:

```bash
./shadow_rl/verl_math/run_all_math.sh --demo --yes
```

That exercises data prep, GRPO on both sides, the verl→HF export, the graft, the
smoke test and the eval. **The numbers are meaningless** — 8 steps will not move
a 4B model — but if this completes, the full run is only a matter of time.

GPU count is detected automatically; training uses all of them, evaluation uses
`--tp 1` since a 4B model fits on one card. Override with `N_GPUS=` / `--tp`.

## Run it

```bash
git clone https://github.com/volcengine/verl ~/verl
export VERL_ROOT=~/verl

./shadow_rl/verl_math/run_all_math.sh --yes
```

Stages run in order and each is skipped if already done:

| stage | what it does |
|---|---|
| `data` | DAPO-Math train parquet + AIME24/AIME25/AMC23 validation parquet |
| `train` | GRPO on both sides — the expensive part, 8 GPUs, hours each |
| `export` | verl FSDP checkpoints → HuggingFace directories |
| `merge` | `W_shadow`, then a smoke test |
| `eval` | five roles × three benchmarks → `math_results.csv` |

Pick a subset with `--stages`, e.g. `--stages merge,eval` once training is done.

## Files

| file | what it does |
|---|---|
| `prepare_data.py` | builds the parquets; tries several HF mirrors per benchmark |
| `train_grpo.sh` | one side of the pair, `base` or `instruct` |
| `math_reward.py` | rule-based reward — also the eval scorer, so both agree |
| `export_hf.py` | verl checkpoint → HuggingFace format |
| `eval_math.py` | avg@k on the three benchmarks |
| `report_math.py` | terminal table, same shape as the search track |
| `run_all_math.sh` | the five stages above |

## Hyperparameters

As specified: GRPO, `train_batch_size=64`, `ppo_mini_batch_size=64` (one update
per step), lr `1e-6`, `rollout.n=8`, temperature 1.0, no KL in reward and no KL
loss, `loss_agg_mode=token-mean`, 1 epoch, save/test every 20 steps.
`ppo_micro_batch_size_per_gpu=1` only affects memory, not the update.

Prompt length 1024, response length 7168 for training; validation generates up to
31744 so AIME solutions have room to finish. `micro_batch_size` aside, **both
sides must use the identical recipe** — tuning one and not the other invalidates
the comparison, which is the entire point of the pair.

## Reward

Binary correctness on the final answer: `\boxed{...}` if present, else "answer
is X", else a trailing number. Comparison is exact after LaTeX normalisation,
falling back to exact rational arithmetic so `\frac{1}{2}` equals `0.5` and
`\frac{2}{6}` equals `1/3`.

No format shaping and no partial credit: GRPO normalises advantages within a
prompt group, so a sparse binary signal is what the recipe assumes, and a shaping
term would change what is being compared between the two runs.

`python shadow_rl/verl_math/math_reward.py` self-tests the extractor, including
the nested-brace case a regex gets wrong (`\boxed{\frac{\sqrt{2}}{2}}`).

## Reading the result

```bash
python shadow_rl/verl_math/report_math.py
```

**These benchmarks are tiny** — 30 problems for each AIME, 40 for AMC23. One
greedy pass is very high variance, so `eval_math.py` defaults to `--k 4` and
reports avg@k. Even then a margin under ~0.05 is not a result; raise `--k`
before believing a small difference.
