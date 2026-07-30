# Qwen3-4B GRPO on verl, with the Shadow-FT graft

Runs GRPO **twice** on the same recipe — once from `Qwen3-4B-Base`, once from
`Qwen3-4B` — then grafts the base-side update onto the instruct backbone and
scores all five resulting models on the same benchmarks:

```
Delta_K  = RL(W_B) - W_B      # what GRPO taught the base model
W_shadow = W_I + Delta_K      # transplanted onto the instruct backbone
```

| role | model |
|---|---|
| `W_B` | `Qwen/Qwen3-4B-Base`, untouched |
| `W_I` | `Qwen/Qwen3-4B`, untouched |
| `RL(W_B)` | GRPO from the base checkpoint |
| `RL(W_I)` | GRPO from the instruct checkpoint — the baseline to beat |
| `W_shadow` | the graft |

The question the run answers is `W_shadow` vs `RL(W_I)`: is it better to run RL
on the base model and transplant the update, than to run the same RL on the
instruct model directly?

This is the training-from-scratch counterpart to `shadow_rl/`, which tests the
same hypothesis on *released* Search-R1 checkpoint pairs without training
anything. The graft arithmetic is shared — `shadow_rl/merge.py`, unchanged.

---

## Read this before you start

**Two things here are assumptions, not transcriptions.** The model card at
`huggingface.co/lllyx/Qwen3-4B-Base-GRPO` is unreachable from the sandbox this
was written in (the environment's network policy returns 403 for
`huggingface.co`), so the GRPO recipe and the benchmark list were chosen as
defaults rather than copied:

* **RL data**: DeepMath-103K with a rule-based reward on the boxed answer.
  Chosen because every row carries a verifiable answer and the repo already
  prepares this dataset. → `RL_DATASET` in `config.sh`.
* **Eval suite**: AIME24, AIME25, MATH-500, GSM8K, OlympiadBench, GPQA-Diamond —
  what a GRPO'd math model is normally reported on. → `EVAL_SUITE` in
  `config.sh`, or `--datasets` on `make_eval_config.py`.

Both are one-line changes. If the card lists a different dataset, reward, or
benchmark set, set them in `config.sh` and nothing else needs to move.

The hyperparameters (`TRAIN_BATCH_SIZE=512`, `ROLLOUT_N=8`, `lr=1e-6`,
`kl_loss_coef=0.001`) are conventional GRPO values sized for a 4B actor on
8×B300, not values read from the card.

---

## Quickstart

```bash
# 0. the B300 stack (torch/vLLM matched to sm_103 -- see docs/B300_PRODUCTION.md)
bash setup_env.sh

# 1. verl on top of it
./verl_rl/setup.sh                       # --with-flash-attn to build one for sm_103

# 2. prove the plumbing works end to end: 200 synthetic rows, 5 steps, ~minutes
./verl_rl/run_all.sh --demo --yes

# 3. the real run
./verl_rl/run_all.sh
```

`--demo` exercises every stage — data, both GRPO runs, both exports, the graft,
an eval — on arithmetic a 4B model can actually solve, so the reward curve
moves. It is a plumbing test, not a result.

Look at the plan before committing to the long one:

```bash
./verl_rl/run_all.sh --dry-run
```

## Stages

`run_all.sh` runs them in order and skips whatever is already complete. Each is
also a standalone command:

| stage | command | output |
|---|---|---|
| data | `python3 verl_rl/prepare_data.py --out $DATA_DIR` | `train.parquet`, `val.parquet` |
| train | `./verl_rl/train_grpo.sh base` / `instruct` | verl FSDP checkpoints |
| export | `python3 verl_rl/export_hf.py --role base` | HF model directory |
| merge | `./verl_rl/merge_shadow.sh` | `W_shadow` + smoke test |
| eval | `python3 verl_rl/make_eval_config.py --out ...` then OpenCompass | the table |

Resume from any point: `./verl_rl/run_all.sh --stages merge,eval`.

## What is held fixed, and why it matters

The comparison is only meaningful if the two GRPO runs differ in the starting
weights and in nothing else. Enforced here:

* **Same data, same order, same seed.** Both runs read the same
  `train.parquet`. The validation split is drawn with a fixed seed, so both are
  validated on identical questions — a test pins this.
* **Same prompt path.** Prompts are stored as chat messages and rendered by
  verl's `apply_chat_template` for both roles, including the base one. Giving
  the base model a raw-text prompt would confound "base vs instruct
  initialisation" with a prompt-format difference.
* **Same recipe.** `train_grpo.sh` takes only the role as an argument; every
  hyperparameter comes from `config.sh` for both.
* **Same eval.** One OpenCompass config, one model class, five roles.

## The reward

`reward_math.py`, passed to verl as `custom_reward_function.path`. Binary: 1.0
if the last `\boxed{...}` in the response is equivalent to the ground truth, 0.0
otherwise.

Custom rather than verl's `data_source` registry because that registry maps a
fixed set of dataset names to scorers and raises on a name it does not know —
which would tie the dataset choice to verl's internals for no benefit.

Equivalence uses `math_verify` when installed, falling back to string
normalisation that collapses spellings of the same value (`0.5`/`.5`,
`1{,}000`/`1000`, `\dfrac`/`\frac`, `12.0`/`12`). The fallback is strict, never
loose: it can score a correct answer 0, never an incorrect answer 1.

The prompt asks for `\boxed{}` and the scorer reads `\boxed{}` — if those two
disagreed, GRPO would be optimising against its own instruction.

```bash
python verl_rl/reward_math.py --self-test
```

## On the B300 specifically

`config.sh` sources `scripts/gpu_profile.sh`, so the GPU count, the attention
implementation and the CUDA-matched wheel index all come from the machine. See
`docs/B300_PRODUCTION.md` for the stack itself.

**Attention is the one that bites.** verl's `use_remove_padding` path packs
variable-length sequences and calls flash-attn's varlen kernels directly — it is
not a preference with a fallback. `gpu_profile.sh` decides by *calling*
`flash_attn_func` on the GPU, because a wheel built for `sm_100a` imports
happily on a B300 and then has no kernels. If flash-attn is unusable,
`train_grpo.sh` turns packing off and switches to sdpa, and says so. That costs
throughput on mixed-length batches; `./verl_rl/setup.sh --with-flash-attn`
builds one for `sm_103` (needs CUDA ≥ 12.9, which `gpu_profile.sh --print`
reports).

Sizing at the defaults, 8×B300: a 4B actor, rollout TP 1, 512 prompts × 8
samples per step. There is a lot of headroom at 279 GB/GPU — `ROLLOUT_N`,
`MAX_RESPONSE_LEN` and `MICRO_BATCH_PER_GPU` are the knobs to spend it on, in
that order. `ROLLOUT_N` changes the algorithm (it is the GRPO group size);
`MICRO_BATCH_PER_GPU` does not.

## Tests

CPU-only, no verl, no model, no GPU:

```bash
python verl_rl/tests/test_pipeline.py
```

They pin the failures that otherwise appear hours into a run: a reward that
scores correct answers 0, a parquet schema verl cannot read, the two runs
getting different validation questions, and an eval config naming a dataset
symbol OpenCompass does not export.

## Verified vs not

Everything above runs and is tested on CPU: the reward function, the data
schema, the eval-config generation against the vendored OpenCompass tree, the
config plumbing, and the orchestrator's dry run.

Two things need the B300 itself and could not be exercised here:

* **The verl version's option names.** `train_grpo.sh` passes an `sdpa`
  attention override with Hydra's `+` syntax; the key has moved between verl
  releases. If your verl rejects it, drop that line and set
  `attn_implementation` in the model's `config.json` instead.
* **The checkpoint merger's entry point.** `export_hf.py` tries
  `verl.model_merger` and `verl.scripts.model_merger`, with and without the
  `merge` subcommand, and reports what it tried if none work. It then verifies
  the output actually contains weights — a merger that exits 0 having written
  only a config is the failure mode that otherwise surfaces during the eval.

## Layout

| file | what it does |
|---|---|
| `config.sh` | every model id, path and hyperparameter; everything else reads it |
| `setup.sh` | verl + vLLM + ray for this GPU; optional flash-attn build |
| `prepare_data.py` | dataset → verl parquet, with a `--demo` set that needs no download |
| `reward_math.py` | the rule-based reward, self-testing |
| `train_grpo.sh` | one GRPO run (`base` or `instruct`) |
| `export_hf.py` | verl FSDP checkpoint → HF directory |
| `merge_shadow.sh` | the graft, via `shadow_rl/merge.py`, plus a smoke test |
| `make_eval_config.py` | OpenCompass config for all five roles |
| `run_all.sh` | one-click, resumable |
| `tests/test_pipeline.py` | CPU tests |
