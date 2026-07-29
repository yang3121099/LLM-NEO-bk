# Shadow-FT weight grafting on matched RL checkpoint pairs

Does Shadow-FT hold for RL the way it does for SFT and DPO?

Search-R1 released, for the same model and the same RL recipe, **both** the
base-initialised and the instruct-initialised checkpoint. That lets us test
grafting on RL without running any RL:

```
Delta_K  = RL(W_B) - W_B     # what the base model learned
W_shadow = W_I + Delta_K     # graft it onto the instruct backbone
```

Four models per pair — `W_I`, `RL(W_I)`, `RL(W_B)`, `W_shadow` — over seven QA
sets, Exact Match. Zero training; weight arithmetic plus evaluation.

## Quickstart, from nothing

```bash
# 1. clone this repo and switch to the branch
git clone https://github.com/yang3121099/LLM-NEO-bk.git
cd LLM-NEO-bk
git checkout claude/shadow-ft-rl-weight-grafting-tc0uow

# 2. install everything and verify (creates ~/shadow-rl-venv, clones Search-R1)
./shadow_rl/setup.sh                 # add --with-bm25 if you want the search pairs too

# 3. activate and go
source ~/shadow-rl-venv/bin/activate
export SEARCH_R1_ROOT=~/Search-R1

./shadow_rl/run_all.sh --pairs nosearch          # ~2h on one H200, no retrieval needed
```

`setup.sh` finishes by running the three CPU test suites; if any fail it stops
rather than letting you start a long run on a broken install.

Before committing to a full run, look at the plan:

```bash
./shadow_rl/run_all.sh --pairs all --dry-run     # prints pairs, disk estimate, stages
```

## Layout

| file | what it does |
|---|---|
| `setup.sh` | one-time install: venv, torch, vllm, Search-R1 checkout, self-test |
| `run_all.sh` | **one-click**: similarity → merge → evaluate → aggregate, any pair subset |
| `similarity.py` | per-parameter σ, cosines and RL-update magnitudes |
| `merge.py` | streaming safetensors merge + sanity stats |
| `pairs.py` | manifest of the 12 pairs and the published reference numbers |
| `evaluate.py` | Search-R1 rollout + official EM scoring → `results.csv` |
| `aggregate.py` | `results.csv` → `FINDINGS.md` |
| `smoke_test.py` | does the merged model load and generate coherent text? |
| `run_pair.sh` | a single pair end to end (`run_all.sh` is usually what you want) |
| `launch_bm25_retriever.sh` | corpus + BM25 index + retrieval server (search pairs only) |
| `tests/` | CPU-only tests: merge arithmetic, similarity stats, eval contract |

The tests need no GPU and no model:

```bash
python shadow_rl/tests/test_merge.py
python shadow_rl/tests/test_similarity.py
python shadow_rl/tests/test_evaluate.py --search-r1-root $SEARCH_R1_ROOT
```

## `run_all.sh`

| option | meaning |
|---|---|
| `--pairs X` | `all`, `nosearch`, `search`, `grpo`, `ppo`, `3b`, `7b`, or explicit ids |
| `--stages X` | subset of `similarity,merge,eval,aggregate` |
| `--roles X` | subset of the four model roles |
| `--limit N` | first N questions per dataset — biased, smoke runs only |
| `--sample N` | deterministic random N per dataset, identical across roles |
| `--auto-retriever` | start and stop the BM25 server automatically |
| `--cleanup` | delete a pair's RL checkpoints and merged model once it is evaluated |
| `--tp N` | tensor parallel size (default 1; an H200 fits 7B at 1) |
| `--dry-run` | print the plan and stop |
| `--yes` | skip the confirmation prompt |

Everything is resumable. Finished datasets are detected in `results.csv` and
skipped, existing merged models are reused, and pairs already in
`similarity_summary.csv` are not recomputed — so re-running after an interruption
picks up where it stopped. A failing pair is logged and the run moves to the next
rather than aborting the batch; the exit code is non-zero if anything failed.
Per-pair logs land in `shadow_rl/logs/`.

Useful invocations:

```bash
# the full comprehensive sweep, unattended
nohup ./shadow_rl/run_all.sh --pairs all --sample 500 \
      --cleanup --auto-retriever --yes > shadow_rl/logs/nohup.out 2>&1 &
tail -f shadow_rl/logs/nohup.out

# quick end-to-end sanity run: 50 questions per dataset, ~15 min
./shadow_rl/run_all.sh --pairs ppo-nosearch-3b-v0.2 --limit 50 --yes

# similarity only, all 12 pairs, no GPU needed
./shadow_rl/run_all.sh --pairs all --stages similarity --yes

# reproduce the published numbers before trusting anything
./shadow_rl/run_all.sh --pairs nosearch --roles rl_on_base,rl_on_instruct

# the whole thing, freeing disk as it goes
./shadow_rl/run_all.sh --pairs all --cleanup --yes
```

## How long a comprehensive run takes

The official harness evaluates the **full** test sets (`val_data_num: null`) —
about 51,700 questions per model role:

| dataset | rows | | dataset | rows |
|---|---|---|---|---|
| nq | 3,610 | | 2wikimultihopqa | 12,576 |
| triviaqa | 11,313 | | musique | 2,417 |
| popqa | 14,267 | | bamboogle | 125 |
| hotpotqa | 7,405 | | **total** | **51,713** |

`--pairs all` means 12 pairs × 4 roles × 51,713 = **~2.5M multi-turn rollouts**:

| run | questions/role | estimated time, 1× H200 |
|---|---|---|
| `--pairs all` (full sets) | 51,713 | **~163 h (~7 days)** |
| `--pairs all --sample 500` | 3,500 | **~10 h** |
| `--pairs all --sample 1000` | 7,000 | ~20 h |
| `--pairs nosearch` (full sets) | 51,713 | ~8 h |

`run_all.sh` prints this estimate during preflight before you commit to a run.

**`--sample 500` is the recommended setting for the first comprehensive sweep.**
It draws a seeded random subset — the same questions for every role and every
pair, so the four roles stay comparable — and gets the whole grid in a night
instead of a week. The sample size is recorded per row in `results.csv` as
`n_questions`, and `FINDINGS.md` flags a subsampled run and warns loudly if two
roles were somehow scored on different question counts.

The tradeoff is noise. At n=500 per dataset the standard error on a single EM
number is roughly ±0.02, and on the 7-dataset average roughly ±0.008. Margins
smaller than that are not real. Once the sweep identifies which pairs show a
promising `shadow − rl_on_instruct`, re-run just those on the full sets:

```bash
./shadow_rl/run_all.sh --pairs grpo-search-3b-v0.3 --yes   # no --sample = full
```

Since `results.csv` is keyed by pair/role/dataset, delete the sampled rows for
that pair first, or point `--out` at a separate file.

## Parameter-level similarity

`similarity.py` runs before any merging and writes two CSVs —
`similarity_params.csv` (one row per parameter tensor) and
`similarity_summary.csv` (aggregated per module type, per layer, and overall).
Per tensor it reports:

| statistic | what it tells you |
|---|---|
| `sigma` | `Σ\|W_B−W_I\| / (Σ\|W_B\|+Σ\|W_I\|)` — the Shadow-FT applicability condition, working range 0.003–0.042 |
| `cos_base_instruct` | angle between the two backbones |
| `rel_delta_base` | `‖RL(W_B)−W_B‖/‖W_B‖` — how much RL moved the base model |
| `rel_delta_instruct` | the same on the instruct side |
| `cos_delta_base_instruct` | **`cos(Δ_B, Δ_I)`** — do the two RL runs point the same way? |

That last one is the statistic worth reading first. It asks whether the update
learned on the base model resembles the one learned on the instruct model. High
cosine means transplanting it is well-posed; near zero means the two runs found
unrelated solutions and the graft is a gamble regardless of what σ says. The
aggregates are ratios of summed quantities, not averages of per-tensor ratios, so
the `ALL` row is the exact whole-model figure.

Group totals are exact under splitting — the test suite pins that folding two
halves of a tensor gives the same numbers as folding it whole.

## Order of work

The `R1-*` pairs are RL **without** a search engine, so they need no retrieval
server and no index. They are both the cheapest experiment and the setting where
the base-initialised model wins — start there.

`run_all.sh --pairs nosearch` does steps 1–4 below in one command. The manual
commands are spelled out so you can run a stage on its own or debug one.

**0. Look at the parameter statistics first.** No GPU needed, and it tells you
whether grafting is even well-posed for a pair before you spend time on it.

```bash
python shadow_rl/similarity.py --pair ppo-nosearch-3b-v0.2
```

**1. Merge the 3B `R1-*` pair and check it.**

```bash
python shadow_rl/merge.py \
    --base        Qwen/Qwen2.5-3B \
    --instruct    Qwen/Qwen2.5-3B-Instruct \
    --rl-base     PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.2 \
    --rl-instruct PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2 \
    --out         shadow_rl/merged/ppo-nosearch-3b-v0.2

python shadow_rl/smoke_test.py --model shadow_rl/merged/ppo-nosearch-3b-v0.2
```

`merge.py` prints `sigma` and both relative delta magnitudes, and warns if sigma
falls outside the paper's 0.003–0.042 working range. Read the smoke-test
completions before going further — a bad merge loads fine and talks nonsense.

**2. Reproduce the published numbers first.**

```bash
for ROLE in rl_on_base rl_on_instruct; do
  python shadow_rl/evaluate.py --search-r1-root $SEARCH_R1_ROOT \
      --pair ppo-nosearch-3b-v0.2 --role $ROLE
done
python shadow_rl/aggregate.py
```

Target `rl_on_base` ≈ 0.229 and `rl_on_instruct` ≈ 0.224. The *Harness
validation* table in `FINDINGS.md` does this diff for you. **Do not evaluate the
merged model until these agree** — if the two released checkpoints do not
reproduce, the merged number cannot be trusted either.

**3. Evaluate the merged model.** The first real result.

```bash
python shadow_rl/evaluate.py --search-r1-root $SEARCH_R1_ROOT \
    --pair ppo-nosearch-3b-v0.2 --role shadow \
    --model-path shadow_rl/merged/ppo-nosearch-3b-v0.2
```

Or run all four roles at once: `./shadow_rl/run_pair.sh ppo-nosearch-3b-v0.2`.

**4. Repeat for 7B** (`ppo-nosearch-7b-v0.2`, reference 0.276 / 0.271, `TP=2`).

**5. Stand up retrieval and move to the GRPO pairs, 3B first.**

```bash
./shadow_rl/launch_bm25_retriever.sh          # serves :8000, leave running
./shadow_rl/run_pair.sh grpo-search-3b-v0.3
```

`python shadow_rl/pairs.py` lists all 12 pair ids.

## Implementation notes

**Merge.** Streams shard by shard and writes incrementally — peak memory is a few
tensors, not four models. Arithmetic is fp32, output dtype follows `W_I`: the RL
checkpoints are often fp32 while official Qwen weights are bf16, and a bf16
subtraction throws away most of a small delta (the test suite pins this — the two
paths diverge by 2.4e-4 on weights of magnitude 2e-2). Key sets and tensor shapes
are asserted across all four checkpoints before anything is written, and a
mismatch fails with a diff naming the offending keys. Tokenizer and config come
from `W_I`; output sharding mirrors `W_I` so the index and any weight tying stay
consistent.

**Prompting.** The merged model is an instruct model in every respect that
matters, and is prompted through the standard instruct path. Search-R1 stores
prompts as chat messages and lets verl apply the template, so *every* role —
including the base-side checkpoints — goes through `apply_chat_template`.

**Scoring.** `verl/utils/reward_score/qa_em.py` is loaded from the Search-R1
checkout and used unmodified. This matters more than it looks:
`extract_solution` returns the **last** `<answer>` match and gives up on
sequences with fewer than two. The prompt itself contributes two matches — the
"inside `<answer>` and `</answer>`" phrasing and the `<answer> Beijing </answer>`
example — so the model's answer lands last. Consequences:

- The scored string is the **full prompt + rollout**, exactly as verl's reward
  manager decodes it. Scoring the response alone drives every model to 0.0.
- A rollout that never answers falls back to the prompt's "Beijing", so it counts
  as correct on the rare question whose gold answer is Beijing. That is the
  official harness's behaviour and is left alone; watch the per-dataset "produced
  no parseable `<answer>`" count instead.

**Rollout.** Reproduces `LLMGenerationManager.run_llm_loop`: `max_turns` (4)
search rounds plus a final answer-only rollout, responses truncated at the first
`</search>` or `</answer>`, observations wrapped in `<information>` and clipped to
500 tokens, malformed actions given the harness's retry nudge. Decoding is greedy
so EM is reproducible.

**Retrieval.** 2018 Wikipedia, top-k 3, action budget 4, using
`PeterJinGo/wiki-18-corpus` with `PeterJinGo/wiki-18-bm25-index`. This is **not**
the paper's dense E5 index — BM25 is used for speed, so absolute EM on the search
pairs will sit below the published numbers. All four roles share the index, so
the comparison between them stays fair. This caveat is repeated in `FINDINGS.md`.

**Baseline caveat.** All four roles get the same Search-R1 prompt so they are
comparable to each other. The paper's `instruct_baseline` *direct inference*
numbers use a plain QA prompt instead, so that row is not a like-for-like
reference and is excluded from the harness-validation table.

## Reading the result

`FINDINGS.md` reports, per pair, the four averages plus
`shadow − rl_on_instruct` and `shadow − rl_on_base`.

Expectations differ by setting, and a modest gain is not a failure:

- **Without search** the base-initialised model wins (0.229 vs 0.224 at 3B,
  0.276 vs 0.271 at 7B) — matching the SFT findings. Grafting should help here.
- **Under GRPO with search** the instruct-initialised model wins (0.336 vs 0.312
  at 3B, 0.396 vs 0.350 at 7B). Direct RL is not damaging the instruct
  checkpoint, so by our own applicability condition the expected margin is small.
  Anything above `RL(W_I)` counts as a positive result.

## Cost and disk

The RL checkpoints are frequently fp32, so they are roughly twice the size of the
official bf16 weights. Per pair:

| | 3B pair | 7B pair |
|---|---|---|
| two RL checkpoints | ~24 GB | ~56 GB |
| merged model | ~6 GB | ~15 GB |
| GPU | 1× H200 at `--tp 1` | 1× H200 at `--tp 1` |

Plus the shared originals, downloaded once: ~12 GB for the 3B base+instruct,
~30 GB for the 7B.

**All 12 pairs at once needs ~607 GB.** `run_all.sh` prints this estimate against
your actual free space during preflight and warns if it will not fit. With
`--cleanup` each pair's checkpoints are deleted after it is evaluated, which keeps
the peak near **~130 GB** — that is the flag to use unless you have plenty of
disk. The search pairs additionally need ~70 GB for the Wikipedia corpus and BM25
index, downloaded once and shared.

An H200 has 141 GB of HBM, so `--tp 1` is right for both sizes; raise `--tp` only
to shorten wall-clock on the 7B pairs if you have GPUs to spare.
