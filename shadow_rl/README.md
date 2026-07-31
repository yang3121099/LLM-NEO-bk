# Shadow-FT weight grafting on matched RL checkpoint pairs

Does Shadow-FT hold for RL the way it does for SFT and DPO?

Search-R1 released, for the same model and the same RL recipe, **both** the
base-initialised and the instruct-initialised checkpoint. That lets us test
grafting on RL without running any RL:

```
Delta_K  = RL(W_B) - W_B     # what the base model learned
W_shadow = W_I + Delta_K     # graft it onto the instruct backbone
```

Five models per pair — `W_B` and `W_I` (the raw HF checkpoints), `RL(W_I)`,
`RL(W_B)`, and `W_shadow` — over seven QA sets, Exact Match. Zero training;
weight arithmetic plus evaluation.

Setup written out as plain commands: **[SETUP.md](SETUP.md)** — conda or venv,
CUDA choice for Blackwell vs Hopper, BM25 dependencies, verl, and where
everything lands. `setup.sh` automates the same steps.

## Quickstart, from nothing

```bash
# 1. clone this repo and switch to the branch
git clone https://github.com/yang3121099/LLM-NEO-bk.git
cd LLM-NEO-bk
git checkout claude/shadow-ft-rl-weight-grafting-tc0uow

# 2. install everything and verify
./shadow_rl/setup.sh                 # add --with-bm25 if you want the search pairs too

# 3. activate and go
source .venv/bin/activate            # only if setup.sh created one
./shadow_rl/run_all.sh --pairs nosearch          # ~2h on one H200, no retrieval needed
```

Nothing is written to `$HOME`. The virtualenv, the Search-R1 checkout
(`third_party/Search-R1`), verl (`third_party/verl`) and the BM25 corpus
(`corpus/`) all live in the working tree, so a box where `/root` is small or
not writable is fine. There is no `SEARCH_R1_ROOT` to export; set one only to
point at a checkout somewhere else. Models still go to the standard
HuggingFace cache, which is shared with everything else on the machine —
`SHADOW_RL_MODEL_DIR` or `HF_HOME` moves those.

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
| `launch_retriever.sh` | corpus + index + retrieval server: e5, e5-hnsw or bm25 |
| `report.py` | terminal comparison table — printed automatically after every run |
| `discover.py` | probes HuggingFace for unreleased-but-plausible pairs |
| `check_env.py` | validates torch/torchvision/transformers and prints exact fixes |
| `run_eval_parallel.py` | spreads one pair's (role, dataset) cells across the GPUs |
| `diagnose.py` | groups the failures in `logs/` by cause and names the fix |
| `paths.sh` / `paths.py` | where checkouts live; keeps shell and python agreeing |
| `setup_search_r1.sh` | clones the eval harness into `third_party/` |
| `setup_bm25.sh` | JDK + faiss + pyserini + a `JAVA_HOME` that survives the shell |
| `SETUP.md` | the environment setup as explicit copy-pasteable commands |
| `tests/` | CPU-only tests: merge arithmetic, similarity stats, eval contract, resume logic |

The tests need no GPU and no model:

```bash
python shadow_rl/tests/test_merge.py
python shadow_rl/tests/test_similarity.py
python shadow_rl/tests/test_evaluate.py
python shadow_rl/tests/test_resume.py
python shadow_rl/tests/test_paths.py
python shadow_rl/tests/test_parallel.py
```

## Troubleshooting

Run this first — it is also run automatically during `run_all.sh` preflight:

```bash
python shadow_rl/check_env.py --full
```

**`Could not import module 'Qwen2ForCausalLM'`** — almost always a CUDA
mismatch between torch and one of its companion packages (`torchvision`,
`torchaudio`, `torchcodec`), not a transformers problem. pip installs torch from
the CUDA-specific PyTorch index but the companions from default PyPI, which may
be built against a different CUDA version; transformers imports them deep inside
`image_utils`, so the first symptom is an unrelated-looking model-class error.

Read the message to see **which** package is named — reinstalling the wrong one
achieves nothing. `check_env.py` extracts it for you:

```bash
pip install --force-reinstall torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use the CUDA version your torch reports
(`python -c 'import torch; print(torch.version.cuda)'`).

**Do not uninstall torchvision to work around this.** transformers tolerates its
absence, but vllm does not: kernel warmup imports
`torchvision.transforms` unconditionally, so removing it trades a CUDA mismatch
for `ModuleNotFoundError: No module named 'torchvision'` inside EngineCore.
Reinstall it matching your torch instead. `torchaudio` and `torchcodec` are not
needed by anything here, so uninstalling those is a legitimate fix.

The check is **advisory**: `run_all.sh` reports problems and continues. Use
`--strict-env` to make it abort, or `--skip-env-check` to skip it entirely.

**Merge stopped on a key mismatch** — see *Merging over mismatched key sets*
below; the current default merges over the intersection and does not stop.

## `run_all.sh`

| option | meaning |
|---|---|
| `--pairs X` | groups (see below) or explicit pair ids |
| `--stages X` | subset of `similarity,merge,eval,aggregate`; default omits `similarity` |
| `--roles X` | subset of the four model roles |
| `--limit N` | first N questions per dataset — biased, smoke runs only |
| `--sample N` | deterministic random N per dataset, identical across roles |
| `--skip-datasets X` | omit datasets entirely; published averages are restricted to match |
| `--auto-retriever` | start and stop the retrieval server automatically |
| `--retriever X` | `auto` (default), `e5`, `e5-hnsw` or `bm25`; only bm25 needs Java |
| `--cleanup` | delete a pair's RL checkpoints and merged model once it is evaluated |
| `--fast` | smallest useful run: one 3B pair, nq+hotpotqa, 200 questions, 5 models |
| `--tp N` | tensor parallel size (default 1; an H200 fits 7B at 1) |
| `--force` | redo work already complete (merge, smoke test) |
| `--skip-env-check` | do not run the environment check |
| `--strict-env` | abort if the environment check reports problems |
| `--dry-run` | print the plan and stop |
| `--yes` | skip the confirmation prompt |

Everything is resumable, and the plan line reports what will be reused before
anything runs:

- **Merged models** are reused only when *complete* — `shadow_merge_stats.json`
  present (merge.py writes it last) and every shard named in the index actually
  on disk. A directory left half-written by an interrupted run is detected and
  redone rather than silently reused; a bare `config.json` no longer counts.
- **Smoke tests** are cached per merged model via a `.smoke_ok` marker, and
  invalidated automatically if the model is re-merged after it.
- **Finished datasets** are detected in `results.csv` and skipped.
- **Similarity** skips pairs already in `similarity_summary.csv`.

Pass `--force` to redo work that is already complete. A failing pair is logged and the run moves to the next
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

### Simplest possible check

One pair, one dataset, 200 questions, five models — enough to confirm the whole
path runs:

```bash
./shadow_rl/run_all.sh --pairs demo --datasets nq --sample 200 \
    --auto-retriever --yes
```

`demo` is `grpo-search-3b-v0.3`. About **6 min on 1 GPU, 1 min on 8** — the GPUs
are detected automatically and the work is spread across them.

### Using 1 GPU or 8

A 3B or 7B model fits on one GPU, so tensor parallelism buys nothing. The win on
a multi-GPU box comes from running independent `(role, dataset)` cells side by
side, which is what `--jobs auto` (the default) does: it pins one worker per GPU
via `CUDA_VISIBLE_DEVICES`, each writes its own CSV shard, and the shards are
merged at the end. Separate shards on purpose — concurrent appends to one CSV
are not reliably atomic, and a torn row is a silently corrupt result.

```bash
./shadow_rl/run_all.sh --pairs demo --datasets nq --yes            # uses every GPU it sees
./shadow_rl/run_all.sh --pairs demo --datasets nq --jobs 4 --yes   # cap at 4 workers
CUDA_VISIBLE_DEVICES=0,1 ./shadow_rl/run_all.sh --pairs demo --yes # restrict to 2
./shadow_rl/run_all.sh --pairs 14b-pair --tp 2 --yes               # a 14B/32B model needs tp>1
```

`--tp` is per worker, so `--tp 2` on 8 GPUs gives 4 workers. Asking for more than
the machine has is refused up front rather than failing inside vLLM. With one
GPU the runner falls back to the sequential path, which loads each model once
instead of once per dataset.

### When the parallel run fails

Almost nothing breaks *one* worker. A vLLM engine that will not start, a model
that will not load, a retriever that is not listening — these break every worker
identically, so the run's job is to say so once, early, and stop.

- **Preflight** rejects a missing `--shadow-path` and a silent retriever before
  any model is loaded.
- **A canary cell runs alone first.** If it fails, the other jobs are never
  started and its log is printed.
- **Every failure prints its own cause inline** — the traceback, or the OOM
  line, or the refused connection — not a bare `FAIL` and a path.
- **`--fail-fast N`** (default 3) stops after N consecutive failures. Thirty
  identical errors take an hour and teach nothing. `--fail-fast 0` to push on.
- **`--stagger SEC`** (default 15) spaces out worker starts. N vLLM engines
  initialising at the same instant contend for ports, the HuggingFace cache lock
  and host RAM, which turns a working configuration into a flaky one.
- **`--job-timeout MIN`** kills a cell that hangs instead of blocking the queue.

For a run that already failed, the logs are still there:

```bash
python shadow_rl/diagnose.py           # group every failing log by cause, with fixes
python shadow_rl/diagnose.py --full    # every log, not one per cause
```

It reads `shadow_rl/logs/*.log`, collapses identical failures, and names the fix
for the ones with a known remedy (missing vllm or faiss, OOM, dead retriever, a
JVM that will not start, a cu12 torch on a Blackwell card, a worker killed by
the host OOM killer).

### The retriever: dense by default, no Java

`bm25` is the only retriever that needs a JVM — `retrieval_server.py` imports
pyserini inside `BM25Retriever.__init__`, so the dense paths never touch Java at
all. If the JVM has been fighting you, switching is the fix, and it is also the
*more* faithful choice: E5 is what Search-R1 used, so absolute EM becomes
comparable to the published numbers instead of sitting below them.

```bash
./shadow_rl/launch_retriever.sh                       # auto (default)
./shadow_rl/launch_retriever.sh --retriever e5-hnsw   # CPU, faiss-cpu, no Java
./shadow_rl/launch_retriever.sh --retriever e5        # exact, wants faiss-gpu
./shadow_rl/launch_retriever.sh --retriever bm25      # sparse, needs a JVM
./shadow_rl/launch_retriever.sh --check               # dependencies only
```

| | index | needs | notes |
|---|---|---|---|
| `e5` | `e5_Flat.index` | faiss-**gpu** | exact; what the paper used |
| `e5-hnsw` | `e5_HNSW64.index` | faiss-cpu | approximate, CPU-fast, no Java |
| `bm25` | `bm25/` | pyserini + JDK 21 | no GPU, but the JVM is the fragile part |

`auto` resolves to `e5` when faiss-gpu is installed and a GPU is visible, and to
`e5-hnsw` otherwise. It never picks `bm25` — Java is opt-in, not a fallback you
land in by accident. `run_all.sh --retriever X` passes the choice through.

Do not mix retrievers within one `results.csv`: search-pair EM is not comparable
across them. `run_all.sh` records the retriever in `results.retriever` and warns
if a later run disagrees.

If you do want BM25, the JDK/faiss/pyserini setup is one command:

```bash
./shadow_rl/setup_bm25.sh            # install + verify + persist JAVA_HOME
./shadow_rl/setup_bm25.sh --check    # verify only, changes nothing
```

- **`No module named 'faiss'` after installing `faiss-cpu`.** It went to a
  different interpreter. `--check` prints the one the server will actually use,
  and the `pip` line for it.
- **`JVM failed to start` / `dlopen(.../lib/jvm/lib/server/libjvm.so)`.**
  pyserini loads Lucene through jnius, which takes its path from `JAVA_HOME`.
  conda's openjdk lives at `$CONDA_PREFIX/lib/jvm`, so jnius tries that even in
  an environment where the JDK came from apt instead. The launcher now searches
  `$JAVA_HOME`, `$CONDA_PREFIX`, `which java` and `/usr/lib/jvm/*` and picks the
  first that genuinely contains `libjvm.so` — existing is not the same as
  usable.

The runtime estimate in the plan accounts for the worker count:

| run | 1 GPU | 8 GPUs |
|---|---|---|
| `--pairs demo --datasets nq --sample 200` | ~6 min | ~1 min |
| `--pairs all,-v0.2`, 6 datasets, full sets | ~55 h | ~8 h |

### Start here: `--fast`

The smallest run that still answers the question — one 3B pair, four datasets
(NQ and HotpotQA in-domain, Musique and Bamboogle out-of-domain), 200 sampled
questions, all five models:

```bash
./shadow_rl/run_all.sh --fast --yes            # ~4 min of generation
```

That covers `W_B` (raw HF base), `W_I` (raw HF instruct), `RL(W_I)`, `RL(W_B)`
and `W_shadow`, and gives both in-domain and generalisation signal. It proves
the pipeline runs end to end and shows the sign of the effect. It is **not**
publishable: at 200 questions the standard error on one EM number is around
±0.03, wider than the margins being measured.

Add a GRPO pair — note this one needs the retrieval server, so the first run
also downloads ~70 GB of corpus and index:

```bash
./shadow_rl/setup.sh --with-bm25
./shadow_rl/run_all.sh --fast --pairs grpo-search-3b-v0.3 --auto-retriever --yes
```

Then the real alignment run, still with no retrieval server:

```bash
./shadow_rl/run_all.sh --pairs nosearch --sample 1000 --yes    # ~2 h
```

Two pairs, both sizes, **no retrieval server and no 70 GB index download**. This
is also the setting the paper says base-init wins (0.229 vs 0.224 at 3B, 0.276 vs
0.271 at 7B) — so it is exactly where grafting should show up. You get the
harness-validation diff against the published numbers *and* the first real
`shadow` number in the same run.

| run | pairs | time, 1× H200 |
|---|---|---|
| `--fast` | 1 | **~2 min** |
| `--pairs nosearch --sample 1000` | 2 | **~2 h** |
| `--pairs nosearch` (full sets) | 2 | ~13.7 h |
| `--pairs 3b --sample 500` | 7 | ~3.7 h |
| `--pairs all --sample 500` | 12 | ~9.9 h |

**The expensive component is the search pairs, not any one dataset** — they are
10 of the 12, their rollouts are 2–5× slower, and they need the corpus and index.
Dropping them (`--pairs nosearch`) is what turns a week into an hour.

Note that once you pass `--sample N`, dataset size stops mattering: every dataset
contributes N questions, so `--skip-datasets popqa` saves a seventh of the time
rather than the 28% of questions popqa represents at full size. Use
`--skip-datasets` only if you want to drop a set for a reason other than speed.

### Sampling

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

## Where things are stored

| what | where | override |
|---|---|---|
| downloaded models | the standard HuggingFace cache (`HF_HOME`, default `~/.cache/huggingface/hub`) | `SHADOW_RL_MODEL_DIR` |
| merged models | `shadow_rl/merged/` | `MERGED_DIR` |
| BM25 corpus + index (~70 GB) | `corpus/` in the working tree | `CORPUS_DIR` |
| verl parquets | `datasets/` in the working tree | `--out` on `prepare_data.py` |
| results, logs | `shadow_rl/` and `shadow_rl/logs/` | `RESULTS`, `LOG_DIR` |

Models use the shared HuggingFace cache so they are downloaded once and reused
by anything else on the box. The corpus and index are the large, single-purpose
downloads, so they live under the working tree rather than filling `/root`.

The HuggingFace *dataset* cache follows `HF_HOME` like the models. To put it in
the working tree as well:

```bash
export HF_DATASETS_CACHE=$PWD/hf_datasets
```

## Selecting pairs

`--pairs` takes either explicit ids or group names. Several groups **intersect**,
so the selection narrows rather than widens:

| group | meaning |
|---|---|
| `all` | every pair in the manifest |
| `v0.1` `v0.2` `v0.3` `latest` | by release version; `latest` aliases `v0.3` |
| `grpo` `ppo` | by algorithm |
| `search` `nosearch` | whether a retrieval server is needed |
| `3b` `7b` `14b` | by scale |
| `qwen` `llama` | by backbone family |

```bash
./shadow_rl/run_all.sh --pairs v0.3 --yes           # only the current recipe
./shadow_rl/run_all.sh --pairs v0.3,grpo --yes      # v0.3 AND grpo
./shadow_rl/run_all.sh --pairs v0.3,llama --yes     # v0.3 AND a Llama backbone
```

Mixing a group name with a pair id is rejected rather than guessed at.

## Adding pairs that are not in the manifest

The 12 shipped pairs are the ones confirmed released. Search-R1's v0.3 scripts
also cover Qwen-14B and DeepSeek distills, and its v0.1 scripts cover
Llama-3.2-3B and Llama-3.1-8B — but a training script is not a release, and
grafting needs **both** sides of a pair published.

`pairs.CANDIDATES` lists ids inferred from the naming convention. They are not
assumed to exist; `discover.py` asks the Hub:

```bash
python shadow_rl/discover.py                    # report what exists
python shadow_rl/discover.py --write            # record it into the manifest
python shadow_rl/discover.py --sweep --write    # probe the whole grid, every version
python shadow_rl/discover.py --also-originals   # check the base/instruct too
```

**Version tags in repo names.** v0.1 predates the convention and carries *no*
suffix — `SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`. v0.2 and v0.3 append
theirs. So an unsuffixed repo name is a **v0.1** checkpoint, not a newer one, and
the three unsuffixed pairs are already in the manifest as `grpo-search-3b-v0.1`,
`ppo-search-3b-v0.1` and `ppo-search-7b-v0.1`. `--sweep` fills the v0.1 grid in
too, since the original list omitted 7B GRPO and the 14B pairs.

Confirmed pairs land in `verified_pairs.json`, which `pairs.py` merges on import,
so they become selectable by every group above. A pair with only one side
released is reported as incomplete and **not** added — the delta needs the base
side and the comparison needs the instruct side.

Llama originals (`meta-llama/Llama-3.2-3B`, `Llama-3.1-8B`) are **gated** on
HuggingFace: accept the licence on the model page or `huggingface-cli login`
first, or the download fails. `discover.py` reports a gated repo as existing
rather than missing.

Everything downstream is backbone-agnostic — config, tokenizer and chat template
all come from `W_I` — so a Llama pair needs no code changes. Disk and runtime
estimates are derived from the parameter count in the size key, so a 14B or 8B
pair is costed correctly rather than assumed to be 3B.

## The two extra datasets

GPQA-Diamond and SimpleQA are not part of the Search-R1 protocol, so neither has
published reference numbers and both are excluded from the harness-validation
table.

**GPQA-Diamond** is multiple choice. Options are rendered into the question and
labelled with **digits, not letters** — `qa_em.normalize_answer` strips English
articles, so a gold label of `"A"` normalises to the empty string, and an empty
gold compares equal to any prediction that also normalises to empty (`"the"`,
`"an"`). That would have scored junk answers correct on every question whose
answer happened to be option A. Digits pass through the normaliser untouched.
Either the option number or the answer text scores as correct. The repo is gated;
accept the licence or `huggingface-cli login` first.

As a general guard, any gold answer that normalises to nothing is dropped before
scoring, and the count is reported.

**SimpleQA** is graded with **substring exact match**, not strict EM. Its own
protocol uses an LLM judge, which would mean depending on a grader model; strict
EM is the opposite extreme and fails on `"in 1963"` against a gold of `"1963"`.
Sub-EM is the fixed, reproducible middle — same normaliser, but the gold only has
to appear in the prediction — and it comes from the official `qa_em` module
rather than being reimplemented here. It is more permissive than strict EM, so
treat SimpleQA numbers as a different metric from the other columns, not a
comparable one.

## Parameter-level similarity

**Not run by default.** It streams four whole checkpoints per pair to compute
per-parameter σ, which is diagnostic rather than part of the result, so it is
opt-in:

```bash
./shadow_rl/run_all.sh --pairs demo --stages similarity,merge,eval,aggregate --yes
./shadow_rl/run_all.sh --pairs all --stages similarity --yes     # just the stats, no GPU
```

The merge still prints σ and the base-side delta magnitude — those are
accumulated during the merge pass itself and cost nothing extra. Only the
instruct-side delta needs a separate pass, and `run_all.sh` no longer requests
it; pass `--rl-instruct` to `merge.py` directly if you want that figure.

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
./shadow_rl/launch_retriever.sh               # serves :8000, leave running
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

## Merging over mismatched key sets

Exports of the same model legitimately disagree on bookkeeping keys. The common
case here: Qwen2.5-3B sets `tie_word_embeddings`, so the official release stores
no `lm_head.weight`, while verl's export of the RL checkpoints materialises the
tied copy.

`merge.py` therefore merges over the **intersection** by default, reports exactly
what it skipped, and — for a dropped `lm_head` — checks whether it is still
identical to `model.embed_tokens.weight` in each checkpoint:

```
[warn] key sets differ; merging over the 434 common keys.
[warn] skipping 1 key(s): lm_head.weight
    rl_base: lm_head is identical to model.embed_tokens.weight (tied) -- safe to drop
```

That check is the point. Dropping a tied `lm_head` is correct, because the merged
model re-ties from `W_I`'s config. But if RL had *untied* it and trained it
separately, silently dropping the key would discard part of the update — so the
script says so loudly instead:

```
    ** rl_base: lm_head DIFFERS from model.embed_tokens.weight (max |diff| 0.0031).
       RL appears to have untied it; dropping the key discards that update.
```

Pass `--strict-keys` to restore the old behaviour of failing on any difference.

## Checking a run in progress

`results.csv` is written one dataset at a time, so both views are live — safe to
run in another shell while the job is going:

```bash
python shadow_rl/report.py --progress    # completion grid + % done
python shadow_rl/report.py               # the numbers scored so far
```

Interrupting is safe. Ctrl-C, then re-run the identical command: finished
datasets are skipped, complete merged models are reused, and passed smoke tests
are not repeated. Nothing is recomputed.

`git pull` during a run is also safe — git replaces files by rename, so the
running shell keeps reading the old copy. (An in-place edit, `sed -i` style,
would not be.)

## Reading the result

`run_all.sh` prints a comparison table when it finishes; re-print it any time
without re-running anything:

```bash
python shadow_rl/report.py                                  # every pair
python shadow_rl/report.py --pair ppo-nosearch-3b-v0.2      # one pair
python shadow_rl/report.py --no-color > results.txt         # for pasting
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ppo-nosearch-3b-v0.2   PPO · 3B · no search · v0.2   4/7 datasets · n=200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  model                                  NQ* HotpotQA*   Musique Bamboogle      Avg
  ──────────────────────────────────────────────────────────────────────────────────
  W_B       untuned base               0.045     0.047     0.011     0.050    0.038
  W_I       untuned instruct           0.113     0.092     0.021     0.111    0.084
  RL(W_I)   direct RL on instruct      0.205     0.188     0.069     0.199    0.165
  RL(W_B)   direct RL on base          0.237     0.211     0.067     0.211    0.182
  W_shadow  ours                       0.251     0.235     0.070     0.240    0.199

  WIN   shadow − RL(W_I)  = +0.0340    (0.1991 vs 0.1652)
  WIN   shadow − RL(W_B)  = +0.0175    (0.1991 vs 0.1816)

  harness check:  RL(W_B) 0.182 vs 0.177 published (+0.005) ok | ...
```

Each average carries a `±se` — the sampling standard error, from the Bernoulli
variance of EM at the question count actually used. Every margin is then judged
against it:

| verdict | meaning |
|---|---|
| `solid (|d| > 2×se)` | margin clears roughly 95% confidence |
| `weak` | margin is between 1 and 2 se — suggestive, needs more questions |
| `WITHIN NOISE` | margin is smaller than its own standard error — not a result |

This matters at small `--sample`: at n=200 over 4 datasets the error on a margin
is around ±0.02, so a `+0.005` difference is indistinguishable from zero. The
bound is conservative — both models are scored on the *same* sampled questions,
so the true paired error is smaller — but if a margin does not clear it, raise
`--sample` before believing it.

`*` marks in-domain sets, `!` marks a role missing data the others have, and the
best value in each column is bolded. With two or more pairs an overview table is
appended tallying how often `W_shadow` beat `RL(W_I)`.

`FINDINGS.md` now opens with a **Verdict** section stating that tally directly,
before the detailed tables.

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
