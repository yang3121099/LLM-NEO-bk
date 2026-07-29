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

## Layout

| file | what it does |
|---|---|
| `merge.py` | streaming safetensors merge + sigma / delta-magnitude sanity stats |
| `pairs.py` | manifest of the 12 pairs and the published reference numbers |
| `evaluate.py` | Search-R1 rollout + official EM scoring → `results.csv` |
| `aggregate.py` | `results.csv` → `FINDINGS.md` |
| `smoke_test.py` | does the merged model load and generate coherent text? |
| `run_pair.sh` | one pair end to end: merge → smoke test → four roles → findings |
| `launch_bm25_retriever.sh` | corpus + BM25 index + retrieval server (search pairs only) |
| `tests/` | CPU-only tests for the merge arithmetic and the eval contract |

## Setup

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git ~/Search-R1
export SEARCH_R1_ROOT=~/Search-R1
pip install torch safetensors huggingface_hub transformers datasets vllm requests
```

The tests need no GPU and no model:

```bash
python shadow_rl/tests/test_merge.py
python shadow_rl/tests/test_evaluate.py --search-r1-root $SEARCH_R1_ROOT
```

## Order of work

The `R1-*` pairs are RL **without** a search engine, so they need no retrieval
server and no index. They are both the cheapest experiment and the setting where
the base-initialised model wins — start there.

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

## Cost

Rough disk and GPU needs, per pair. The RL checkpoints are frequently fp32, so
they are about twice the size of the official bf16 weights.

| | 3B pair | 7B pair |
|---|---|---|
| download | ~40 GB | ~90 GB |
| merged model | ~6 GB | ~15 GB |
| GPU | 1×24 GB | 1×80 GB or 2×40 GB (`TP=2`) |

The search pairs additionally need ~70 GB for the Wikipedia corpus and BM25
index, downloaded once and shared across pairs.
