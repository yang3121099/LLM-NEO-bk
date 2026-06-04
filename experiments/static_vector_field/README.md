# Static Post-Training Vector-Field Audit

A **pure static weight-space** analysis of a single model lineage:

```
Base → SFT → DPO → RLVR
```

For Llama-3.1-8B / Tulu-3 the four stages are:

| stage | checkpoint |
|-------|------------|
| base | `meta-llama/Llama-3.1-8B` |
| sft  | `allenai/Llama-3.1-Tulu-3-8B-SFT` |
| dpo  | `allenai/Llama-3.1-Tulu-3-8B-DPO` |
| rlvr | `allenai/Llama-3.1-Tulu-3-8B` |

This experiment **does not** use any downstream-task data, **does not** train,
**does not** evaluate, and **does not** compute a loss or gradient. It only reads
checkpoint weights and characterises the *post-training vector field* in parameter
space: per-stage deltas, direction consistency, path curvature, subspace structure,
and from those a **static rollback candidate**.

It complements the Shadow-FT finding that Base/Instruct weights are highly similar:
here we ask whether the `Base→SFT→DPO→RLVR` path is close to a straight line, and
which stages / layers / modules carry *curved*, *residual*, or *late-stage-specific*
deltas — candidates for a future "plasticized shadow position".

> ⚠️ **Interpretation caveat.** Everything here is geometry of weights. A high
> curvature / residual / rollback score flags a candidate for further study; it is
> **not**, on its own, a claim about downstream performance.

## Layout

```
experiments/static_vector_field/
  configs/llama31_8b_lineage.yaml   # lineage paths + analysis options
  src/
    hf_checkpoint_io.py    # streaming sharded-safetensors / bin reader
    grouping.py            # param name -> {global,layer,module,...} group keys
    accumulators.py        # float64 streaming accumulators per group
    metrics.py             # accumulators -> metric CSV tables
    analyze_lineage.py     # orchestrator: stream -> CSVs + summary.md
    plot_lineage.py        # CSVs -> figures/*.png
    build_static_shadow.py # rollback scores -> static delta checkpoint
  tests/
    test_grouping.py       # pure-stdlib grouping tests
    test_metrics_toy.py    # closed-form numeric checks of the metric layer
```

## How it works

The analyzer never materialises whole-model vectors. It builds a
`tensor name → shard file` index for each checkpoint, then for every shared
floating tensor:

```python
w0,w1,w2,w3 = load(base), load(sft), load(dpo), load(rlvr)   # flat float64
v_sft  = w1 - w0      # SFT step
v_dpo  = w2 - w1      # DPO step
v_rlvr = w3 - w2      # RLVR step
v_post = w3 - w0      # Base→RLVR chord
update_accumulators(groups_of(name), w*, v*)   # float64 dot/norm/abs sums
del w0,w1,w2,w3,v_*                            # release immediately
```

All metrics are then derived from the accumulated sums — so memory stays bounded
to a single tensor at a time regardless of model size.

### Grouping granularities

`global`, `layer:N`, `module:{attn,mlp,norm,embed,lm_head,other}`,
`layer_module:N:M`, `submodule:{q_proj,…,lm_head}`, and
`depth_bin:{shallow,middle,deep}`. A tensor contributes to every group it belongs
to. (`norm` is matched before `attention` so `post_attention_layernorm` is a norm.)

## Metrics (CSV outputs)

| file | content |
|------|---------|
| `pairwise_sigma.csv` | relative gap ratio σ = Σ\|θb−θa\| / (Σ\|θa\|+Σ\|θb\|) per adjacent + Base→RLVR pair |
| `stage_norm_budget.csv` | ‖vᵢ‖, rel-L2 vs previous checkpoint, stage energy fraction, L1, L∞ |
| `pairwise_cosine.csv` | cosines between the 4 stage vectors (6 pairs) |
| `path_geometry.csv` | L_path, L_chord, straightness, cancellation, turning angles, curvature |
| `chord_projection.csv` | progress along chord, chord alignment, residual-to-chord |
| `subspace_rank.csv` | 3×3 Gram eigenvalues, PC1 energy, effective rank |
| `orth_residual.csv` | DPO residual after removing SFT; RLVR residual after removing span(SFT,DPO) |
| `topk_overlap.csv` | approximate per-tensor top-k absolute-coordinate overlap (marked approximate) |
| `rollback_scores.csv` | per (group, late-stage) z-scored rollback score |
| `top_rollback_candidates.csv` | highest-scoring rollback candidates |
| `summary.md` | auto-generated human-readable report |

The rollback score (z-scored within each granularity):

```
score = z(residual_to_chord) + z(orth_residual_ratio) + z(curvature)
      + z(cancellation) + z(stage_fraction) - z(chord_align)
```

## Usage

### One command (recommended)

From the repo root, set up a **minimal** environment (only torch, numpy,
safetensors, matplotlib, pyyaml — no LlamaFactory / OpenCompass):

```bash
bash setup_env_static.sh        # creates the `factory` conda env + minimal deps
conda activate factory
bash run_static_vector_field.sh                 # analyze -> plot -> shadow
```

> The full Shadow-FT pipeline setup still lives in `setup_env.sh`; this experiment
> only needs `setup_env_static.sh`. Useful flags: `--cpu` (CPU-only torch wheel),
> `--no-conda` (install into the active interpreter), `--env NAME`, `--skip-conda`.

`run_static_vector_field.sh` activates the `factory` env, makes sure the extra deps
are present, then runs all steps. Useful flags:

```bash
bash run_static_vector_field.sh --device cuda            # use the GPU path
bash run_static_vector_field.sh --steps analyze --limit 50   # quick smoke test
bash run_static_vector_field.sh --config path/to/other.yaml
```

### Checkpoints: local dir or HF Hub

Each `path` in the config may be a local directory **or** a HuggingFace Hub repo
id (the default config uses repo ids). Hub repos are fetched on demand — only the
`*.safetensors` / `*.json` files — via `snapshot_download`. You can pin a commit
with `revision: <sha>` per lineage entry.

`meta-llama/Llama-3.1-8B` is **gated**: accept its license on the model page with
the same account whose token you use, or the download returns 403.

### Authenticating with HuggingFace (do NOT commit your token)

Your token is a secret — never put it in the config, a script, or git. Use one of:

```bash
# Recommended: interactive login (token is pasted at the prompt, stored in
# ~/.cache/huggingface/token, which is outside the repo and not committed).
huggingface-cli login

# Or: an environment variable, set in your shell session only.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`snapshot_download` picks either up automatically. Prefer a **read-only**
fine-grained token scoped to just the models you need. If a token is ever exposed
(e.g. pasted into a chat or a log), revoke it at
<https://huggingface.co/settings/tokens> and issue a new one.

### Manual / per-step

```bash
cd experiments/static_vector_field
# 1. edit configs/llama31_8b_lineage.yaml with local checkpoint paths
python src/analyze_lineage.py     --config configs/llama31_8b_lineage.yaml
python src/plot_lineage.py        --config configs/llama31_8b_lineage.yaml
python src/build_static_shadow.py --config configs/llama31_8b_lineage.yaml
```

`--limit N` on `analyze_lineage.py` processes only the first N tensors.

## CPU vs GPU

The audit is **largely I/O bound**: the dominant cost is streaming the four
checkpoints' shards off disk (~64 GB for 8B bf16 ×4), and the per-tensor work is
just `dot / norm / abs / sign / top-k` reductions. A GPU therefore gives a
**moderate** speedup (mostly on fast NVMe), not an order of magnitude.

Set `device: cuda` in the config (or `--device cuda` on the run script) to enable
the GPU path: per-tensor reductions run on-device while cross-tensor sums stay
**float64 on the host**, so accumulation precision is unchanged. `reduce_dtype`
selects the on-device reduction precision (`float32` default — far faster than
`float64` on consumer GPUs). If `cuda` is requested but no GPU is visible, the run
falls back to CPU with a warning.

## Tests

```bash
python tests/test_grouping.py
python tests/test_metrics_toy.py
# or, if pytest is available:
pytest tests/
```

`test_grouping.py` is pure stdlib; `test_metrics_toy.py` feeds hand-chosen tensors
with known geometry (orthogonal vs. colinear paths) and checks curvature,
straightness, residual-to-chord, subspace rank, etc. against closed-form values.

## Relationship to the rest of the repo

This experiment is self-contained under `experiments/static_vector_field/` and does
not touch the existing Shadow-FT training / grafting code. It reuses the same
weight-reading idea as the repo's `weight_similarity_analysis.py` but generalises it
to a 4-stage streaming vector-field audit.
