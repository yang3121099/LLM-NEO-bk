# Environment setup, spelled out

`setup.sh` does all of this for you. This page is the same thing as plain
commands, for when you would rather run them yourself, adapt them, or see
exactly what is being installed.

Two paths: **conda** (self-contained, recommended for a fresh box) or **venv**
(if the system Python is already what you want). Pick one.

**Nothing lands in `$HOME`.** Checkouts, the virtualenv and the corpus all go
into the working tree, because on a container `$HOME` is frequently `/root` —
small, or not writable at all — and a clone that fails there surfaces much later
as something that looks unrelated. Only the HuggingFace model cache stays where
it normally is, so models are shared with everything else on the machine.

---

## 0. What the machine needs

| | why |
|---|---|
| NVIDIA driver + GPU | evaluation and training; merge/similarity run on CPU |
| **CUDA 13.0 build of torch** if the GPU is **B200/B300** | Blackwell is `sm_100`; a cu12 wheel has no kernels for it and fails at every launch, not at import |
| ~130 GB disk with `--cleanup`, ~600 GB without | fp32 RL checkpoints are ~2× the bf16 originals |
| Java 21 + ~70 GB | only for the BM25 retrieval used by the `SearchR1-*` pairs |

Check what you have:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# compute_cap 10.x or 12.x  -> Blackwell, use CUDA 13.0
# compute_cap 9.0           -> H100/H200,  CUDA 12.8 is fine
```

---

## 1. Conda path

```bash
# --- environment -----------------------------------------------------------
conda create -y -n shadow-rl python=3.11
conda activate shadow-rl

# --- torch: pick the index matching your GPU -------------------------------
# B200 / B300 (Blackwell, sm_100):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# H100 / H200 (Hopper, sm_90) instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# torchvision must come from the SAME index as torch. transformers imports it
# inside image_utils, and a CUDA mismatch surfaces much later as a confusing
# "Could not import module 'Qwen2ForCausalLM'".

# --- the rest ---------------------------------------------------------------
pip install -U pip setuptools numpy
pip install safetensors huggingface_hub transformers datasets accelerate requests pandas pyarrow
pip install vllm

# --- the evaluation harness -------------------------------------------------
./shadow_rl/setup_search_r1.sh       # -> third_party/Search-R1, nothing to export

# --- verify -----------------------------------------------------------------
python shadow_rl/check_env.py --full
```

The harness is read from, never installed: `evaluate.py` imports `qa_em.py` and
the generation loop by path. Do **not** `pip install -e` the Search-R1 checkout —
it pins its own torch and vllm and will undo the two lines above.

`check_env.py` compares your GPU's compute capability against
`torch.cuda.get_arch_list()`, which is the decisive test — not the CUDA version
string.

## 1b. venv path

Identical, only the first two lines differ:

```bash
python3 -m venv .venv          # in the working tree, not $HOME
source .venv/bin/activate
```

---

## 2. BM25 retrieval — only for the `SearchR1-*` pairs

The `R1-*` (no-search) pairs skip this entirely.

```bash
# Java 21: pyserini wraps Lucene, and an older JDK fails at query time
apt-get install -y openjdk-21-jdk-headless
# no root? conda works too:
# conda install -y -c conda-forge openjdk=21 maven

pip install -U faiss-cpu pyserini

# corpus + index, ~70 GB, into the working tree (not $HOME)
./shadow_rl/launch_bm25_retriever.sh          # downloads then serves on :8000
```

Leave that running in its own shell, or let `run_all.sh --auto-retriever` start
and stop it for you.

If `import pyserini` fails, check `JAVA_HOME`:

```bash
java -version
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
```

---

## 3. verl — only for the math track

```bash
./shadow_rl/verl_math/setup_verl.sh
```

That clones verl to `third_party/verl` **inside the working tree** (gitignored,
~30 MB), pip-installs it editable, and adds only the packages it actually
imports. Nothing to export afterwards — `import verl` works from anywhere, so
neither `train_grpo.sh` nor `export_hf.py` needs a path.

The install is deliberately `--no-deps` plus a hand-picked list. verl's own
`requirements.txt` pins `transformers>=5.5.3,<5.11` and pulls torch
transitively, so letting pip resolve it reinstalls the default-CUDA torch over
the cu130 wheel vllm is working against — a working environment turns into
`no kernels for sm_100` for no visible reason.

If you would rather do it by hand, that is the whole of it:

```bash
git clone --depth 1 https://github.com/volcengine/verl.git third_party/verl
python -m pip install -e third_party/verl --no-deps
python -m pip install -U 'ray[default]' tensordict omegaconf hydra-core \
    codetiming dill torchdata pylatexenc datasets pillow accelerate \
    pyarrow pandas tqdm
python shadow_rl/verl_math/setup_verl.sh --check   # or: ./…/setup_verl.sh --check
```

`wandb` is intentionally absent: verl's default `trainer.logger` is
`["console", "wandb"]`, which stops at an auth prompt on a machine that has
never logged in, so `train_grpo.sh` passes `trainer.logger=[console]`. Set
`LOGGER='[console,wandb]'` to opt back in.

---

## 4. Where things land

| what | where | override |
|---|---|---|
| downloaded models | standard HuggingFace cache (`~/.cache/huggingface/hub`) | `SHADOW_RL_MODEL_DIR` |
| merged models | `shadow_rl/merged/` | `MERGED_DIR` |
| BM25 corpus + index | `corpus/` in the working tree | `CORPUS_DIR` |
| Search-R1 checkout | `third_party/Search-R1` in the working tree | `SEARCH_R1_ROOT` |
| verl checkout | `third_party/verl` in the working tree | `VERL_ROOT` |
| virtualenv | `.venv/` in the working tree | `VENV` |
| verl parquets | `datasets/` in the working tree | — |
| verl checkpoints | `shadow_rl/verl_math/ckpt/` | `CKPT_DIR` |
| results, logs | `shadow_rl/`, `shadow_rl/logs/` | `RESULTS`, `LOG_DIR` |

To keep the HuggingFace *dataset* cache in the working tree as well:

```bash
export HF_DATASETS_CACHE=$PWD/hf_datasets
```

---

## 5. Put it in your shell profile

Only the environment activation is worth persisting — the checkout locations are
resolved from the repo, so there is nothing else to export:

```bash
echo 'conda activate shadow-rl' >> ~/.bashrc     # or: source <repo>/.venv/bin/activate
```

---

## 6. First run

```bash
python shadow_rl/check_env.py --full          # should be all green

# smallest thing that proves the pipeline works: one pair, one dataset
./shadow_rl/run_all.sh --pairs demo --datasets nq --sample 200 \
    --auto-retriever --yes

python shadow_rl/report.py
```

The CPU test suites need no GPU and no model, and are worth running once after
setup:

```bash
for t in merge similarity resume check_env report pairs paths; do
    python shadow_rl/tests/test_$t.py | tail -1
done
python shadow_rl/tests/test_evaluate.py | tail -1
python shadow_rl/verl_math/math_reward.py | tail -1
python shadow_rl/verl_math/tests/test_prepare_data.py | tail -1
python shadow_rl/verl_math/tests/test_train_config.py | tail -1   # needs verl
```

`test_train_config.py` is the one worth running after any verl upgrade: it
composes the real training config and checks each hyperparameter by the path
verl reads it from. Hydra accepts a well-formed override that nothing consults,
so "training started" is not evidence that a setting took effect.

---

## Troubleshooting

**`Could not import module 'Qwen2ForCausalLM'`** — a CUDA mismatch between torch
and `torchvision`/`torchaudio`/`torchcodec`, not a transformers problem. Read the
error to see *which* package it names, then reinstall that one from the index
matching your torch:

```bash
python -c 'import torch; print(torch.version.cuda)'      # e.g. 13.0
pip install --force-reinstall torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Do **not** uninstall `torchvision` to dodge this: vllm imports it during kernel
warmup and will not start without it. `torchaudio` and `torchcodec` are needed by
nothing here, so removing those is fine.

**`no kernels for sm_100`** — a cu12 torch on a Blackwell GPU. Reinstall from the
cu130 index.

**Gated repos** — Llama originals and GPQA-Diamond need the licence accepted on
their model page, then `huggingface-cli login`.
