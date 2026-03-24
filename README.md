# Shadow-FT

Official repository for the paper **Shadow-FT: Tuning Instruct via Base**.

Shadow-FT fine-tunes a *Base* language model with LoRA to obtain lightweight delta parameters, then **merges** them onto its *Instruct* counterpart to boost instruction following. Training is powered by **LLaMA-Factory**, evaluation by **OpenCompass**.

---

## Install

```bash
conda create -n factory python=3.10 -y
conda activate factory

git clone https://github.com/yang3121099/Shadow-FT-bk && cd Shadow-FT-bk
pip install -e ".[torch,metrics]"
pip install importlib_metadata omegaconf
pip install torch==2.6.0 transformers==4.52.1 torchvision deepspeed -U

# OpenCompass (evaluation)
cd ./opencompass
pip install -e .
export COMPASS_DATA_CACHE="$(pwd)"
cd ..

# Optional dependencies
pip install lmdeploy evalplus==0.3.1 peft==0.15.2
pip install latex2sympy2_extended math_verify prettytable jieba rouge_chinese
pip install rank_bm25 gradio_client tree_sitter_languages fuzzywuzzy h5py
```

---

## Workflow

### Step 1: Generate training scripts

Edit `BASE_MODELS` in `run.sh` to select models (e.g., `"Llama3.2-1B"`), then:

```bash
bash ./run.sh
```

This generates:
- `./scripts/train_<MODEL>_<TIMESTAMP>.sh` — training + merge commands
- `./opencompass/eval_generated.py` — OpenCompass evaluation config

### Step 2: Run training

```bash
bash ./scripts/train_Llama-3.2-1B_<TIMESTAMP>.sh
```

The training script will:
1. Train LoRA on the **Base** model (B)
2. Train LoRA on the **Instruct** model (I) as baseline
3. Merge B-trained LoRA delta onto Instruct model (**B2I = Shadow-FT**)
4. Merge I-trained LoRA onto Instruct model (**I2I = ordinary SFT baseline**)

### Step 3: Run evaluation

```bash
cd ./opencompass
python3 ./run.py ./eval_generated.py -r <TIMESTAMP>
```

The evaluation config automatically includes all merged model paths from Step 2.

### Step 4: Upload results (optional)

```bash
pip install huggingface_hub
huggingface-cli login
python upload_hf.py
```

---

## Project Structure

```
.
├── run.sh                      # Pipeline generator (start here)
├── scripts/                    # Auto-generated training scripts
├── results/                    # Training outputs & merged models
├── data/                       # Training datasets
├── examples/
│   ├── model_pair.json         # Base/Instruct model pair definitions
│   └── deepspeed/              # DeepSpeed configs
├── src/
│   ├── shadow/
│   │   ├── merge_lora.py       # LoRA delta merge (B2I, I2I)
│   │   └── apply_diff.py       # Weight diff merge (for full SFT)
│   └── llamafactory/           # LLaMA-Factory training framework
├── opencompass/
│   ├── eval_generated.py       # Auto-generated eval config
│   ├── eval_shadow_*.py        # Manual eval configs
│   └── run.py                  # OpenCompass runner
└── upload_hf.py                # Upload models to HuggingFace
```

---

## Cite

```bibtex
@article{shadowft2025,
  title  = {Shadow-FT: Tuning Instruct via Base},
  author = {Wu et al.},
  year   = {2025},
  eprint = {2505.12716},
  archivePrefix = {arXiv}
}
```

## License

Apache-2.0. Please also comply with the licenses of any upstream models and datasets.
