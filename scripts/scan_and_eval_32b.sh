#!/usr/bin/env bash
###############################################################################
# scan_and_eval_32b.sh — Scan existing merged models and generate eval config
#
# Scans results/0402/ for successfully merged models (has config.json),
# generates an eval Python config, and runs it with -r eval32b for resume.
#
# Usage:
#   bash scripts/scan_and_eval_32b.sh              # dry-run: show what would eval
#   bash scripts/scan_and_eval_32b.sh --run        # actually run eval
#   bash scripts/scan_and_eval_32b.sh --run --bg   # run in background via nohup
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCAN_DIR="${RESULTS_DIR}/0402"
EVAL_DIR="$WORKSPACE_DIR/opencompass"
RUN_TAG="eval32b"

DO_RUN=false
DO_BG=false
for arg in "$@"; do
  case "$arg" in
    --run) DO_RUN=true ;;
    --bg)  DO_BG=true ;;
  esac
done

echo "=========================================="
echo "  Scanning: $SCAN_DIR"
echo "=========================================="

# Find all merged-* dirs that have a config.json (= successful merge)
MERGED_DIRS=()
while IFS= read -r d; do
  MERGED_DIRS+=("$d")
done < <(find "$SCAN_DIR" -type d -name "merged-*" 2>/dev/null | sort)

if [[ ${#MERGED_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No merged-* directories found under $SCAN_DIR"
  exit 1
fi

echo ""
echo "Found ${#MERGED_DIRS[@]} merged directories:"
VALID_DIRS=()
for d in "${MERGED_DIRS[@]}"; do
  if [[ -f "$d/config.json" ]]; then
    rel="${d#$RESULTS_DIR/}"
    echo "  [OK]   $rel"
    VALID_DIRS+=("$d")
  else
    rel="${d#$RESULTS_DIR/}"
    echo "  [SKIP] $rel  (no config.json — merge incomplete)"
  fi
done

if [[ ${#VALID_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No valid merged models found (none have config.json)"
  exit 1
fi

echo ""
echo "Valid models for eval: ${#VALID_DIRS[@]}"

# Generate eval config
TIMESTAMP=$(date +%m%d%H%M%S)
EVAL_CONFIG="$EVAL_DIR/eval_32b_scan_${TIMESTAMP}.py"

cat > "$EVAL_CONFIG" << 'PYHEADER'
# Auto-generated eval config for existing merged models (16 benchmarks, no code)
import subprocess as _sp
import os as _os
import importlib as _il
from opencompass.models import TurboMindModelwithChatTemplate

_NUM_GPUS = max(1, len(_sp.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()))

def _load(mod, var):
    return getattr(_il.import_module(mod), var)

######################### Math-7 #########################
math_500_datasets     = _load('opencompass.configs.datasets.math.math_500_gen', 'math_datasets')
minerva_math_datasets = _load('opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31', 'minerva_math_datasets')
math_datasets         = _load('opencompass.configs.datasets.math.math_0shot_gen_393424', 'math_datasets')
gsm8k_datasets        = _load('opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4', 'gsm8k_datasets')
gsm8k_0shot_datasets  = _load('opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799', 'gsm8k_datasets')
aime2024_datasets     = _load('opencompass.configs.datasets.aime2024.aime2024_gen_17d799', 'aime2024_datasets')
svamp_datasets        = _load('opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4', 'svamp_datasets')

######################### Reasoning-9 #########################
mmlu_datasets         = _load('opencompass.configs.datasets.mmlu.mmlu_gen_4d595a', 'mmlu_datasets')
mmlu_pro_datasets     = _load('opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de', 'mmlu_pro_datasets')
bbh_datasets          = _load('opencompass.configs.datasets.bbh.bbh_gen_5b92b0', 'bbh_datasets')
bbh3_datasets         = _load('opencompass.configs.datasets.bbh.bbh_0shot_nocot_gen_925fc4', 'bbh3_datasets')
drop_datasets         = _load('opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0', 'drop_datasets')
winogrande_datasets   = _load('opencompass.configs.datasets.winogrande.winogrande_gen_a027b6', 'winogrande_datasets')
ARC_c_datasets        = _load('opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652', 'ARC_c_datasets')
gpqa_datasets         = _load('opencompass.configs.datasets.gpqa.gpqa_gen_4baadb', 'gpqa_datasets')
TheoremQA_datasets    = _load('opencompass.configs.datasets.TheoremQA.ThroremQA_0shot_cot_gen_8acdf7', 'TheoremQA_datasets')

datasets = (
    math_500_datasets + minerva_math_datasets + math_datasets +
    gsm8k_datasets + gsm8k_0shot_datasets + aime2024_datasets + svamp_datasets +
    mmlu_datasets + mmlu_pro_datasets + bbh_datasets + bbh3_datasets +
    drop_datasets + winogrande_datasets + ARC_c_datasets + gpqa_datasets + TheoremQA_datasets
)

# ======= Instruct baselines =======
HF_baselines = [
    ('Qwen2.5-32B-Instruct-hf', 'Qwen/Qwen2.5-32B-Instruct'),
    ('Qwen3-30B-A3B-hf', 'Qwen/Qwen3-30B-A3B'),
]

# ======= Trained models =======
Baseline_settings = [
PYHEADER

# Add valid merged models
for d in "${VALID_DIRS[@]}"; do
  # Extract a readable abbreviation from path
  # e.g. results/0402/result-Qwen2.5-32B-0402/B-2k-lora-rank256-lr0.0001-shadow2k/merged-B2I
  parent="$(basename "$(dirname "$d")")"   # B-2k-lora-rank256-lr0.0001-shadow2k
  merge_tag="$(basename "$d")"             # merged-B2I
  model_dir="$(basename "$(dirname "$(dirname "$d")")")"  # result-Qwen2.5-32B-0402
  model_short="${model_dir#result-}"       # Qwen2.5-32B-0402
  mtag="${merge_tag#merged-}"              # B2I
  abbr="${model_short}-${parent}-${mtag}"
  echo "    ('${abbr}', '${d}')," >> "$EVAL_CONFIG"
done

cat >> "$EVAL_CONFIG" << 'PYFOOTER'
]

models = []

for abbr, path in HF_baselines:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=2048, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=1024,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )

for abbr, path in Baseline_settings:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=2048, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=1024,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )
PYFOOTER

echo ""
echo "Eval config: $EVAL_CONFIG"
echo "Run tag: -r $RUN_TAG (resume-safe, shared across all eval runs)"
echo ""

if $DO_RUN; then
  CMD="cd $EVAL_DIR && python3 ./run.py $(basename "$EVAL_CONFIG") -r $RUN_TAG"
  if $DO_BG; then
    echo "Running in background..."
    nohup bash -c "$CMD" > "$WORKSPACE_DIR/eval_32b_scan.log" 2>&1 &
    echo "PID: $!"
    echo "Log: $WORKSPACE_DIR/eval_32b_scan.log"
  else
    echo "Running eval..."
    eval "$CMD"
  fi
else
  echo "Dry run. To execute:"
  echo "  cd $EVAL_DIR && python3 ./run.py $(basename "$EVAL_CONFIG") -r $RUN_TAG"
  echo ""
  echo "Or re-run with --run:"
  echo "  bash scripts/scan_and_eval_32b.sh --run"
fi
