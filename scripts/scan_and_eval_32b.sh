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
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
import os as _os

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results')

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Reasoning-9 (general reasoning) #########################
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets  # few-shot
    from opencompass.configs.datasets.bbh.bbh_0shot_nocot_gen_925fc4 import bbh_datasets as bbh3_datasets  # 0-shot
    from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
    from opencompass.configs.datasets.TheoremQA.ThroremQA_0shot_cot_gen_8acdf7 import TheoremQA_datasets

    ######################### Math-7 (mathematical) #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets   # noqa: F401, F403
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets  # minerva_math
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets  # MATH
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets  # 0-shot eval_v2
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

Baseline_settings = [
('Qwen2.5-32B-Instruct-hf', 'Qwen/Qwen2.5-32B-Instruct'),
('Qwen3-30B-A3B-hf', 'Qwen/Qwen3-30B-A3B'),

PYHEADER

# Add valid merged models
for d in "${VALID_DIRS[@]}"; do
  parent="$(basename "$(dirname "$d")")"
  merge_tag="$(basename "$d")"
  model_dir="$(basename "$(dirname "$(dirname "$d")")")"
  model_short="${model_dir#result-}"
  mtag="${merge_tag#merged-}"
  abbr="${model_short}-${parent}-${mtag}"
  echo "    ('${abbr}', '${d}')," >> "$EVAL_CONFIG"
done

cat >> "$EVAL_CONFIG" << 'PYFOOTER'
]

models = []

for abbr, path in Baseline_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=8),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=8),
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
