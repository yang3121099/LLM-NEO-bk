#!/usr/bin/env bash
###############################################################################
# merge_and_eval_70b_medical.sh
#
# 1. Auto-discover trained Llama3-70B medical adapters in results/
# 2. Merge B2I and I2I for each config
# 3. Generate eval config: Medical-6 + Math-7
#
# Usage:
#   bash merge_and_eval_70b_medical.sh
###############################################################################
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"
TEMPLATE="llama3"
HF_INSTRUCT="meta-llama/Meta-Llama-3-70B-Instruct"

echo ""
echo "=========================================="
echo "  70B Medical: Merge + Eval"
echo "=========================================="

###############################################################################
# Step 1: Find trained adapters
###############################################################################
echo ""
echo "=== Step 1: Scanning for trained adapters ==="

# Search all possible date directories for Llama3-70B results
FOUND_ADAPTERS=()
FOUND_ROOT=""

for DATE_DIR in "$RESULTS_DIR"/*/; do
    [ -d "$DATE_DIR" ] || continue
    for MODEL_DIR in "$DATE_DIR"result-Llama3-70B-*/; do
        [ -d "$MODEL_DIR" ] || continue
        FOUND_ROOT="$MODEL_DIR"
        echo "  Found model root: $MODEL_DIR"
        # Find all adapter dirs (B-* and I-*)
        for ADAPTER_DIR in "$MODEL_DIR"/{B,I}-*-med2k/; do
            [ -d "$ADAPTER_DIR" ] || continue
            if [ -f "$ADAPTER_DIR/adapter_config.json" ] || \
               [ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || \
               [ -f "$ADAPTER_DIR/adapter_model.bin" ]; then
                FOUND_ADAPTERS+=("$ADAPTER_DIR")
                echo "    Adapter: $(basename "$ADAPTER_DIR")"
            fi
        done
    done
done

if [ ${#FOUND_ADAPTERS[@]} -eq 0 ]; then
    echo ""
    echo "WARNING: No trained adapters found."
    echo "Expected pattern: results/MMDD/result-Llama3-70B-MMDD/{B,I}-*-med2k/"
    echo "Will generate eval config with baseline Instruct model only."
fi

echo ""
echo "  Total adapters found: ${#FOUND_ADAPTERS[@]}"

###############################################################################
# Step 2: Merge B2I and I2I
###############################################################################
EVAL_ENTRIES=()

if [ ${#FOUND_ADAPTERS[@]} -gt 0 ]; then
echo ""
echo "=== Step 2: Merging adapters ==="

for ADAPTER_DIR in "${FOUND_ADAPTERS[@]}"; do
    DIRNAME=$(basename "$ADAPTER_DIR")
    # Extract B or I prefix
    PREFIX="${DIRNAME:0:1}"

    if [ "$PREFIX" = "B" ]; then
        MERGE_TAG="B2I"
    elif [ "$PREFIX" = "I" ]; then
        MERGE_TAG="I2I"
    else
        echo "  SKIP: Unknown prefix in $DIRNAME"
        continue
    fi

    MERGED_DIR="${ADAPTER_DIR}/merged-${MERGE_TAG}"

    # Check if already merged
    if [ -f "$MERGED_DIR/config.json" ]; then
        echo "  SKIP (already merged): $DIRNAME → $MERGE_TAG"
    else
        echo "  MERGE: $DIRNAME → $MERGE_TAG"
        python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
            --adapter_path "$ADAPTER_DIR" \
            --target_base "$HF_INSTRUCT" \
            --merge_tag "$MERGE_TAG" \
            --template "$TEMPLATE"
    fi

    # Build eval entry - derive config name from directory
    # Pattern: {B,I}-2k-lora-rank{R}-lr{LR}-med2k
    # Extract rank and lr for naming
    RANK=$(echo "$DIRNAME" | grep -oP 'rank\K\d+')
    LR=$(echo "$DIRNAME" | grep -oP 'lr\K[0-9.]+')

    # Derive config name
    if [ "$RANK" = "256" ] && [ "$LR" = "0.00005" ]; then
        CFG="med_cfgC"
    elif [ "$RANK" = "512" ] && [ "$LR" = "0.00005" ]; then
        CFG="med_r512"
    elif [ "$RANK" = "256" ] && [ "$LR" = "0.00002" ]; then
        CFG="med_lr2e5"
    else
        CFG="r${RANK}_lr${LR}"
    fi

    ABBR="Llama3-70B-med2k-2k-${CFG}-${MERGE_TAG}"
    REL_PATH="${MERGED_DIR#$WORKSPACE_DIR/}"
    EVAL_ENTRIES+=("    ('${ABBR}','\$RESULTS_DIR/../${REL_PATH}'),")
done

echo ""
echo "  Merge complete. ${#EVAL_ENTRIES[@]} models ready for eval."

else
echo ""
echo "=== Step 2: SKIPPED (no adapters) — baseline only ==="
fi

###############################################################################
# Step 3: Generate eval config
###############################################################################
echo ""
echo "=== Step 3: Generating eval config ==="

EVAL_CONFIG="$WORKSPACE_DIR/opencompass/eval_70b_medical.py"

cat > "$EVAL_CONFIG" << 'PYHEADER'
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
import os as _os

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results')

with read_base():
    ######################### Medical-6 #########################
    from opencompass.configs.datasets.MedQA.MedQA_gen_3bf756 import MedQA_datasets
    from opencompass.configs.datasets.medmcqa.medmcqa_gen_60c8f5 import medmcqa_datasets
    # medbench commented out — evaluator / data issues
    # from opencompass.configs.datasets.MedBench.medbench_gen_0b4fff import medbench_datasets
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_gen_a67965 import proteinlmbench_datasets
    from opencompass.configs.datasets.Medbullets.medbullets_gen_60c8f5 import medbullets_datasets

    ######################### Math-7 (alignment preservation) #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ---------------------------------------------------------------------------
# Inline summarizer (Medical-6 + Math-7)
# ---------------------------------------------------------------------------

medical_groups = [
    dict(
        name="Medical",
        subsets=[
            ["MedQA_US", "accuracy"],
            ["MedQA_Mainland", "accuracy"],
            ["MedQA_Taiwan", "accuracy"],
            ["medmcqa", "accuracy"],
            ["ProteinLMBench", "accuracy"],
            ["medbullets", "accuracy"],
        ],
    )
]

math_groups = [
    dict(
        name="Math",
        subsets=[
            ["math", "accuracy"],
            ["math-500", "accuracy"],
            ["minerva_math", "accuracy"],
            ["gsm8k", "accuracy"],
            ["gsm8k_0shot", "accuracy"],
            ["aime2024", "accuracy"],
            ["svamp", "accuracy"],
        ],
    )
]

average_groups = [
    {"name": "average_medical", "subsets": [["Medical", "naive_average"]]},
    {"name": "average_math", "subsets": [["Math", "naive_average"]]},
    {
        "name": "overall_average",
        "subsets": [
            ["average_medical", "naive_average"],
            ["average_math", "naive_average"],
        ],
    },
]

dataset_abbrs = [
    "--------- Medical ---------",
    ["MedQA_US", "accuracy"],
    ["MedQA_Mainland", "accuracy"],
    ["MedQA_Taiwan", "accuracy"],
    ["medmcqa", "accuracy"],
    ["ProteinLMBench", "accuracy"],
    ["medbullets", "accuracy"],
    "",
    "--------- Math ---------",
    ["math", "accuracy"],
    ["math-500", "accuracy"],
    ["minerva_math", "accuracy"],
    ["gsm8k", "accuracy"],
    ["gsm8k_0shot", "accuracy"],
    ["aime2024", "accuracy"],
    ["svamp", "accuracy"],
    "",
    "--------- Section AVG ---------",
    ["Medical", "naive_average"],
    ["Math", "naive_average"],
    "",
    "--------- Overall AVG ---------",
    ["overall_average", "naive_average"],
]

summary_groups = medical_groups + math_groups + average_groups

summarizer = dict(
    dataset_abbrs=dataset_abbrs,
    summary_groups=summary_groups,
)

PYHEADER

# Write model entries
{
    echo "Baseline_settings = ["
    echo "('Meta-Llama-3-70B-Instruct-hf', 'meta-llama/Meta-Llama-3-70B-Instruct'),"
    echo ""
    for entry in "${EVAL_ENTRIES[@]}"; do
        echo "$entry"
    done
    echo "]"
} >> "$EVAL_CONFIG"

cat >> "$EVAL_CONFIG" << 'PYFOOTER'

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

echo "  Eval config: $EVAL_CONFIG"

###############################################################################
# Summary
###############################################################################
echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
echo ""
echo "=== Models in eval ==="
echo "  Baseline: Meta-Llama-3-70B-Instruct"
for entry in "${EVAL_ENTRIES[@]}"; do
    ABBR=$(echo "$entry" | grep -oP "'\K[^']+(?=',')")
    echo "  $ABBR"
done
echo ""
echo "=== Eval benchmarks (13 total) ==="
echo "  Medical-6: MedQA (US/Mainland/Taiwan), medmcqa, ProteinLMBench, medbullets"
echo "  Math-7:    math, math-500, minerva_math, gsm8k, gsm8k_0shot, aime2024, svamp"
echo ""
echo "=== Run eval ==="
echo "  # Step 1: sync opencompass files"
echo "  bash src/copy_files.sh \$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')"
echo ""
echo "  # Step 2: run eval (-r reuses existing inference results)"
echo "  cd opencompass && python3 ./run.py eval_70b_medical.py -r eval70b_med"
echo ""
