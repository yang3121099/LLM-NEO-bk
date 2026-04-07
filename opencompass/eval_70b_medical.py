from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
import os as _os

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results')

with read_base():
    ######################### Medical (no medbench) #########################
    from opencompass.configs.datasets.MedQA.MedQA_gen_3bf756 import MedQA_datasets
    from opencompass.configs.datasets.medmcqa.medmcqa_gen_60c8f5 import medmcqa_datasets
    # medbench commented out — evaluator / data issues
    # from opencompass.configs.datasets.MedBench.medbench_gen_0b4fff import medbench_datasets

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
# Inline summarizer (Medical + Math-7)
# ---------------------------------------------------------------------------

medical_groups = [
    dict(
        name="Medical",
        subsets=[
            ["MedQA_US", "accuracy"],
            ["MedQA_Mainland", "accuracy"],
            ["MedQA_Taiwan", "accuracy"],
            ["medmcqa", "accuracy"],
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
    # Medical
    "--------- Medical ---------",
    ["MedQA_US", "accuracy"],
    ["MedQA_Mainland", "accuracy"],
    ["MedQA_Taiwan", "accuracy"],
    ["medmcqa", "accuracy"],

    "",

    # Math
    "--------- Math ---------",
    ["math", "accuracy"],
    ["math-500", "accuracy"],
    ["minerva_math", "accuracy"],
    ["gsm8k", "accuracy"],
    ["gsm8k_0shot", "accuracy"],
    ["aime2024", "accuracy"],
    ["svamp", "accuracy"],

    "",

    # Section AVG
    "--------- Section AVG ---------",
    ["Medical", "naive_average"],
    ["Math", "naive_average"],

    "",

    # Overall AVG
    "--------- Overall AVG ---------",
    ["overall_average", "naive_average"],
]

summary_groups = medical_groups + math_groups + average_groups

summarizer = dict(
    dataset_abbrs=dataset_abbrs,
    summary_groups=summary_groups,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

Baseline_settings = [
('Meta-Llama-3-70B-Instruct-hf', 'meta-llama/Meta-Llama-3-70B-Instruct'),

    ('Llama3-70B-med2k-2k-med_cfgC-B2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/B-2k-lora-rank256-lr0.00005-med2k/merged-B2I'),
    ('Llama3-70B-med2k-2k-med_cfgC-I2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/I-2k-lora-rank256-lr0.00005-med2k/merged-I2I'),
    ('Llama3-70B-med2k-2k-med_r512-B2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/B-2k-lora-rank512-lr0.00005-med2k/merged-B2I'),
    ('Llama3-70B-med2k-2k-med_r512-I2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/I-2k-lora-rank512-lr0.00005-med2k/merged-I2I'),
    ('Llama3-70B-med2k-2k-med_lr2e5-B2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/B-2k-lora-rank256-lr0.00002-med2k/merged-B2I'),
    ('Llama3-70B-med2k-2k-med_lr2e5-I2I','$RESULTS_DIR/0407/result-Llama3-70B-0407/I-2k-lora-rank256-lr0.00002-med2k/merged-I2I'),
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
