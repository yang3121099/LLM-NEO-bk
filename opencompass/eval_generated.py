# Auto-generated evaluation config for Shadow-FT
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_generated.py -r <TIMESTAMP>

import os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR relative to this config file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'results')

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import HuggingFacewithChatTemplate, HuggingFaceBaseModel

work_dir = 'outputs/shadow-ft-0325075759/'

# ======= Instruct-type models (B2I Shadow-FT, I2I baseline) =======
Baseline_settings = [
    ('result-Qwen3.5-0.8B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','$RESULTS_DIR/0325/result-Qwen3.5-0.8B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-0.8B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','$RESULTS_DIR/0325/result-Qwen3.5-0.8B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
    ('result-Qwen3.5-2B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','$RESULTS_DIR/0325/result-Qwen3.5-2B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-2B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','$RESULTS_DIR/0325/result-Qwen3.5-2B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
    ('result-Qwen3.5-4B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','$RESULTS_DIR/0325/result-Qwen3.5-4B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-4B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','$RESULTS_DIR/0325/result-Qwen3.5-4B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
]

# ======= Base-type models (B2B, I2B — usually commented out) =======
BASE_settings = [
    ('result-Qwen3.5-0.8B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','$RESULTS_DIR/0325/result-Qwen3.5-0.8B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-0.8B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','$RESULTS_DIR/0325/result-Qwen3.5-0.8B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
    ('result-Qwen3.5-2B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','$RESULTS_DIR/0325/result-Qwen3.5-2B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-2B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','$RESULTS_DIR/0325/result-Qwen3.5-2B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
    ('result-Qwen3.5-4B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','$RESULTS_DIR/0325/result-Qwen3.5-4B-Base-0325/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-4B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','$RESULTS_DIR/0325/result-Qwen3.5-4B-Base-0325/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
]

models = []

for abbr, path in Baseline_settings:
    # Resolve $RESULTS_DIR references
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=HuggingFacewithChatTemplate,
            abbr=abbr,
            path=path,
            model_kwargs=dict(device_map='auto', torch_dtype='auto', trust_remote_code=True),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=8,
            run_cfg=dict(num_gpus=1),
        )
    )

for abbr, path in BASE_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=HuggingFaceBaseModel,
            abbr=abbr,
            path=path,
            model_kwargs=dict(device_map='auto', torch_dtype='auto', trust_remote_code=True),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=8,
            run_cfg=dict(num_gpus=1),
        )
    )
