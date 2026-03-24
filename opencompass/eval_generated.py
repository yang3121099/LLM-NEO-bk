# Auto-generated evaluation config for Shadow-FT
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_generated.py -r <TIMESTAMP>

from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

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

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel

work_dir = 'outputs/shadow-ft-0324154403/'

# ======= Instruct-type models (B2I Shadow-FT, I2I baseline) =======
Baseline_settings = [
    ('result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
    ('result-Qwen3.5-2B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-2B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-2B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-2B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
    ('result-Qwen3.5-4B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-4B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3.5-4B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-4B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
]

# ======= Base-type models (B2B, I2B — usually commented out) =======
BASE_settings = [
    ('result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
    ('result-Qwen3.5-2B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-2B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-2B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-2B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
    ('result-Qwen3.5-4B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-4B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
    ('result-Qwen3.5-4B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-4B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
]

models = []

for abbr, path in Baseline_settings:
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

for abbr, path in BASE_settings:
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1),
        )
    )
