# Auto-generated evaluation config for Weight Similarity Experiment
# Usage:
#   cd opencompass
#   python3 ./run.py ./eval_weight_similarity.py -r <TIMESTAMP>

import os as _os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
del _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 Benchmarks #########################
    # Full MATH benchmark (commented out per request)
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets

    # MATH-500 (subset)
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

    # Minerva MATH
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets

    # GSM8K (8-shot)
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

    # GSM8K (0-shot)
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets

    # AIME 2024
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

    # SVAMP
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

work_dir = 'outputs/weight-similarity-exp/'

# ======= Original models (baselines without LoRA training) =======
original_baselines = [
    ('Instruct3.1-orig', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('Tulu3-SFT-orig', 'allenai/Llama-3.1-Tulu-3-8B-SFT'),
    ('Tulu3-DPO-orig', 'allenai/Llama-3.1-Tulu-3-8B-DPO'),
    ('Tulu3-RLVR-orig', 'allenai/Llama-3.1-Tulu-3-8B'),
    ('Tulu3.1-orig', 'allenai/Llama-3.1-Tulu-3.1-8B'),
    ('Instruct3-orig', 'meta-llama/Llama-3-8B-Instruct'),
    ('R1-Distill-orig', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
]

# ======= Merged models (LoRA delta transfers) =======
merged_models = [
    ('Base3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3.1'),
    ('Tulu3-SFT2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Instruct3.1'),
    ('Tulu3-DPO2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Instruct3.1'),
    ('Tulu3-RLVR2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Instruct3.1'),
    ('Instruct3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Instruct3.1'),
    ('Tulu3-SFT2Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-SFT'),
    ('Tulu3-DPO2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-DPO'),
    ('Tulu3-RLVR2Tulu3-RLVR','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-RLVR'),
    ('Base3.12Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-SFT'),
    ('Base3.12Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-DPO'),
    ('Base3.12Tulu3-RLVR','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-RLVR'),
    ('Base3.12Tulu3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3.1'),
    ('Base32Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3'),
    ('Base32R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32R1-Distill'),
    ('Base3.12R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12R1-Distill'),
    ('Instruct32Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Instruct3'),
    ('R1-Distill2R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2R1-Distill'),
    ('Tulu3.12Tulu3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Tulu3.1'),
    # ('Tulu3-SFT2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-DPO'),
    # ('Tulu3-DPO2Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-SFT'),
    # ('Tulu3-RLVR2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-DPO'),
    # ('Instruct3.12Base3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Base3.1'),
    # ('Instruct32Base3','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Base3'),
    # ('R1-Distill2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Instruct3.1'),
    # ('R1-Distill2Base3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Base3.1'),
    # ('Tulu3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Instruct3.1'),
    # ('Base3.12Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3'),
    # ('Base32Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3.1'),
]

models = []

# Original baselines
for abbr, path in original_baselines:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
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

# Merged models
for abbr, path in merged_models:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
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
