# Auto-generated: Cross-Delta Transfer Experiment evaluation
# Usage: cd opencompass && python3 ./run.py ./eval_cross_delta.py
import os as _os
from mmengine.config import read_base

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
del _SCRIPT_DIR

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 (MATH commented out) #########################
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate


work_dir = 'outputs/cross-delta-exp/'

# === Original target models (no training, reference only) ===
original_targets = [
    ('Base-orig', 'meta-llama/Llama-3.1-8B'),
    ('SFT-orig', 'allenai/Llama-3.1-Tulu-3-8B-SFT'),
    ('DPO-orig', 'allenai/Llama-3.1-Tulu-3-8B-DPO'),
    ('RLVR-orig', 'allenai/Llama-3.1-Tulu-3-8B'),
    ('Instruct-orig', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('Tulu3.1-orig', 'allenai/Llama-3.1-Tulu-3.1-8B'),
    ('Llama3-Inst-orig', 'meta-llama/Meta-Llama-3-8B-Instruct'),
    ('R1-Distill-orig', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
]

# === Direct-FT baselines (each target trained on itself) ===
direct_ft_baselines = [
    ('DirectFT-Base', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-DirectFT-Base'),
    ('DirectFT-SFT', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-DirectFT-SFT'),
    ('DirectFT-DPO', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DirectFT-DPO'),
    ('DirectFT-RLVR', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-DirectFT-RLVR'),
    ('DirectFT-Instruct', '$RESULTS_DIR/0327/cross-delta-0327/tgt-Instruct-lora-r128-shadow2k/merged-DirectFT-Instruct'),
    ('DirectFT-Tulu3.1', '$RESULTS_DIR/0327/cross-delta-0327/tgt-Tulu3.1-lora-r128-shadow2k/merged-DirectFT-Tulu3.1'),
    ('DirectFT-Llama3-Inst', '$RESULTS_DIR/0327/cross-delta-0327/tgt-Llama3-Inst-lora-r128-shadow2k/merged-DirectFT-Llama3-Inst'),
    ('DirectFT-R1-Distill', '$RESULTS_DIR/0327/cross-delta-0327/tgt-R1-Distill-lora-r128-shadow2k/merged-DirectFT-R1-Distill'),
]

# === Cross-delta merged models (source adapter → target model) ===
cross_delta_models = [
    ('Base2Base', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Base'),
    ('Base2SFT', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2SFT'),
    ('Base2DPO', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2DPO'),
    ('Base2RLVR', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2RLVR'),
    ('Base2Instruct', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Instruct'),
    ('Base2Tulu3.1', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Tulu3.1'),
    ('Base2Llama3-Inst', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Llama3-Inst'),
    ('Base2R1-Distill', '$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2R1-Distill'),
    ('SFT2Base', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Base'),
    ('SFT2SFT', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2SFT'),
    ('SFT2DPO', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2DPO'),
    ('SFT2RLVR', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2RLVR'),
    ('SFT2Instruct', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Instruct'),
    ('SFT2Tulu3.1', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Tulu3.1'),
    ('SFT2Llama3-Inst', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Llama3-Inst'),
    ('SFT2R1-Distill', '$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2R1-Distill'),
    ('DPO2Base', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Base'),
    ('DPO2SFT', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2SFT'),
    ('DPO2DPO', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2DPO'),
    ('DPO2RLVR', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2RLVR'),
    ('DPO2Instruct', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Instruct'),
    ('DPO2Tulu3.1', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Tulu3.1'),
    ('DPO2Llama3-Inst', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Llama3-Inst'),
    ('DPO2R1-Distill', '$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2R1-Distill'),
    ('RLVR2Base', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Base'),
    ('RLVR2SFT', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2SFT'),
    ('RLVR2DPO', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2DPO'),
    ('RLVR2RLVR', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2RLVR'),
    ('RLVR2Instruct', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Instruct'),
    ('RLVR2Tulu3.1', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Tulu3.1'),
    ('RLVR2Llama3-Inst', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Llama3-Inst'),
    ('RLVR2R1-Distill', '$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2R1-Distill'),
]

models = []

def _make_model(abbr, path):
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    return dict(
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

for abbr, path in original_targets:
    models.append(_make_model(abbr, path))

for abbr, path in direct_ft_baselines:
    models.append(_make_model(abbr, path))

for abbr, path in cross_delta_models:
    models.append(_make_model(abbr, path))
