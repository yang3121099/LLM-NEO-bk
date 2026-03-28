# Cross-Delta Transfer Experiment — Evaluation Config (0327)
#
# 4 sources × 8 targets + 8 direct-FT baselines + 8 original targets
# Math-7 benchmarks (MATH full commented out)
#
# Usage: cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py

import os as _os

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
ADAPTER_ROOT = _os.path.join(RESULTS_DIR, '0327', 'result-weight-similarity-0327')
del _SCRIPT_DIR

from mmengine.config import read_base

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 (MATH full commented out) #########################
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

work_dir = 'outputs/cross-delta-0327/'

###############################################################################
# Model Path Helpers
###############################################################################

# Adapter root → merged model path
def _merged(adapter_name, merge_tag):
    """Return path to merged model: {ADAPTER_ROOT}/{adapter_name}-2k-.../merged-{merge_tag}"""
    suffix = '2k-lora-rank128-lr0.0002-shadow2k'
    return _os.path.join(ADAPTER_ROOT, f'{adapter_name}-{suffix}', f'merged-{merge_tag}')

# Adapter name mapping (for constructing paths)
# Source short names → adapter directory names
_SRC_MAP = {
    'Base':    'Base3.1',
    'SFT':     'Tulu3-SFT',
    'DPO':     'Tulu3-DPO',
    'RLVR':    'Tulu3-RLVR',
}

# Target short names → adapter directory names
_TGT_MAP = {
    'Base':       'Base3.1',
    'SFT':        'Tulu3-SFT',
    'DPO':        'Tulu3-DPO',
    'RLVR':       'Tulu3-RLVR',
    'Instruct':   'Instruct3.1',
    'Tulu3.1':    'Tulu3.1',
    'Llama3-Inst':'Instruct3',
    'R1-Distill': 'R1-Distill',
}

###############################################################################
# 1. Original target models (no training, reference baselines)
###############################################################################
original_targets = [
    ('Base-orig',       'meta-llama/Llama-3.1-8B'),
    ('SFT-orig',        'allenai/Llama-3.1-Tulu-3-8B-SFT'),
    ('DPO-orig',        'allenai/Llama-3.1-Tulu-3-8B-DPO'),
    ('RLVR-orig',       'allenai/Llama-3.1-Tulu-3-8B'),
    ('Instruct-orig',   'meta-llama/Llama-3.1-8B-Instruct'),
    ('Tulu3.1-orig',    'allenai/Llama-3.1-Tulu-3.1-8B'),
    ('Llama3-Inst-orig','meta-llama/Meta-Llama-3-8B-Instruct'),
    ('R1-Distill-orig', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
]

###############################################################################
# 2. Direct-FT baselines (target trained on itself, merged back)
###############################################################################
direct_ft_baselines = [
    (f'DirectFT-{tgt_short}', _merged(tgt_adapter, f'DirectFT-{tgt_short}'))
    for tgt_short, tgt_adapter in _TGT_MAP.items()
]

###############################################################################
# 3. Cross-delta merged models (4 sources × 8 targets = 32)
###############################################################################
cross_delta_models = [
    (f'{src_short}2{tgt_short}', _merged(src_adapter, f'{src_short}2{tgt_short}'))
    for src_short, src_adapter in _SRC_MAP.items()
    for tgt_short in _TGT_MAP.keys()
]

###############################################################################
# Build model list
###############################################################################
def _make_model(abbr, path):
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

models = []

for abbr, path in original_targets:
    models.append(_make_model(abbr, path))

for abbr, path in direct_ft_baselines:
    models.append(_make_model(abbr, path))

for abbr, path in cross_delta_models:
    models.append(_make_model(abbr, path))

# Print summary
print(f'\n[eval_cross_delta_0327] Models loaded:')
print(f'  Original targets: {len(original_targets)}')
print(f'  Direct-FT baselines: {len(direct_ft_baselines)}')
print(f'  Cross-delta merges: {len(cross_delta_models)}')
print(f'  Total: {len(models)}')
