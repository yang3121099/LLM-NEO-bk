# Cross-Delta Transfer Experiment — Evaluation Config (0327)
# Usage: cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py

import os as _os
from mmengine.config import read_base

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
_ROOT = _os.path.join(_RESULTS_DIR, '0327', 'result-weight-similarity-0327')
_S = '2k-lora-rank128-lr0.0002-shadow2k'

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
# 1. Original target models (no training)
###############################################################################
original_targets = [
    ('Base-orig',       'meta-llama/Llama-3.1-8B'),
    ('SFT-orig',        'allenai/Llama-3.1-Tulu-3-8B-SFT'),
    ('DPO-orig',        'allenai/Llama-3.1-Tulu-3-8B-DPO'),
    ('RLVR-orig',       'allenai/Llama-3.1-Tulu-3-8B'),
    ('Instruct-orig',   'meta-llama/Llama-3.1-8B-Instruct'),
    ('Tulu3.1-orig',    'allenai/Llama-3.1-Tulu-3.1-8B'),
    # ('Llama3-Inst-orig','meta-llama/Meta-Llama-3-8B-Instruct'),  # uncomment if available
    # ('R1-Distill-orig', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
]

###############################################################################
# 2. Direct-FT baselines (target trained on itself → merged back)
###############################################################################
direct_ft_baselines = [
    ('DirectFT-Base',       _ROOT + '/Base3.1-' + _S + '/merged-DirectFT-Base'),
    ('DirectFT-SFT',        _ROOT + '/Tulu3-SFT-' + _S + '/merged-DirectFT-SFT'),
    ('DirectFT-DPO',        _ROOT + '/Tulu3-DPO-' + _S + '/merged-DirectFT-DPO'),
    ('DirectFT-RLVR',       _ROOT + '/Tulu3-RLVR-' + _S + '/merged-DirectFT-RLVR'),
    ('DirectFT-Instruct',   _ROOT + '/Instruct3.1-' + _S + '/merged-DirectFT-Instruct'),
    ('DirectFT-Tulu3.1',    _ROOT + '/Tulu3.1-' + _S + '/merged-DirectFT-Tulu3.1'),
    # ('DirectFT-Llama3-Inst', _ROOT + '/Instruct3-' + _S + '/merged-DirectFT-Llama3-Inst'),
    # ('DirectFT-R1-Distill',  _ROOT + '/R1-Distill-' + _S + '/merged-DirectFT-R1-Distill'),
]

###############################################################################
# 3. Cross-delta merged models (4 sources × 8 targets)
#    Source adapter dir → merged-{Src}2{Tgt}
###############################################################################
# Source: Base3.1 (adapter trained on Base)
_BASE_A = _ROOT + '/Base3.1-' + _S
# Source: Tulu3-SFT (adapter trained on SFT)
_SFT_A = _ROOT + '/Tulu3-SFT-' + _S
# Source: Tulu3-DPO (adapter trained on DPO)
_DPO_A = _ROOT + '/Tulu3-DPO-' + _S
# Source: Tulu3-RLVR (adapter trained on RLVR)
_RLVR_A = _ROOT + '/Tulu3-RLVR-' + _S

cross_delta_models = [
    # === Base delta → 8 targets ===
    ('Base2Base',       _BASE_A + '/merged-Base2Base'),
    ('Base2SFT',        _BASE_A + '/merged-Base2SFT'),
    ('Base2DPO',        _BASE_A + '/merged-Base2DPO'),
    ('Base2RLVR',       _BASE_A + '/merged-Base2RLVR'),
    ('Base2Instruct',   _BASE_A + '/merged-Base2Instruct'),
    ('Base2Tulu3.1',    _BASE_A + '/merged-Base2Tulu3.1'),
    ('Base2Llama3-Inst', _BASE_A + '/merged-Base2Llama3-Inst'),
    ('Base2R1-Distill', _BASE_A + '/merged-Base2R1-Distill'),

    # === SFT delta → 8 targets ===
    ('SFT2Base',       _SFT_A + '/merged-SFT2Base'),
    ('SFT2SFT',        _SFT_A + '/merged-SFT2SFT'),
    ('SFT2DPO',        _SFT_A + '/merged-SFT2DPO'),
    ('SFT2RLVR',       _SFT_A + '/merged-SFT2RLVR'),
    ('SFT2Instruct',   _SFT_A + '/merged-SFT2Instruct'),
    ('SFT2Tulu3.1',    _SFT_A + '/merged-SFT2Tulu3.1'),
    ('SFT2Llama3-Inst', _SFT_A + '/merged-SFT2Llama3-Inst'),
    ('SFT2R1-Distill', _SFT_A + '/merged-SFT2R1-Distill'),

    # === DPO delta → 8 targets ===
    ('DPO2Base',       _DPO_A + '/merged-DPO2Base'),
    ('DPO2SFT',        _DPO_A + '/merged-DPO2SFT'),
    ('DPO2DPO',        _DPO_A + '/merged-DPO2DPO'),
    ('DPO2RLVR',       _DPO_A + '/merged-DPO2RLVR'),
    ('DPO2Instruct',   _DPO_A + '/merged-DPO2Instruct'),
    ('DPO2Tulu3.1',    _DPO_A + '/merged-DPO2Tulu3.1'),
    ('DPO2Llama3-Inst', _DPO_A + '/merged-DPO2Llama3-Inst'),
    ('DPO2R1-Distill', _DPO_A + '/merged-DPO2R1-Distill'),

    # === RLVR delta → 8 targets ===
    ('RLVR2Base',       _RLVR_A + '/merged-RLVR2Base'),
    ('RLVR2SFT',        _RLVR_A + '/merged-RLVR2SFT'),
    ('RLVR2DPO',        _RLVR_A + '/merged-RLVR2DPO'),
    ('RLVR2RLVR',       _RLVR_A + '/merged-RLVR2RLVR'),
    ('RLVR2Instruct',   _RLVR_A + '/merged-RLVR2Instruct'),
    ('RLVR2Tulu3.1',    _RLVR_A + '/merged-RLVR2Tulu3.1'),
    ('RLVR2Llama3-Inst', _RLVR_A + '/merged-RLVR2Llama3-Inst'),
    ('RLVR2R1-Distill', _RLVR_A + '/merged-RLVR2R1-Distill'),
]

###############################################################################
# Build model list — filter out paths that don't exist
###############################################################################
models = []

_all_entries = original_targets + direct_ft_baselines + cross_delta_models
_skipped = []

for abbr, path in _all_entries:
    # For HF hub paths (no /), always include
    # For local paths, check existence
    if '/' in path and not path.startswith(('meta-llama', 'allenai', 'deepseek', 'Qwen')):
        if not _os.path.isdir(path):
            _skipped.append(abbr)
            continue

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

if _skipped:
    print('[eval_cross_delta_0327] Skipped (path not found): ' + ', '.join(_skipped))
print('[eval_cross_delta_0327] Total models to evaluate: ' + str(len(models)))
