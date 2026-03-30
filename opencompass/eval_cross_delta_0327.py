# Cross-Delta Transfer Experiment — Evaluation Config (0327)
# Usage: cd opencompass && python3 ./run.py ./eval_cross_delta_0327.py

import os as _os

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')
_ROOT = _os.path.join(_RESULTS_DIR, '0327', 'result-weight-similarity-0327')
_S = '2k-lora-rank128-lr0.0002-shadow2k'

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

    ######################### General benchmarks #########################
    from opencompass.configs.datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModel, TurboMindModelwithChatTemplate

work_dir = 'outputs/cross-delta-0327/'

###############################################################################
# Adapter path shortcuts
###############################################################################
_BASE_A   = _ROOT + '/Base3.1-' + _S
_SFT_A    = _ROOT + '/Tulu3-SFT-' + _S
_DPO_A    = _ROOT + '/Tulu3-DPO-' + _S
_RLVR_A   = _ROOT + '/Tulu3-RLVR-' + _S
_INST_A   = _ROOT + '/Instruct3.1-' + _S
_T31_A    = _ROOT + '/Tulu3.1-' + _S
_INST3_A  = _ROOT + '/Instruct3-' + _S
_R1D_A    = _ROOT + '/R1-Distill-' + _S

###############################################################################
# Model entries: (abbr, path, is_base)
#   is_base=True  → TurboMindModel (no chat template)
#   is_base=False → TurboMindModelwithChatTemplate
###############################################################################

# --- 1. Original target models (no training) ---
original_targets = [
    # (abbr, path, is_base)
    ('Base-orig',        'meta-llama/Llama-3.1-8B',                      True),
    ('SFT-orig',         'allenai/Llama-3.1-Tulu-3-8B-SFT',             False),
    ('DPO-orig',         'allenai/Llama-3.1-Tulu-3-8B-DPO',             False),
    ('RLVR-orig',        'allenai/Llama-3.1-Tulu-3-8B',                 False),
    ('Instruct-orig',    'meta-llama/Llama-3.1-8B-Instruct',            False),
    ('Tulu3.1-orig',     'allenai/Llama-3.1-Tulu-3.1-8B',              False),
    ('Llama3-Inst-orig', 'meta-llama/Meta-Llama-3-8B-Instruct',        False),
    ('R1-Distill-orig',  'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',   False),
]

# --- 2. Direct-FT baselines (self-merge) ---
direct_ft_baselines = [
    ('DirectFT-Base',       _BASE_A  + '/merged-DirectFT-Base',       True),
    ('DirectFT-SFT',        _SFT_A   + '/merged-DirectFT-SFT',       False),
    ('DirectFT-DPO',        _DPO_A   + '/merged-DirectFT-DPO',       False),
    ('DirectFT-RLVR',       _RLVR_A  + '/merged-DirectFT-RLVR',      False),
    ('DirectFT-Instruct',   _INST_A  + '/merged-DirectFT-Instruct',   False),
    ('DirectFT-Tulu3.1',    _T31_A   + '/merged-DirectFT-Tulu3.1',   False),
    ('DirectFT-Llama3-Inst', _INST3_A + '/merged-DirectFT-Llama3-Inst', False),
    ('DirectFT-R1-Distill',  _R1D_A  + '/merged-DirectFT-R1-Distill', False),
]

# --- 3. Cross-delta merged models (4 sources × 8 targets = 32) ---
cross_delta_models = [
    # Base delta → 8 targets (Base is a base model, everything else is chat)
    ('Base2Base',        _BASE_A + '/merged-Base2Base',        True),
    ('Base2SFT',         _BASE_A + '/merged-Base2SFT',         False),
    ('Base2DPO',         _BASE_A + '/merged-Base2DPO',         False),
    ('Base2RLVR',        _BASE_A + '/merged-Base2RLVR',        False),
    ('Base2Instruct',    _BASE_A + '/merged-Base2Instruct',    False),
    ('Base2Tulu3.1',     _BASE_A + '/merged-Base2Tulu3.1',    False),
    ('Base2Llama3-Inst', _BASE_A + '/merged-Base2Llama3-Inst', False),
    ('Base2R1-Distill',  _BASE_A + '/merged-Base2R1-Distill', False),

    # SFT delta → 8 targets
    ('SFT2Base',        _SFT_A + '/merged-SFT2Base',        True),
    ('SFT2SFT',         _SFT_A + '/merged-SFT2SFT',         False),
    ('SFT2DPO',         _SFT_A + '/merged-SFT2DPO',         False),
    ('SFT2RLVR',        _SFT_A + '/merged-SFT2RLVR',        False),
    ('SFT2Instruct',    _SFT_A + '/merged-SFT2Instruct',    False),
    ('SFT2Tulu3.1',     _SFT_A + '/merged-SFT2Tulu3.1',    False),
    ('SFT2Llama3-Inst', _SFT_A + '/merged-SFT2Llama3-Inst', False),
    ('SFT2R1-Distill',  _SFT_A + '/merged-SFT2R1-Distill', False),

    # DPO delta → 8 targets
    ('DPO2Base',        _DPO_A + '/merged-DPO2Base',        True),
    ('DPO2SFT',         _DPO_A + '/merged-DPO2SFT',         False),
    ('DPO2DPO',         _DPO_A + '/merged-DPO2DPO',         False),
    ('DPO2RLVR',        _DPO_A + '/merged-DPO2RLVR',        False),
    ('DPO2Instruct',    _DPO_A + '/merged-DPO2Instruct',    False),
    ('DPO2Tulu3.1',     _DPO_A + '/merged-DPO2Tulu3.1',    False),
    ('DPO2Llama3-Inst', _DPO_A + '/merged-DPO2Llama3-Inst', False),
    ('DPO2R1-Distill',  _DPO_A + '/merged-DPO2R1-Distill', False),

    # RLVR delta → 8 targets
    ('RLVR2Base',        _RLVR_A + '/merged-RLVR2Base',        True),
    ('RLVR2SFT',         _RLVR_A + '/merged-RLVR2SFT',         False),
    ('RLVR2DPO',         _RLVR_A + '/merged-RLVR2DPO',         False),
    ('RLVR2RLVR',        _RLVR_A + '/merged-RLVR2RLVR',        False),
    ('RLVR2Instruct',    _RLVR_A + '/merged-RLVR2Instruct',    False),
    ('RLVR2Tulu3.1',     _RLVR_A + '/merged-RLVR2Tulu3.1',    False),
    ('RLVR2Llama3-Inst', _RLVR_A + '/merged-RLVR2Llama3-Inst', False),
    ('RLVR2R1-Distill',  _RLVR_A + '/merged-RLVR2R1-Distill', False),
]

###############################################################################
# Build model list — auto-skip missing paths
###############################################################################
models = []
_skipped = []

_all_entries = original_targets + direct_ft_baselines + cross_delta_models

for abbr, path, is_base in _all_entries:
    # Local merged models: check existence; HF Hub IDs (org/model) are not local
    is_local = path.startswith('/') or path.startswith(_RESULTS_DIR)
    if is_local and not _os.path.isdir(path):
        _skipped.append(abbr)
        continue

    model_type = TurboMindModel if is_base else TurboMindModelwithChatTemplate
    models.append(
        dict(
            type=model_type,
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
    print('[eval] Skipped (not found): ' + ', '.join(_skipped))
print('[eval] Total models: ' + str(len(models)))
