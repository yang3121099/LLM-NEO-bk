# OpenCompass evaluation config for Shadow-FT tool-use experiments
#
# Evaluates tool-use capabilities (T-Eval full + IFEval) for models
# trained on dolci_instruct_tool_use dataset.
#
# Prerequisites:
#   1. Download T-Eval data:  bash scripts/download_eval_data.sh
#   2. Run eval:
#      cd opencompass
#      python3 ./run.py ./eval_tool_use.py -r $(date +%m%d%H%M%S)

import os as _os
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# Resolve RESULTS_DIR relative to this config file
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), 'results')

# Auto-detect GPU count
_NUM_GPUS = 1
try:
    import subprocess as _sp
    _NUM_GPUS = max(1, len(
        _sp.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()))
except Exception:
    _NUM_GPUS = 1
del _os, _SCRIPT_DIR

with read_base():
    ##################### Tool Use (T-Eval full: 7 subtasks) ####################
    from opencompass.configs.datasets.teval.teval_en_gen_1ac254 import teval_datasets

    ################## Instruction Following ##################
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    ################## General baselines ##################
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import TurboMindModelwithChatTemplate

work_dir = 'outputs/tool-use-sft/'

# ======= Original Instruct baselines (lmdeploy) =======
HF_baselines = [
    ('Llama-3.1-8B-Instruct-hf', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
]

# ======= Trained models: tool-use SFT (B2I Shadow-FT, I2I baseline) =======
Baseline_settings = [
    # dolci_instruct_tool_use 2k
    ('Llama-3.1-8B-dolci_tool-2k-B2I', '$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I'),
    ('Llama-3.1-8B-dolci_tool-2k-I2I', '$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I'),
    ('Qwen3-8B-dolci_tool-2k-B2I', '$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I'),
    ('Qwen3-8B-dolci_tool-2k-I2I', '$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I'),
    # dolci_instruct_mix 2k (no-tools baseline for comparison)
    ('Llama-3.1-8B-dolci_mix-2k-B2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I'),
    ('Llama-3.1-8B-dolci_mix-2k-I2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I'),
    ('Qwen3-8B-dolci_mix-2k-B2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I'),
    ('Qwen3-8B-dolci_mix-2k-I2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I'),
]

models = []

# Original Instruct baselines
for abbr, path in HF_baselines:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=4096, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=2048,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )

for abbr, path in Baseline_settings:
    if '$RESULTS_DIR' in path:
        path = path.replace('$RESULTS_DIR', RESULTS_DIR)
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=4096, max_batch_size=4096, tp=_NUM_GPUS),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=2048),
            max_seq_len=4096,
            max_out_len=2048,
            batch_size=2048,
            run_cfg=dict(num_gpus=_NUM_GPUS),
        )
    )
