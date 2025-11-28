# configs/eval_proxy_tuned_qwen3.py
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    ######################### Math-7 (mathematical) #########################
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = 'outputs/proxy_tuning_eval/'

from opencompass.models import ProxyTunedHuggingFace, TurboMindModelwithChatTemplate

# ========== Proxy-Tuning 模型配置 ==========
# A (Anti-Expert): Qwen/Qwen3-8B-Base
# A+ (Expert): 你的微调模型
# B (Base): Qwen/Qwen3-8B

ProxyTuning_settings = [
    # (abbr, expert_path)
    ('proxy-B2k-Shadow_2k', '/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
    # 可以添加更多...
]

# 对比用的普通模型
Baseline_settings = [
    ('Qwen3-8B-Instruct', 'Qwen/Qwen3-8B'),
]

models = []

# ========== Proxy-Tuning 模型 ==========
for abbr, expert_path in ProxyTuning_settings:
    models.append(
        dict(
            type=ProxyTunedHuggingFace,
            abbr=abbr,
            base_model_path='Qwen/Qwen3-8B',           # B 模型
            expert_model_path=expert_path,              # A+ 模型
            anti_expert_model_path='Qwen/Qwen3-8B-Base', # A 模型
            alpha=1.0,
            max_seq_len=4096,
            max_out_len=4096,
            # GPU 分配 (8×H20): 每个模型约2张卡
            base_device='cuda:0',
            expert_device='cuda:3',
            anti_expert_device='cuda:5',
            # 或者用自动分配
            # use_auto_device_map=True,
            batch_size=1,  # Proxy-tuning 建议 batch_size=1
            run_cfg=dict(num_gpus=6),
        )
    )

# ========== 对比基线模型 (LMDeploy 加速) ==========
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
            run_cfg=dict(num_gpus=8)
        )
    )
