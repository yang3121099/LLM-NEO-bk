# ========================================================
from mmengine.config import read_base
# 原来的 import 可以保留，也可以删掉，但为了 debug 模式兼容建议保留
try:
    from opencompass.models.proxy_qwen import ProxyQwen
except ImportError:
    pass

# ================== 【新增这几行】 ==================
# 这告诉所有子进程：在干活之前，先把 opencompass.models.proxy_qwen 导入进来
# 这样 @register_module 就会运行，注册表就有了
custom_imports = dict(
    imports=['opencompass.models.proxy_qwen'],
    allow_failed_imports=False
)
# ====================================================
# 读取数据集配置（这里以 MMLU 为例，你可以换成 ceval 等）
with read_base():
    # 如果你想测数学，可以解开下面这行
    # from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

# 选择要跑的数据集

    from opencompass.configs.summarizers.chat_core_shadow_2511_med import summarizer
    ######################### Math-7 (mathematical) #########################
    #from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets   # noqa: F401, F403
    #from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets # minerva_math
    #from opencompass.configs.datasets.TheoremQA.ThroremQA_0shot_cot_gen_8acdf7 import TheoremQA_datasets # 0-shot
    from opencompass.configs.datasets.Medbullets.medbullets_gen_60c8f5 import (
        medbullets_datasets,
    )
# from opencompass.configs.datasets.humaneval_plus.humaneval_plus_openai_simple_evals_gen_159614 import humaneval_plus_datasets 
    ######################## Code-3 (coding) #########################
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
# 定义模型配置
models = [
    dict(
        # 如果你用了 custom_imports，这里建议改回字符串形式，避免 pickle 问题
        # 但保留类对象通常也是可以的，保险起见可以用字符串完整路径
        type='opencompass.models.proxy_qwen.ProxyQwen', 
        abbr='qwen3-8b-proxy-tuning',
        path="Qwen/Qwen3-8B",
        # ... 其他参数保持不变 ...
        expert_path="/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B",
        anti_expert_path="Qwen/Qwen3-8B-Base",
        tokenizer_path="Qwen/Qwen3-8B",
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=512,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
# 推理运行配置
# 你有 8 张 H20，这里设置 max_num_workers=8
# 意味着会同时启动 8 个进程，每个进程加载 3 个模型（B, A+, A），并行跑不同的题目
infer = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=1,  # 利用你的 8 卡并行
        task=dict(type='OpenICLInferTask'),
    ),
)
