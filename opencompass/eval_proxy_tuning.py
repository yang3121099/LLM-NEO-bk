from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
# 导入上面定义的类，请确保路径正确
# 如果上面的代码保存在 opencompass/models/proxy_tuning_model.py
from opencompass.models.proxy_tuning_model import ProxyTuningQwen 

# 读取你需要的数据集配置，比如 MMLU, CMMLU 等
with read_base():
    # from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


# 定义模型路径
# B 模型 (Base Large/Main Model)
base_path = "Qwen/Qwen3-8B" 
# A+ 模型 (Expert, Tuned)
expert_path = "/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B"
# A 模型 (Anti-Expert, Untuned)
anti_expert_path = "Qwen/Qwen3-8B-Base"

models = [
    dict(
        type=ProxyTuningQwen,
        abbr='qwen3-8b-proxy-tuning', # 显示在结果中的名字
        path=base_path,
        expert_path=expert_path,
        anti_expert_path=anti_expert_path,
        tokenizer_path=base_path,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1, # Proxy Tuning 强烈建议 Batch Size = 1，因为KV Cache管理复杂
        run_cfg=dict(num_gpus=1, num_procs=1), # 单个进程占用1张卡
    )
]

# 推理配置
# 由于我们有8张卡，我们可以并行运行8个这样的模型实例
infer = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=8, # 这里设置为8，充分利用你的8张H20
        task=dict(type='OpenICLInferTask'),
    ),
)
