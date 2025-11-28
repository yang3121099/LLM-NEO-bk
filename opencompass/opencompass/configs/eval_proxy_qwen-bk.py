from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# === 关键：从刚才创建的文件中 Import 类，而不是在这里定义 ===
from opencompass.models.proxy_qwen import ProxyQwen

# ----------------- 路径配置 -----------------
expert_path = '/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'
anti_expert_path = 'Qwen/Qwen3-8B-Base' 
base_target_path = 'Qwen/Qwen3-8B'

# ----------------- 数据集 -----------------
with read_base():
    #from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    # from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
#    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets # 0-shot eval_v2
#    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), []) 

# ----------------- 模型配置 -----------------
models = [
    dict(
        type=ProxyQwen, # 使用 Import 进来的类
        # Base 模型参数
        path=base_target_path,
        tokenizer_path=base_target_path,
        tokenizer_kwargs=dict(
            padding_side='left', 
            truncation_side='left', 
            trust_remote_code=True
        ),
        model_kwargs=dict(
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype='torch.bfloat16'
        ),
        # Proxy 参数
        expert_path=expert_path,
        anti_expert_path=anti_expert_path,
        alpha=1.0, 
        
        # 生成参数
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=1,     
        run_cfg=dict(num_gpus=1), 
    )
]

# ----------------- 运行配置 -----------------
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,  
        task=dict(type=OpenICLInferTask),
    ),
)
