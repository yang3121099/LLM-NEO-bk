from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

# ================= 关键步骤：导入自定义模型 =================
# 这行代码会让 Python 执行 proxy_qwen.py，从而触发 @MODELS.register_module()
# 请确保 proxy_qwen.py 就在 opencompass/models/ 目录下
from opencompass.models.proxy_qwen import ProxyQwen 
# ========================================================

# 读取数据集配置（这里以 MMLU 为例，你可以换成 ceval 等）
with read_base():
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    # 如果你想测数学，可以解开下面这行
    # from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

# 选择要跑的数据集
datasets = [*mmlu_datasets]

# 定义模型配置
models = [
    dict(
        type=ProxyQwen,  # 直接使用导入的类，不要用字符串
        abbr='qwen3-8b-proxy-tuning', # 结果中显示的名字
        
        # Base Model (B) 路径
        path="Qwen/Qwen3-8B", 
        
        # Expert Model (A+) 路径
        expert_path="/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B",
        
        # Anti-Expert Model (A) 路径
        anti_expert_path="Qwen/Qwen3-8B-Base",
        
        # 通用参数
        tokenizer_path="Qwen/Qwen3-8B",
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1, # 必须为 1，因为 LogitsProcessor 处理 Batch 的 KV Cache 很复杂
        run_cfg=dict(num_gpus=1, num_procs=1), # 每个进程独占一张卡
    )
]

# 推理运行配置
# 你有 8 张 H20，这里设置 max_num_workers=8
# 意味着会同时启动 8 个进程，每个进程加载 3 个模型（B, A+, A），并行跑不同的题目
infer = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=8,  # 利用你的 8 卡并行
        task=dict(type='OpenICLInferTask'),
    ),
)
