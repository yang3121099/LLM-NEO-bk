import os.path as osp
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel
from opencompass.models.openai_api import OpenAISDK

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    # 引入主观评测数据集
    # 1. AlpacaEval v2 (Judge by GPT4)
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import  alpacav2_datasets
    # 2. MTBench (Single Judge, Different Temperature) - 多轮对话
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets
    # 3. Fofo (Judge by GPT4)
    from opencompass.configs.datasets.subjective.fofo.fofo_judge_new import fofo_datasets 

    # 注意：为了避免冲突，我注释掉了你原来的客观数学评测数据集。
    # 主观评测和客观评测通常建议分开运行，因为Eval Task类型不同。
    
    # from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    # from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

# 自动收集上述引入的主观数据集
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                          PART 2  Models List                        #
#######################################################################

# 定义你的模型路径列表
Baseline_settings = [
    # Qwen3
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
    
    # 你的自定义模型路径
    ('result-Qwen3-8B-Base-1124/B-2k-lora-rank128-lr0.0002-mgsm/merged-B2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-2k-lora-rank128-lr0.0002-mgsm/merged-B2I'),
    ('result-Qwen3-8B-Base-1124/I-2k-lora-rank128-lr0.0002-mgsm/merged-I2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-2k-lora-rank128-lr0.0002-mgsm/merged-I2I'),

    ('result-Qwen3-8B-Base-1124/B-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch1/merged-B2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch1/merged-B2I'),
    ('result-Qwen3-8B-Base-1124/I-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch1/merged-I2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch1/merged-I2I'),

    ('result-Qwen3-8B-Base-1124/B-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch3/merged-B2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch3/merged-B2I'),
    ('result-Qwen3-8B-Base-1124/I-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch3/merged-I2I','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-2k-lora-rank64-lr0.0002-mgsm_re_adapt_epoch3/merged-I2I'),
]

BASE_settings = [
    # ('Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base'),
]

models = []

# 配置待评测模型
# 注意：主观评测建议开启 do_sample=True，这里我添加了该参数
for abbr, path in Baseline_settings:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=8),
            gen_config=dict(
                top_k=1, 
                temperature=0, 
                top_p=0.9, 
                max_new_tokens=4096,
                do_sample=True  # 主观评测通常推荐
            ),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=8)
        )
    )    

for abbr, path in BASE_settings:
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(
                top_k=1, 
                temperature=0, 
                top_p=0.9, 
                max_new_tokens=4096,
                do_sample=True
            ),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1)
        )
    )    

#######################################################################
#                          PART 3  Judge Models                       #
#######################################################################

# JudgeLLM 的对话模板
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# 配置 GPT-4 作为裁判
judge_models = [
    dict(
        type=OpenAISDK,
        abbr='gpt-4o-2024-08-06',
        path='gpt-4o-2024-08-06',
        # 如果需要自定义 endpoint，请取消注释并修改
        # openai_api_base='YOUR_API_BASE_URL', 
        key='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',  # <--- 请在这里填入你的 OpenAI API Key
        retry=10,
        meta_template=api_meta_template,
        rpm_verbose=True,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=16384,
        batch_size=8,
        temperature=0.01, # 裁判模型通常温度较低以保证一致性
        tokenizer_path='gpt-4o-2024-08-06'
    )
]

#######################################################################
#                 PART 4  Inference/Evaluation Config                 #
#######################################################################

# 1. 推理阶段 (Inference)
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,
        task=dict(type=OpenICLInferTask) # 使用标准的推理任务
    ),
)

# 2. 评测阶段 (Evaluation) - 关键修改
# 主观评测需要使用 SubjectiveEvalTask 和 SubjectiveNaivePartitioner
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models, # 传入裁判模型
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=SubjectiveEvalTask) # 使用主观评测任务
    ),
)

#######################################################################
#                          PART 5  Utils Config                       #
#######################################################################

# 主观评测汇总器
summarizer = dict(type=SubjectiveSummarizer, function='subjective')

base_exp_dir = 'outputs/Rebuttal-0729/shadow-example/'
work_dir = osp.join(base_exp_dir, 'subjective_eval_results')
