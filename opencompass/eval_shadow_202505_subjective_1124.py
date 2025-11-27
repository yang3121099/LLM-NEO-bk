from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    # 引入主观评测数据集
    # 1. AlpacaEval v2 (Judge by GPT4)
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import alpacav2_datasets
    # 2. MTBench (Single Judge, Different Temperature) - 多轮对话
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets
    # 3. Fofo (Judge by GPT4)
    from opencompass.configs.datasets.subjective.fofo.fofo_judge_new import fofo_datasets

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

# 主观评测数据集
datasets = [
    *alpacav2_datasets,  # AlpacaEval v2
    *mtbench_datasets,    # MT-Bench
    *fofo_datasets,       # Fofo
]

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'outputs/Subjective-Evaluation/shadow-mtbench-alpaca-fofo/'

from opencompass.models import TurboMindModelwithChatTemplate, OpenAI

# API模板配置
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# 待评测模型配置
Baseline_settings = [
    # Qwen3-8B基线
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
    # 微调模型
    ('result-Qwen3-8B-Base-1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I', 
     '/dockerdata/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3-8B-Base-1122/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I', 
     '/dockerdata/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
]

models = []

#######################################################################
#  主观评测模型配置 - 使用do_sample=True, temperature>0            #
#######################################################################

for abbr, path in Baseline_settings:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=512, tp=8),
            gen_config=dict(
                top_k=50,           # 主观评测需要更多样化的输出
                temperature=0.7,    # 主观评测建议使用温度采样
                top_p=0.9, 
                max_new_tokens=2048
            ),
            max_seq_len=16384,
            max_out_len=2048,
            batch_size=8,
            run_cfg=dict(num_gpus=8)
        )
    )

#######################################################################
#                   PART 3  Judge Models (评判模型)                   #
#######################################################################

# 使用GPT-4作为评判模型
judge_models = [
    dict(
        abbr='GPT4-Turbo',
        type=OpenAI,
        path='gpt-4-1106-preview',
        key='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',  # 请替换为你的OpenAI API Key，或通过环境变量 $OPENAI_API_KEY 设置
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        temperature=0,
    )
]

#######################################################################
#                   PART 4  Inference & Evaluation Config             #
#######################################################################

# 推理配置
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)
    ),
)

# 评测配置
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=SubjectiveEvalTask)
    ),
)

# 结果汇总配置
summarizer = dict(type=SubjectiveSummarizer, function='subjective')
