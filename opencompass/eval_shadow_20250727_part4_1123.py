from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.models import OpenAI

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################
    
    # ######################### 主观评测数据集 (Subjective Evaluation) #########################
    from opencompass.configs.datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import alpacav2_datasets
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare import compassarena_datasets
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import arenahard_datasets
    from opencompass.configs.datasets.subjective.compassbench.compassbench_compare import compassbench_datasets
    from opencompass.configs.datasets.subjective.fofo.fofo_judge import fofo_datasets
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import wildbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import mtbench101_datasets

    # ######################### 以下为注释掉的原有数据集 #########################
    # ######################### Math-7 (mathematical) #########################
    # from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets
    # from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets
    # from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets

# 合并所有主观评测数据集
datasets = [
    *alignbench_datasets, 
    *alpacav2_datasets, 
    *arenahard_datasets,
    *compassarena_datasets, 
    *compassbench_datasets, 
    *fofo_datasets,
    *mtbench_datasets, 
    *mtbench101_datasets, 
    *wildbench_datasets
]

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'outputs/Rebuttal-0729/Subjective'

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# input .sh output, for B2B and I2B only
Base_settings = [    
    # # Qwen3
    ('Qwen3-8B-Base-hf', 'Qwen/Qwen3-8B-Base'),
]


# input .sh output, for B2I and I2I only
Instruct_settings = [
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),   
    ('result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I'),
    ('result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I'),
    ('result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2I'),
    ('result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2I'),
]

models = []

#######################################################################
#              gpu=4, max_new_tokens=4096, batch_size=512             #
#######################################################################

for abbr, path in Base_settings:
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1)
        )
    )    

for abbr, path in Instruct_settings:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=1),
            gen_config=dict(top_k=50, temperature=0.7, top_p=0.9, max_new_tokens=4096),  # 主观评测建议开启采样
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=1)
        )
    )    

#######################################################################
#                    PART 3  Judge Model Configuration                #
#######################################################################

judge_models = [
    dict(
        abbr='GPT4-Turbo',
        type=OpenAI,
        path='gpt-4-1106-preview',
        key='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',  # 请替换为你的 OpenAI API Key，或通过环境变量 $OPENAI_API_KEY 设置
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        temperature=0,
    )
]

#######################################################################
#                    PART 4  Inference Configuration                  #
#######################################################################

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)
    ),
)

#######################################################################
#                    PART 5  Evaluation Configuration                 #
#######################################################################

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

#######################################################################
#                    PART 6  Summarizer Configuration                 #
#######################################################################

summarizer = dict(type=SubjectiveSummarizer, function='subjective')