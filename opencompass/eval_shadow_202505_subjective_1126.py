from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

with read_base():
    from opencompass.configs.datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import alpacav2_datasets
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare import compassarena_datasets
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import arenahard_datasets
    from opencompass.configs.datasets.subjective.compassbench.compassbench_compare import compassbench_datasets
    from opencompass.configs.datasets.subjective.fofo.fofo_judge import fofo_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import mtbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import mtbench101_datasets

datasets = [
    #*alignbench_datasets,
    #*alpacav2_datasets,
    #*arenahard_datasets,
    *compassarena_datasets,
    #*compassbench_datasets,
    #*fofo_datasets,
    #*mtbench_datasets,
    *mtbench101_datasets,
]

work_dir = f'outputs/Subjective-Evaluation/shadow-comprehensive/'

from opencompass.models import TurboMindModelwithChatTemplate

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

Baseline_settings = [
    # 改成你的本地路径
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
    ('result-Qwen3-8B-Base-1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I', 
     '/dockerdata/LLM-NEO-bk/results/1122/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
    ('result-Qwen3-8B-Base-1122/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I', 
     '/dockerdata/LLM-NEO-bk/results/1122/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
]

models = []

# 🔧 关键修改：单卡推理
for abbr, path in Baseline_settings:
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(
                session_len=16384, 
                max_batch_size=512, 
                tp=8  # 单卡推理
            ),
            gen_config=dict(
                top_k=50,
                temperature=0.7,
                top_p=0.9, 
                max_new_tokens=2048
            ),
            max_seq_len=16384,
            max_out_len=2048,
            batch_size=128,
            run_cfg=dict(num_gpus=8)  # 使用1个GPU
        )
    )

judge_models = []

# 🔧 关键修改：使用 SizePartitioner + max_num_workers=1
infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=100000,  # 增大任务大小
        gen_task_coef=20,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,  # 同时只运行1个任务
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=SubjectiveEvalTask)
    ),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
