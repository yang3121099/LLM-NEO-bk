from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    from opencompass.configs.summarizers.chat_core_0423_ptq import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_few_shot_gen_ba58ea import BoolQ_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    from opencompass.configs.datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_6447e6 import winogrande_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = 'outputs/eval-qwen3-4b-base-bench/'

from opencompass.models import VLLM

models = []

# --- Qwen3-4B (Instruct) ---
models.append(
    dict(
        type=VLLM,
        abbr='Qwen3-4B',
        path='Qwen/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
)

# --- Qwen3-4B-Base ---
models.append(
    dict(
        type=VLLM,
        abbr='Qwen3-4B-Base',
        path='Qwen/Qwen3-4B-Base',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
)
