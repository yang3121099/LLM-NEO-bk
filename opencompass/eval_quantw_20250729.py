from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    # from opencompass.configs.summarizers.chat_core import summarizer
    from opencompass.configs.summarizers.chat_core_0423_ptq import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################
    # from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    # from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_6447e6 import winogrande_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.piqa.piqa_gen_1194eb import piqa_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets as ARC_c2_datasets #效果不好，直接去掉
    # from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets #效果还行，比官方还高    
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets
    
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'outputs/Rebuttal-0729/Quant-lmdeploy'

from opencompass.models import TurboMindModelwithChatTemplate,TurboMindModel

# input .sh output "##### Evaluation list #####" below
Baseline_settings = [

    # ('Llama-2-7b-Baseline', 'meta-llama/Llama-2-7b-hf'),
    # # ('Llama-2-7b-PTQTP-1.58Bit','/home/ubuntu/PTQTB/tptq/20250427/output/llama-2-7b-hf'), #参数不匹配
    # ('Llama-2-7b-PTQTP-1.58Bit-new','yang31210999/Rebuttal-0729_Llama2-7B-PTQTP-1.58b'),

    # # ('Llama-2-7b-E8P-2Bit', 'relaxml/Llama-2-7b-E8P-2Bit'), #不配套
    # # ('Llama-2-7b-E8PRVQ-3Bit', 'relaxml/Llama-2-7b-E8PRVQ-3Bit'), #同上错误
    # # ('Llama-2-7b-E8PRVQ-4Bit', 'relaxml/Llama-2-7b-E8PRVQ-4Bit'), #预计同上
    # ('Llama-2-7b-AQLM-2Bit-1x16-hf','ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf'),
    # ('Llama-2-7b-AQLM-PV-2Bit-1x16-hf','ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-1x16-hf'), #另一台已跑
    # ('Llama-2-7b-AQLM-PV-2Bit-1x16-hf',']ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-1x16-hf'),
    # ('Llama-2-7b-AQLM-PV-1Bit-1x16-hf','ISTA-DASLab/Llama-2-7b-AQLM-PV-1Bit-1x16-hf'), #已完成
    
    # ("",""),
    # ("",""),   
    # ("",""),
    # ("",""),
]

Instruct_settings=[
    
    ("Meta-Llama-3-8B-Instruct","/home/ubuntu/PTQTB/tptq/20250728/Meta-Llama-3-8B-Instruct"),
    ("Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16","ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"),
    ("Meta-Llama-3-8B-Instruct-PTQTP-1.58b","yang31210999/Meta-Llama-3-8B-Instruct-PTQTP-1.58b"),
    # ("Meta-Llama-3.1-8B-Instruct-AQLM-PV-1Bit-1x16-hf","ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-1Bit-1x16-hf"),
    # ("Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf","ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf"),   
    
    
]

models = []


# #######################################################################
# #              gpu=8, max_new_tokens=4096, batch_size=2048             #
# #######################################################################

# from opencompass.models import HuggingFaceBaseModel
# for abbr, path in Baseline_settings:  ## classic 4096
#     models = [
#         dict(
#             type=HuggingFaceBaseModel,
#             abbr=abbr,
#             path=path,
#             max_out_len=1024,
#             batch_size=512,
#             run_cfg=dict(num_gpus=8),
#         )
#     ]

# from opencompass.models import HuggingFaceBaseModel
# for abbr, path in Baseline_settings:  ## classic 4096
#     models = [
#         dict(
#             type=HuggingFaceBaseModel,
#             abbr=abbr,
#             path=path,
#             max_out_len=1024,
#             batch_size=32,
#             run_cfg=dict(num_gpus=1),
#         )
#     ]

# from opencompass.models import VLLM

# for abbr, path in vllm_settings:
#     models.append(
#         dict(
#             type=VLLM,
#             abbr=abbr,
#             path=path,
#             model_kwargs=dict(tensor_parallel_size=8),
#             max_out_len=1024,
#             max_seq_len=2048,
#             batch_size=1024,
#             generation_kwargs=dict(temperature=0),
#             run_cfg=dict(num_gpus=8, num_procs=1),
#         )
#     )

    
# # #######################################################################
# # #              gpu=8, max_new_tokens=4096, batch_size=512             #
# # #######################################################################

for abbr, path in Instruct_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=1024, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=512,
            run_cfg=dict(num_gpus=1)
        )
    )    

for abbr, path in Baseline_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModel,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=1024, tp=1),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=512,
            run_cfg=dict(num_gpus=1)
        )
    )    
    
models = models
# models = []

