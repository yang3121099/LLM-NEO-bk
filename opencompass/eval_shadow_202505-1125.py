
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    # from opencompass.configs.summarizers.chat_core import summarizer
    from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer

    #######################################################################
    #                          PART 1  Datasets List                      #
    #######################################################################
    
    # # ######################### Reasoning-9 (general reasoning) #########################
    # from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    # from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import  mmlu_pro_datasets  #mmlu_pro_gen_cdbebf
    # from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets # few-shot
    # from opencompass.configs.datasets.bbh.bbh_0shot_nocot_gen_925fc4 import bbh_datasets as bbh3_datasets #0-shot
    # from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import  drop_datasets
    # from opencompass.configs.datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets 
    # from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets # ARC_c   
    # from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets #noCoT openai_simple and 0-shot

    ######################### Math-7 (mathematical) #########################
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets   # noqa: F401, F403
    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import minerva_math_datasets # minerva_math

    # from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import math_datasets as minerva_math_datasets # minerva_math
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets # MATH
    # from opencompass.configs.datasets.TheoremQA.ThroremQA_0shot_cot_gen_8acdf7 import TheoremQA_datasets # 0-shot
    from opencompass.configs.datasets.SVAMP.svamp_gen_fb25e4 import svamp_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets as gsm8k_0shot_datasets # 0-shot eval_v2
    from opencompass.configs.datasets.math.math_500_gen import math_datasets as math_500_datasets  # math_500
    # from opencompass.configs.datasets.mgsm.mgsm_gen import mgsm_datasets
    ######################### Code-3 (coding) #########################
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets  # noqa: F401, F403

    # # original OpenCompass may has bug for MBPP and Humaneval+
    # from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets 
    from opencompass.configs.datasets.humaneval_plus.humaneval_plus_openai_simple_evals_gen_159614 import humaneval_plus_datasets 

    
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'outputs/Rebuttal-0729/shadow-example/'

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel

# input .sh output "##### Evaluation list #####" below
Baseline_settings = [
    
# # Qwen3
('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
# ('Meta-Llama-3.1-8b-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),    
# ('Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'),    



('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2I-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2I-s10'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2I-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2I-s20'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2I-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2I-s30'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2I-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2I-s40'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2I-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2I-s50'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2I-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2I-s60'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2I-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2I-s70'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2I-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2I-s80'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2I-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2I-s90'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2I-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2I-s100'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2I-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2I-s110'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2I-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2I-s120'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2I-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2I-s130'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2I-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2I-s140'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2I-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2I-s150'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2I-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2I-s160'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2I-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2I-s170'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2I-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2I-s180'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2I-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2I-s190'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2I-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2I-s200'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2I-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2I-s210'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2I-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2I-s220'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2I-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2I-s230'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2I-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2I-s240'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2I-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2I-s250'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2I-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2I-s260'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2I-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2I-s270'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2I-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2I-s280'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2I-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2I-s290'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2I-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2I-s300'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2I-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2I-s310'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2I-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2I-s320'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2I-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2I-s330'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2I-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2I-s340'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2I-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2I-s350'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2I-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2I-s360'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2I-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2I-s370'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2I-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2I-s380'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2I-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2I-s390'),
('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2I-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2I-s400'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2I-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2I-s10'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2I-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2I-s20'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2I-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2I-s30'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2I-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2I-s40'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2I-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2I-s50'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2I-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2I-s60'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2I-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2I-s70'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2I-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2I-s80'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2I-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2I-s90'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2I-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2I-s100'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2I-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2I-s110'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2I-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2I-s120'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2I-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2I-s130'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2I-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2I-s140'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2I-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2I-s150'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2I-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2I-s160'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2I-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2I-s170'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2I-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2I-s180'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2I-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2I-s190'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2I-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2I-s200'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2I-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2I-s210'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2I-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2I-s220'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2I-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2I-s230'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2I-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2I-s240'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2I-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2I-s250'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2I-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2I-s260'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2I-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2I-s270'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2I-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2I-s280'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2I-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2I-s290'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2I-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2I-s300'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2I-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2I-s310'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2I-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2I-s320'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2I-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2I-s330'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2I-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2I-s340'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2I-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2I-s350'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2I-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2I-s360'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2I-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2I-s370'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2I-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2I-s380'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2I-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2I-s390'),
('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2I-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2I-s400'),


]

BASE_settings=[
    
    
# ('Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base'),
# ('Meta-Llama-3.1-8b-Base', 'meta-llama/Meta-Llama-3-8B'),    
# ('Llama-3.2-1B-Base', 'meta-llama/Llama-3.2-1B'),    



]
models = []


# #######################################################################
# #              gpu=4, max_new_tokens=4096, batch_size=2048             #
# #######################################################################

for abbr, path in Baseline_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=16384, max_batch_size=4096, tp=8),
            gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
            max_seq_len=16384,
            max_out_len=4096,
            batch_size=2048,
            run_cfg=dict(num_gpus=8)
        )
    )    
    
# #######################################################################
# #              gpu=4, max_new_tokens=4096, batch_size=512             #
# #######################################################################

# for abbr, path in Baseline_settings:  ## classic 4096
#     models.append(
#         dict(
#             type=TurboMindModelwithChatTemplate,
#             abbr=abbr,
#             path=path,
#             engine_config=dict(session_len=16384, max_batch_size=4096, tp=8),
#             gen_config=dict(top_k=1, temperature=0, top_p=0.9, max_new_tokens=4096),
#             max_seq_len=16384,
#             max_out_len=4096,
#             batch_size=2048,
#             run_cfg=dict(num_gpus=8)
#         )
#     )    
    
    
    


for abbr, path in BASE_settings:  ## classic 4096
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
    
# 
    

models = models

