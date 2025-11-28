

from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    # from opencompass.configs.summarizers.chat_core import summarizer
    #from opencompass.configs.summarizers.chat_core_shadow_2505 import summarizer
    from opencompass.configs.summarizers.chat_core_shadow_2511_med import summarizer
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
 #   from opencompass.configs.datasets.mgsm.mgsm_gen import mgsm_datasets
    ######################### Code-3 (coding) #########################
    # from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets  # noqa: F401, F403

    # # original OpenCompass may has bug for MBPP and Humaneval+
    # from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets 
    # from opencompass.configs.datasets.humaneval_plus.humaneval_plus_openai_simple_evals_gen_159614 import humaneval_plus_datasets 



    ######################### Medical & Bio Benchmarks #########################
    # 下面这几行就是你说要加的几个任务
    # 注意：变量名（xxx_datasets）要和对应 *_gen_*.py 里定义的一致，
    # 我这边给的是“典型命名”，如果 import 报错，进去文件看一下实际变量名改一下就行。
    # MedQA
    from opencompass.configs.datasets.MedQA.MedQA_gen_3bf756 import (
            MedQA_datasets,
    )
    # MedCalc_Bench
    # ProteinLMBench
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_gen_a67965 import (
            proteinlmbench_datasets,   # 若文件里是 proteinlmbench_datasets / proteinlmb_datasets 等，请改成实际名字
    )
    # Medbullets
    from opencompass.configs.datasets.Medbullets.medbullets_gen_60c8f5 import (
        medbullets_datasets,
    )

    # MedCalc_Bench
   # from opencompass.configs.datasets.MedCalc_Bench.MedCalcBench_official_gen_a5155f import (
    #    medcal_bench_datasets,
    #)

    # MedMCQA
    from opencompass.configs.datasets.medmcqa.medmcqa_gen_60c8f5 import (
        medmcqa_datasets,
    )


# 这一行不用动，会把所有 *_datasets 的列表拼起来


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



('result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-medical_o1_reasoning/merged-B2I'),
('result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-medical_o1_reasoning/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-medical_o1_reasoning/merged-I2I'),


#('result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-mgsm/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-mgsm/merged-B2I'),
#('result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-mgsm/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-mgsm/merged-I2I'),


#('result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-open_r1_math/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-open_r1_math/merged-B2I'),
#('result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-open_r1_math/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-open_r1_math/merged-I2I'),

('result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
('result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),

('result-Qwen3-8B-Base-1127/B-4k-lora-rank128-lr0.0002-shadow_medical_mix_sharegpt_4k/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-4k-lora-rank128-lr0.0002-shadow_medical_mix_sharegpt_4k/merged-B2I'),
('result-Qwen3-8B-Base-1127/I-4k-lora-rank128-lr0.0002-shadow_medical_mix_sharegpt_4k/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-4k-lora-rank128-lr0.0002-shadow_medical_mix_sharegpt_4k/merged-I2I'),


('result-Qwen3-8B-Base-1127/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I'),
('result-Qwen3-8B-Base-1127/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I'),


('result-Qwen3-8B-Base-1127/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I-LoRE','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-B2I-LoRE'),
('result-Qwen3-8B-Base-1127/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I-LoRE','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch3/merged-I2I-LoRE'),


('result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-1.0','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-1.0'),
('result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-0.5','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2I-Shadow_2k+medical_o1_reasoning-0.5'),
('result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-1.0','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-1.0'),
('result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-0.5','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-B2B-Shadow_2k+medical_o1_reasoning-0.5'),
('result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-1.0','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-1.0'),
('result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-0.5','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-I2I-Shadow_2k+medical_o1_reasoning-0.5'),

('merged-RE-Adapt-Shadow_2k_re_adapt-0.5+0.5','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-RE-Adapt-Shadow_2k_re_adapt-0.5+0.5'),
('merged-LoRE-Shadow_2k_re_adapt-0.5+0.5','/dockerdata/LLM-NEO-bk/results/1127/result-Qwen3-8B-Base-1127/merged-LoRE-Shadow_2k_re_adapt-0.5+0.5'),
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
