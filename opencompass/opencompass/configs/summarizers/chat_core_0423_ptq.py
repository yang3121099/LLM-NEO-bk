# from mmengine.config import read_base

# with read_base():
#     from .groups.mmlu import mmlu_summary_groups
#     from .groups.cmmlu import cmmlu_summary_groups
#     from .groups.ceval import ceval_summary_groups
#     from .groups.bbh import bbh_summary_groups
#     from .groups.GaokaoBench import GaokaoBench_summary_groups
#     from .groups.lcbench import lcbench_summary_groups
#     from .groups.agieval import agieval_summary_groups

# # summary_groups = [
# #     {
# #         'name': 'average',
# #         'subsets': [
# #             ['mmlu', 'naive_average'],
# #             ['triviaqa_wiki_1shot', 'score'],
# #             ['nq_open_1shot', 'score'],
# #             ['race-high', 'accuracy'],
# #             ['winogrande', 'accuracy'],
# #             ['hellaswag', 'accuracy'],
# #             ['bbh', 'naive_average'],
# #             ['gsm8k', 'accuracy'],
# #             ['math', 'accuracy'],
# #             ['TheoremQA', 'score'],
# #             ['openai_humaneval', 'humaneval_pass@1'],
# #             ['sanitized_mbpp', 'score'],
# #             ['GPQA_diamond', 'accuracy'],
# #             ['IFEval', 'Prompt-level-strict-accuracy'],
# #         ],
# #     },
# #     {
# #         'name': 'core_average',
# #         'subsets': [
# #             ['IFEval', 'Prompt-level-strict-accuracy'],
# #             ['bbh', 'naive_average'],
# #             ['aime2024', 'accuracy'],
# #             ['GPQA_diamond', 'accuracy'],
# #             ['mmlu_pro', 'naive_average'],
# #             ['openai_humaneval', 'humaneval_pass@1'],
# #             ['lcb_code_generation', 'pass@1'],
# #         ],
# #     },
# # ]

# summarizer = dict(
#     dataset_abbrs=[
#         ['average', 'naive_average'],
#         ['mmlu', 'naive_average'],
#         ['triviaqa_wiki_1shot', 'score'],
#         ['winogrande', 'accuracy'],
#         ['hellaswag', 'accuracy'],
#         ['bbh', 'naive_average'],
#         ['gsm8k', 'accuracy'],
#         ['math', 'accuracy'],
#         ['TheoremQA', 'score'],
#         ['openai_humaneval', 'humaneval_pass@1'],
#         ['sanitized_mbpp', 'score'],
#         ['GPQA_diamond', 'accuracy'],
#         ['IFEval', 'Prompt-level-strict-accuracy'],
#         ['agieval', 'naive_average'],

#         ['bbh', 'naive_average'],
#         ['GPQA_diamond', 'accuracy'],
        
#         ['aime2024', 'accuracy'],
        
#         ['mmlu_pro', 'naive_average'],
#         ['lcb_code_generation', 'pass@1'],
#         ['drop', 'accuracy'],

#         '',
#         'mmlu',
#         'mmlu-stem',
#         'mmlu-social-science',
#         'mmlu-humanities',
#         'mmlu-other',
        
#         '',

#     ],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
# )



from mmengine.config import read_base

with read_base():
    from .groups.bbh import bbh_summary_groups,bbh1_summary_groups,bbh2_summary_groups,bbh3_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.mmlu_pro import mmlu_pro_summary_groups
    from .groups.gpqa import GPQA_summary_groups,GPQA1_summary_groups,GPQA2_summary_groups

summary_groups = [
    {
        'name': 'average',
        'subsets': [
            # ['agieval', 'naive_average'],
            ['ARC-c', 'accuracy'],
            ['ARC-c1', 'accuracy'],
            ['ARC-c2', 'accuracy'],
            ['ARC-e', 'accuracy'],
            ['aime2024', 'accuracy'],
            ['bbh', 'naive_average'],
            
            ['bbh1', 'naive_average'],
            ['bbh2', 'naive_average'],
            ['bbh3', 'naive_average'],
            ['BoolQ', 'accuracy'],
            ['hellaswag', 'accuracy'],
            ['piqa', 'accuracy'],
            # ['bigcodebench_hard_complete', 'pass@1'],
            # ['bigcodebench_hard_instruct', 'pass@1'],

            ['drop', 'accuracy'],
            ['drop1', 'accuracy'],
            # ['GPQA_diamond', 'accuracy'],
            # ['GPQA_main', 'accuracy'],
            # ['GPQA_extended', 'accuracy'],
            
            # ['GPQA1_diamond', 'accuracy'],
            # ['GPQA1_main', 'accuracy'],
            # ['GPQA1_extended', 'accuracy'],
            
            # ['GPQA2_diamond', 'accuracy'],
            # ['GPQA2_main', 'accuracy'],
            # ['GPQA2_extended', 'accuracy'],
            
            ['GPQA_diamond', 'accuracy'],
            ['GPQA', 'accuracy'],  
                      
            ['GPQA1_diamond', 'accuracy'],
            ['GPQA1', 'accuracy'],
            
            ['GPQA2_diamond', 'accuracy'],
            ['GPQA2', 'accuracy'],
            
            ['gsm8k', 'accuracy'],
            ['gsm8k1', 'accuracy'],
            ['gsm8k2', 'accuracy'],
            ['gsm8k3', 'accuracy'],
            ['humaneval_plus', 'humaneval_plus_pass@1'],
            ['humaneval_plus1', 'humaneval_plus_pass@1'],
            ['humaneval_plus2', 'humaneval_plus_pass@1'],

            # ['hellaswag', 'accuracy'],
            # ['IFEval', 'Prompt-level-strict-accuracy'],
            # ['IFEval', 'Inst-level-strict-accuracy'],
            # ['IFEval', 'Prompt-level-loose-accuracy'],
            # ['IFEval', 'Inst-level-loose-accuracy'],
            
            ['lcb_code_execution', 'pass@1'],
            ['lcb_code_generation', 'pass@1'],
            ['lcb_test_output', 'pass@1'],

            ['math', 'accuracy'],
            ['math-500', 'accuracy'],
            ['minerva_math', 'accuracy'],

            ['mbpp', 'score'],
            ['mbpp_plus', 'mbpp_plus_pass@1'],

            ['mmlu', 'naive_average'],
            ['mmlu_pro', 'naive_average'],
            
            ['nq', 'score'],
            ['nq_open_1shot', 'score'],
            ['openai_humaneval', 'humaneval_pass@1'],
            
            ['openai_humaneval1', 'humaneval_pass@1'],
            ['openai_humaneval2', 'humaneval_pass@1'],
            ['openai_humaneval3', 'humaneval_pass@1'],
            ['openai_humaneval4', 'humaneval_pass@1'],
            ['openai_humaneval5', 'humaneval_pass@1'],
            ['sanitized_mbpp', 'score'],
            ['sanitized_mbpp1', 'score'],

            # ['simpleqa', 'accuracy_given_attempted'],
            # ['simpleqa', 'f1'],
            ['svamp', 'accuracy'],

            
            ['TheoremQA', 'score'],
            # ['triviaqa_wiki_1shot', 'score'],
            ['winogrande', 'accuracy'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['average', 'naive_average'],
        '',
        # ['agieval', 'naive_average'],
        ['ARC-c', 'accuracy'],
        
        ['ARC-c1', 'accuracy'],
        ['ARC-c2', 'accuracy'],

        ['ARC-e', 'accuracy'],
        ['aime2024', 'accuracy'],
        ['bbh', 'naive_average'],
        ['BoolQ', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['piqa', 'accuracy'],

        ['bbh1', 'naive_average'],
        ['bbh2', 'naive_average'],
        ['bbh3', 'naive_average'],
        # ['bigcodebench_hard_complete', 'pass@1'],
        # ['bigcodebench_hard_instruct', 'pass@1'],

        ['drop', 'accuracy'],
        ['drop1', 'accuracy'],

        # ['GPQA_diamond', 'accuracy'],
        # ['GPQA_main', 'accuracy'],
        # ['GPQA_extended', 'accuracy'],

        # ['GPQA1_diamond', 'accuracy'],
        # ['GPQA1_main', 'accuracy'],
        # ['GPQA1_extended', 'accuracy'],
        
        # ['GPQA2_diamond', 'accuracy'],
        # ['GPQA2_main', 'accuracy'],
        # ['GPQA2_extended', 'accuracy'],
        
        
        ['GPQA_diamond', 'accuracy'],
        ['GPQA', 'accuracy'],  
                    
        ['GPQA1_diamond', 'accuracy'],
        ['GPQA1', 'accuracy'],
        
        ['GPQA2_diamond', 'accuracy'],
        ['GPQA2', 'accuracy'],
        
        
        ['gsm8k', 'accuracy'],
        ['gsm8k1', 'accuracy'],
        ['gsm8k2', 'accuracy'],
        ['gsm8k3', 'accuracy'],

        ['humaneval_plus', 'humaneval_plus_pass@1'],
        ['humaneval_plus1', 'humaneval_plus_pass@1'],
        ['humaneval_plus2', 'humaneval_plus_pass@1'],

        # ['hellaswag', 'accuracy'],
        # ['IFEval', 'Prompt-level-strict-accuracy'],
        # ['IFEval', 'Inst-level-strict-accuracy'],
        # ['IFEval', 'Prompt-level-loose-accuracy'],
        # ['IFEval', 'Inst-level-loose-accuracy'],
        
        ['lcb_code_execution', 'pass@1'],
        ['lcb_code_generation', 'pass@1'],
        ['lcb_test_output', 'pass@1'],

        ['math', 'accuracy'],
        ['math-500', 'accuracy'],
        ['minerva_math', 'accuracy'],

        ['mbpp', 'score'],
        ['mbpp_plus', 'mbpp_plus_pass@1'],

        ['mmlu', 'naive_average'],
        ['mmlu_pro', 'naive_average'],
        ['nq', 'score'],
        ['nq_open_1shot', 'score'],

        ['openai_humaneval', 'humaneval_pass@1'],
        ['openai_humaneval1', 'humaneval_pass@1'],
        ['openai_humaneval2', 'humaneval_pass@1'],
        ['openai_humaneval3', 'humaneval_pass@1'],
        ['openai_humaneval4', 'humaneval_pass@1'],
        ['openai_humaneval5', 'humaneval_pass@1'],

        ['sanitized_mbpp', 'score'],
        ['sanitized_mbpp1', 'score'],

        # ['simpleqa', 'accuracy_given_attempted'],
        # ['simpleqa', 'f1'],
        ['svamp', 'accuracy'],

        
        ['TheoremQA', 'score'],
        # ['triviaqa_wiki_1shot', 'score'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=summary_groups + sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)


# from mmengine.config import read_base
# with read_base():
#     from .groups.cmmlu import cmmlu_summary_groups
#     from .groups.ceval import ceval_summary_groups
#     from .groups.GaokaoBench import GaokaoBench_summary_groups

# other_summary_groups = []
# other_summary_groups.append({'name': 'Exam', 'subsets': ['cmmlu', 'ceval', 'GaokaoBench']})
# other_summary_groups.append({'name': 'Knowledge', 'subsets': ['nq', 'triviaqa']})
# other_summary_groups.append({'name': 'Understanding', 'subsets': ['race-middle', 'race-high', 'lambada']})
# other_summary_groups.append({'name': 'Reasoning', 'subsets': ['piqa', 'obqa', 'AX_b', 'AX_g', 'BoolQ', 'CB', 'COPA', 'MultiRC', 'RTE', 'ReCoRD', 'WiC', 'WSC']})
# other_summary_groups.append({'name': 'Overall', 'subsets': ['Exam', 'Knowledge', 'Understanding', 'Reasoning']})

# summarizer = dict(
#     dataset_abbrs=[
#         'Overall',
#         'Exam',
#         'Knowledge',
#         'Understanding',
#         'Reasoning',
#         '--------- 考试 Exam ---------',  # category
#         'cmmlu',
#         'ceval',
#         'GaokaoBench',
#         '--------- 知识 Knowledge ---------',  # category
#         'nq',
#         'triviaqa',
#         '--------- 理解 Understanding ---------',  # category
#         'race-middle',
#         'race-high',
#         'lambada',
#         'summedits',
#         '--------- 推理 Reasoning ---------',  # category
#         'piqa',
#         'obqa',
#         'AX_b',
#         'AX_g',
#         'BoolQ',
#         'CB',
#         'COPA',
#         'MultiRC',
#         'RTE',
#         'ReCoRD',
#         'WiC',
#         'WSC',
#     ],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []) + other_summary_groups,
# )