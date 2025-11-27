from mmengine.config import read_base

with read_base():
    from .groups.bbh import bbh_summary_groups, bbh_0shot_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.mmlu_pro import mmlu_pro_summary_groups
    from .groups.mgsm import mgsm_summary_groups   # 新增：MGSM 汇总


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

math_groups = [
    dict(
        name="Math",
        subsets=[
            ["math", "accuracy"],
            ["math-500", "accuracy"],
            ["minerva_math", "accuracy"],
            ["gsm8k", "accuracy"],
            ["gsm8k_0shot", "accuracy"],
            ["aime2024", "accuracy"],
            ["svamp", "accuracy"],
        ],
    )
]

code_groups = [
    dict(
        name="Code",
        subsets=[
            ["openai_humaneval", "humaneval_pass@1"],
            ["humaneval_plus", "humaneval_plus_pass@1"],
            ["lcb_code_generation", "pass@1"],
        ],
    )
]

general_v2_groups = [
    dict(
        name="General_v2",
        subsets=[
            ["mmlu", "naive_average"],
            ["mmlu_pro", "naive_average"],
            ["winogrande_prompt_1", "accuracy"],
            ["drop", "accuracy"],
            ["ARC-c", "accuracy"],
            ["bbh", "naive_average"],
            ["bbh_0shot", "naive_average"],
            ["GPQA_diamond", "accuracy"],
            ["TheoremQA", "score"],
        ],
    )
]

# ---------------------------------------------------------------------------
# MedQA_all group: average over three MedQA subsets
# ---------------------------------------------------------------------------

medqa_all_group = [
    dict(
        name="MedQA_all",
        subsets=[
            ["MedQA_Mainland", "accuracy"],
            ["MedQA_Taiwan", "accuracy"],
            ["MedQA_US", "accuracy"],
        ],
    )
]

# ---------------------------------------------------------------------------
# Medical Section
# ---------------------------------------------------------------------------

medical_groups = [
    dict(
        name="Medical",
        subsets=[
            # 在 Medical 的 avg 里，把 MedQA 三个子集作为一个整体 MedQA_all
            ["MedQA_all", "naive_average"],
            ["medbullets", "accuracy"],
            ["ProteinLMBench", "accuracy"],
            # 按要求：不包含 medmcqa / MedCalc_Bench
        ],
    )
]

# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

livecodebench_groups = [
    dict(
        name="LiveCodeBench",
        subsets=[
            ["lcb_code_execution", "pass@1"],
            ["lcb_code_generation", "pass@1"],
            ["lcb_test_output", "pass@1"],
        ],
    )
]

code_v1_groups = [
    dict(
        name="Code_v1",
        subsets=[
            ["openai_humaneval", "humaneval_pass@1"],
            ["humaneval_plus", "humaneval_plus_pass@1"],
            ["LiveCodeBench", "naive_average"],
        ],
    )
]

# ---------------------------------------------------------------------------
# Averages
# ---------------------------------------------------------------------------

average_groups2 = [
    {"name": "average_math2", "subsets": [["Math", "naive_average"]]},
    {"name": "average_code1", "subsets": [["Code_v1", "naive_average"]]},
    {"name": "average_general2", "subsets": [["General_v2", "naive_average"]]},
    {"name": "average_medical", "subsets": [["Medical", "naive_average"]]},
    {
        "name": "average2",
        "subsets": [
            ["average_math2", "naive_average"],
            ["average_code1", "naive_average"],
            ["average_general2", "naive_average"],
            ["average_medical", "naive_average"],
        ],
    },
]

# ---------------------------------------------------------------------------
# Dataset abbreviations (table display)
# ---------------------------------------------------------------------------

dataset_abbrs = [
    # Math
    "--------- Math ---------",
    ["math", "accuracy"],
    ["math-500", "accuracy"],
    ["minerva_math", "accuracy"],
    ["gsm8k", "accuracy"],
    ["gsm8k_0shot", "accuracy"],
    ["aime2024", "accuracy"],
    ["svamp", "accuracy"],

    # Code
    "--------- Code ---------",
    ["openai_humaneval", "humaneval_pass@1"],
    ["sanitized_mbpp", "score"],
    ["humaneval_plus", "humaneval_plus_pass@1"],
    ["lcb_code_generation", "pass@1"],

    # General
    "--------- General ---------",
    ["mmlu", "naive_average"],
    ["mmlu_pro", "naive_average"],
    ["winogrande_prompt_1", "accuracy"],
    ["drop", "accuracy"],
    ["ARC-c", "accuracy"],
    ["bbh", "naive_average"],
    ["bbh_0shot", "naive_average"],
    ["GPQA_diamond", "accuracy"],
    ["TheoremQA", "score"],

    "",  # blank line

    # Medical
    "--------- Medical ---------",
    ["MedQA_Mainland", "accuracy"],
    ["MedQA_Taiwan", "accuracy"],
    ["MedQA_US", "accuracy"],
    ["MedQA_all", "naive_average"],
    ["medbullets", "accuracy"],
    ["ProteinLMBench", "accuracy"],

    "",  # blank line

    # MGSM (Multilingual Grade School Math)
    "--------- MGSM ---------",
    ["mgsm_bn", "accuracy"],
    ["mgsm_de", "accuracy"],
    ["mgsm_en", "accuracy"],
    ["mgsm_es", "accuracy"],
    ["mgsm_fr", "accuracy"],
    ["mgsm_ja", "accuracy"],
    ["mgsm_ru", "accuracy"],
    ["mgsm_sw", "accuracy"],
    ["mgsm_te", "accuracy"],
    ["mgsm_th", "accuracy"],
    ["mgsm_zh", "accuracy"],
    ["mgsm_latin", "naive_average"],
    ["mgsm_non_latin", "naive_average"],
    ["mgsm", "naive_average"],

    "",  # blank line

    # LiveCodeBench
    "--------- LiveCodeBench ---------",
    ["lcb_code_execution", "pass@1"],
    ["lcb_code_generation", "pass@1"],
    ["lcb_test_output", "pass@1"],
    ["LiveCodeBench", "naive_average"],

    "",  # blank line

    # Section AVG
    "--------- Section AVG ---------",
    ["Math", "naive_average"],
    ["Code_v1", "naive_average"],
    ["General_v2", "naive_average"],
    ["Medical", "naive_average"],

    "",  # blank line

    # Overall AVG
    "--------- Overall AVG ---------",
    ["average2", "naive_average"],
]

# ---------------------------------------------------------------------------
# Summary groups (order matters)
# ---------------------------------------------------------------------------

summary_groups = (
    bbh_summary_groups
    + bbh_0shot_summary_groups
    + mmlu_summary_groups
    + mmlu_pro_summary_groups
    + math_groups
    + code_groups
    + medqa_all_group      # 先算 MedQA_all
    + medical_groups       # 再算 Medical
    + mgsm_summary_groups  # MGSM 各语言 + mgsm_latin / non_latin / mgsm
    + livecodebench_groups
    + code_v1_groups
    + general_v2_groups
    + average_groups2
)

# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

summarizer = dict(
    dataset_abbrs=dataset_abbrs,
    summary_groups=summary_groups,
)

