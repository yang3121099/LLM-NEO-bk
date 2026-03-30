from mmengine.config import read_base

"""Benchmark configuration

This file cleans up the original configuration by:
1. Removing all commented‑out code sections.
2. Dropping GPQA statistics/aggregation, keeping only the **GPQA_diamond** subset.
3. Converting remaining comments to concise, open‑source‑friendly English.
"""

with read_base():
    from .groups.bbh import bbh_summary_groups, bbh_0shot_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.mmlu_pro import mmlu_pro_summary_groups
    from .groups.mgsm import mgsm_summary_groups
    from .groups.teval import teval_summary_groups

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

# ---------------------------------------------------------------------------
# Instruction Following
# ---------------------------------------------------------------------------

ifeval_groups = [
    dict(
        name="IF_Eval",
        subsets=[
            ["IFEval", "Prompt-level-strict-accuracy"],
            ["IFEval", "Inst-level-strict-accuracy"],
            ["IFEval", "Prompt-level-loose-accuracy"],
            ["IFEval", "Inst-level-loose-accuracy"],
        ],
    )
]

# ---------------------------------------------------------------------------
# Tool Use (T-Eval)
# ---------------------------------------------------------------------------

teval_top_groups = [
    dict(
        name="T-Eval",
        subsets=[
            ["teval-review_str_v1", "review_quality"],
        ],
    )
]

general_v2_groups = [
    dict(
        name="General_v2",
        subsets=[
            ["mmlu", "naive_average"],
            ["mmlu_pro", "naive_average"],
            ["winogrande_prompt_2", "accuracy"],
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
    {
        "name": "average2",
        "subsets": [
            ["average_math2", "naive_average"],
            ["average_code1", "naive_average"],
            ["average_general2", "naive_average"],
        ],
    },
]

# ---------------------------------------------------------------------------
# Dataset abbreviations (used by the summarizer)
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
    ["winogrande_prompt_2", "accuracy"],
    ["drop", "accuracy"],
    ["ARC-c", "accuracy"],
    ["bbh", "naive_average"],
    ["bbh_0shot", "naive_average"],
    ["GPQA_diamond", "accuracy"],
    ["TheoremQA", "score"],

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

    # Instruction Following (IFEval)
    "--------- Instruction Following ---------",
    ["IFEval", "Prompt-level-strict-accuracy"],
    ["IFEval", "Inst-level-strict-accuracy"],
    ["IFEval", "Prompt-level-loose-accuracy"],
    ["IFEval", "Inst-level-loose-accuracy"],
    ["IF_Eval", "naive_average"],

    "",  # blank line

    # Tool Use (T-Eval — review only)
    "--------- Tool Use (T-Eval) ---------",
    ["teval-review_str_v1", "review_quality"],

    "",  # blank line

    # LiveCodeBench
    "--------- LiveCodeBench ---------",
    ["lcb_code_execution", "pass@1"],
    ["lcb_code_generation", "pass@1"],
    ["lcb_test_output", "pass@1"],
    ["LiveCodeBench", "naive_average"],

    "",  # blank line

    # Section AVG v2
    "--------- Section AVG ---------",
    ["Math", "naive_average"],
    ["Code_v1", "naive_average"],
    ["General_v2", "naive_average"],

    "",  # blank line

    # Overall AVG v2
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
    + mgsm_summary_groups
    + livecodebench_groups
    + code_v1_groups
    + ifeval_groups
    + teval_summary_groups
    + teval_top_groups
    + general_v2_groups
    + average_groups2
)

# ---------------------------------------------------------------------------
# Summarizer configuration
# ---------------------------------------------------------------------------

summarizer = dict(
    dataset_abbrs=dataset_abbrs,
    summary_groups=summary_groups,
)
