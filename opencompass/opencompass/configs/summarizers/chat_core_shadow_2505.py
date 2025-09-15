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
    ["winogrande_prompt_1", "accuracy"],
    ["drop", "accuracy"],
    ["ARC-c", "accuracy"],
    ["bbh", "naive_average"],
    ["bbh_0shot", "naive_average"],
    ["GPQA_diamond", "accuracy"],
    ["TheoremQA", "score"],

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
    + livecodebench_groups
    + code_v1_groups
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
