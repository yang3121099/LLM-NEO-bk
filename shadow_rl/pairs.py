#!/usr/bin/env python3
"""Manifest of the matched Search-R1 base/instruct RL checkpoint pairs.

Every released pair trained *both* the base and the instruct checkpoint of the
same model under the same RL recipe, which is what lets us test Shadow-FT
grafting on RL without running any RL ourselves.

Run this file directly to print the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

HF_PREFIX = "PeterJinGo/"

ORIGINALS: Dict[str, Dict[str, str]] = {
    "3b": {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"},
    "7b": {"base": "Qwen/Qwen2.5-7B", "instruct": "Qwen/Qwen2.5-7B-Instruct"},
}

# The seven evaluation sets.  NQ and HotpotQA are in-domain (the checkpoints
# were trained on nq_hotpotqa_train); the other five are out-of-domain.
DATASETS: List[str] = [
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
]
IN_DOMAIN = {"nq", "hotpotqa"}

MODEL_ROLES = ["instruct_baseline", "rl_on_instruct", "rl_on_base", "shadow"]


@dataclass(frozen=True)
class Pair:
    version: str
    size: str            # "3b" | "7b"
    algo: str            # "ppo" | "grpo"
    with_search: bool
    rl_base: str         # repo name under PeterJinGo/
    rl_instruct: str

    @property
    def pair_id(self) -> str:
        tag = "search" if self.with_search else "nosearch"
        return f"{self.algo}-{tag}-{self.size}-{self.version}"

    @property
    def base(self) -> str:
        return ORIGINALS[self.size]["base"]

    @property
    def instruct(self) -> str:
        return ORIGINALS[self.size]["instruct"]

    def repo(self, role: str) -> Optional[str]:
        """HuggingFace id for a role, or None for `shadow` (produced locally)."""
        if role == "instruct_baseline":
            return self.instruct
        if role == "rl_on_instruct":
            return HF_PREFIX + self.rl_instruct
        if role == "rl_on_base":
            return HF_PREFIX + self.rl_base
        if role == "shadow":
            return None
        raise KeyError(role)


PAIRS: List[Pair] = [
    # ---- No retrieval server needed: the paper's R1 baseline (reason + answer only).
    # Cheapest to evaluate, and the setting where the base-initialised model wins,
    # so this is the right first experiment.
    Pair("v0.2", "3b", "ppo", False,
         "R1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.2",
         "R1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2"),
    Pair("v0.2", "7b", "ppo", False,
         "R1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.2",
         "R1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo-v0.2"),

    # ---- GRPO with search.
    Pair("v0.3", "3b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3"),
    Pair("v0.3", "7b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3"),
    Pair("v0.2", "3b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.2",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2"),
    Pair("v0.2", "7b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.2",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"),
    Pair("v0.1", "3b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo"),

    # ---- PPO with search.
    Pair("v0.3", "3b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.3"),
    Pair("v0.2", "3b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.2",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2"),
    Pair("v0.2", "7b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.2",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo-v0.2"),
    Pair("v0.1", "3b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo"),
    Pair("v0.1", "7b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo"),
]

PAIRS_BY_ID = {p.pair_id: p for p in PAIRS}


# --------------------------------------------------------------------------- #
# Published reference numbers, used to validate the harness before any merged
# model is trusted.  Taken from the v0.1 paper, so expect only ballpark
# agreement for the v0.2 / v0.3 checkpoints.
# --------------------------------------------------------------------------- #
def _row(nq, tqa, pop, hqa, wiki, mus, bam):
    return dict(zip(DATASETS, (nq, tqa, pop, hqa, wiki, mus, bam)))


REFERENCE: Dict[str, Dict[str, float]] = {
    # untuned instruct baselines, direct inference (no search, no CoT)
    "instruct_baseline/3b/direct": _row(0.106, 0.288, 0.108, 0.149, 0.244, 0.020, 0.024),
    "instruct_baseline/3b/cot":    _row(0.023, 0.032, 0.005, 0.021, 0.021, 0.002, 0.000),
    "instruct_baseline/3b/rag":    _row(0.348, 0.544, 0.387, 0.255, 0.226, 0.047, 0.080),
    "instruct_baseline/7b/direct": _row(0.134, 0.408, 0.140, 0.183, 0.250, 0.031, 0.120),
    "instruct_baseline/7b/cot":    _row(0.048, 0.185, 0.054, 0.092, 0.111, 0.022, 0.232),
    "instruct_baseline/7b/rag":    _row(0.349, 0.585, 0.392, 0.299, 0.235, 0.058, 0.208),

    # R1, no search
    "nosearch/3b/rl_on_base":     _row(0.226, 0.455, 0.173, 0.201, 0.268, 0.055, 0.224),
    "nosearch/3b/rl_on_instruct": _row(0.210, 0.449, 0.171, 0.208, 0.275, 0.060, 0.192),
    "nosearch/7b/rl_on_base":     _row(0.297, 0.539, 0.202, 0.242, 0.273, 0.083, 0.296),
    "nosearch/7b/rl_on_instruct": _row(0.270, 0.537, 0.199, 0.237, 0.292, 0.072, 0.293),

    # GRPO with search
    "grpo/3b/rl_on_base":     _row(0.421, 0.583, 0.413, 0.297, 0.274, 0.066, 0.128),
    "grpo/3b/rl_on_instruct": _row(0.397, 0.565, 0.391, 0.331, 0.310, 0.124, 0.232),
    "grpo/7b/rl_on_base":     _row(0.395, 0.560, 0.388, 0.326, 0.297, 0.125, 0.360),
    "grpo/7b/rl_on_instruct": _row(0.429, 0.623, 0.427, 0.386, 0.346, 0.162, 0.400),

    # PPO with search
    "ppo/3b/rl_on_base":      _row(0.406, 0.587, 0.435, 0.284, 0.273, 0.049, 0.088),
    "ppo/3b/rl_on_instruct":  _row(0.341, 0.545, 0.378, 0.324, 0.319, 0.103, 0.264),
    "ppo/7b/rl_on_base":      _row(0.480, 0.638, 0.457, 0.433, 0.382, 0.196, 0.432),
    "ppo/7b/rl_on_instruct":  _row(0.393, 0.610, 0.397, 0.370, 0.414, 0.146, 0.368),
}


def reference_for(pair: Pair, role: str) -> Optional[Dict[str, float]]:
    """Published numbers for a (pair, role), or None if there is no reference.

    `shadow` has no published number by construction -- that is the experiment.
    """
    if role == "shadow":
        return None
    if role == "instruct_baseline":
        return REFERENCE.get(f"instruct_baseline/{pair.size}/direct")
    key = "nosearch" if not pair.with_search else pair.algo
    return REFERENCE.get(f"{key}/{pair.size}/{role}")


def reference_avg(pair: Pair, role: str) -> Optional[float]:
    ref = reference_for(pair, role)
    return sum(ref.values()) / len(ref) if ref else None


if __name__ == "__main__":
    print(f"{len(PAIRS)} pairs\n")
    hdr = f"{'pair_id':<26} {'algo':<5} {'size':<5} {'search':<7} rl_base"
    print(hdr)
    print("-" * len(hdr))
    for p in PAIRS:
        print(f"{p.pair_id:<26} {p.algo:<5} {p.size:<5} "
              f"{str(p.with_search):<7} {HF_PREFIX}{p.rl_base}")
    print(f"\ndatasets: {', '.join(DATASETS)}")
    print(f"in-domain: {', '.join(sorted(IN_DOMAIN))}")
