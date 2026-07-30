#!/usr/bin/env python3
"""Manifest of the matched Search-R1 base/instruct RL checkpoint pairs.

Every released pair trained *both* the base and the instruct checkpoint of the
same model under the same RL recipe, which is what lets us test Shadow-FT
grafting on RL without running any RL ourselves.

Run this file directly to print the manifest.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

HF_PREFIX = "PeterJinGo/"

# Keyed by the `size` field of a Pair, which is really "which original pair of
# checkpoints does this graft onto" -- it covers backbone family as well as scale.
ORIGINALS: Dict[str, Dict[str, str]] = {
    "3b": {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"},
    "7b": {"base": "Qwen/Qwen2.5-7B", "instruct": "Qwen/Qwen2.5-7B-Instruct"},
    "14b": {"base": "Qwen/Qwen2.5-14B", "instruct": "Qwen/Qwen2.5-14B-Instruct"},
    # v0.3 studied scale up to 32B. Evaluating one needs multiple GPUs (--tp).
    "32b": {"base": "Qwen/Qwen2.5-32B", "instruct": "Qwen/Qwen2.5-32B-Instruct"},
    # Search-R1 trained these too. Note both are gated on HuggingFace -- accept
    # the licence on the model page, or `huggingface-cli login`, before use.
    "llama3.2-3b": {"base": "meta-llama/Llama-3.2-3B",
                    "instruct": "meta-llama/Llama-3.2-3B-Instruct"},
    "llama3.1-8b": {"base": "meta-llama/Llama-3.1-8B",
                    "instruct": "meta-llama/Llama-3.1-8B-Instruct"},
}

# --------------------------------------------------------------------------- #
# Evaluation sets. The seven FlashRAG QA sets are the Search-R1 protocol; NQ and
# HotpotQA are in-domain (the checkpoints trained on nq_hotpotqa_train). GPQA-D
# and SimpleQA are added on top and are a different shape -- see DatasetSpec.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_repo: str
    config: Optional[str]          # HF dataset config, if the repo needs one
    splits: tuple                  # tried in order; first present wins
    question_field: str
    answer_field: str
    kind: str = "qa"               # "qa" (free-form short answer) | "mcq"
    # For mcq: fields holding the distractors, shuffled into lettered options.
    distractor_fields: tuple = ()
    note: str = ""


_FLASHRAG = "RUC-NLPIR/FlashRAG_datasets"


def _qa(name: str) -> DatasetSpec:
    return DatasetSpec(name, _FLASHRAG, name, ("test", "dev", "train"),
                       "question", "golden_answers")


DATASET_SPECS: Dict[str, DatasetSpec] = {
    d: _qa(d) for d in
    ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")
}

# GPQA-Diamond: 198 graduate-level science questions, multiple choice. The repo
# is gated -- accept the licence on the model page or huggingface-cli login.
# Scored by exact match against either the option letter or the answer text, so
# the official qa_em scorer still applies unchanged.
DATASET_SPECS["gpqa_diamond"] = DatasetSpec(
    "gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond", ("train", "test"),
    "Question", "Correct Answer", kind="mcq",
    distractor_fields=("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
    note="gated repo; MCQ scored on letter or answer text",
)

# SimpleQA: short-factoid questions with a single reference answer. NOTE the
# official protocol grades with an LLM judge, not exact match -- EM here is a
# strict lower bound and will read lower than published SimpleQA numbers.
DATASET_SPECS["simpleqa"] = DatasetSpec(
    "simpleqa", "basicv8vc/SimpleQA", None, ("test", "train"),
    "problem", "answer",
    note="official protocol uses an LLM grader; EM here is a strict lower bound",
)

DATASETS: List[str] = list(DATASET_SPECS)
IN_DOMAIN = {"nq", "hotpotqa"}
# Not part of the Search-R1 protocol; no published reference numbers exist.
EXTRA_DATASETS = {"gpqa_diamond", "simpleqa"}

MODEL_ROLES = ["base_baseline", "instruct_baseline",
               "rl_on_instruct", "rl_on_base", "shadow"]


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
        if role == "base_baseline":
            return self.base
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

    # ---- Confirmed present by discovery against the Hub.
    Pair("v0.3", "14b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3"),
    Pair("v0.1", "llama3.2-3b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-em-grpo",
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-grpo"),
    Pair("v0.1", "llama3.2-3b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-em-ppo",
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-ppo"),
]

# --------------------------------------------------------------------------- #
# Candidate pairs: plausible from Search-R1's naming convention and its v0.3
# training scripts, but NOT confirmed to exist as released repos. Both sides of a
# pair have to be published for it to be usable, and a training script only shows
# what was trained. Rather than hardcode ids that may 404, these are probed by
# shadow_rl/discover.py, which writes the confirmed ones to verified_pairs.json
# for this module to pick up.
#
# The v0.3 scripts cover Qwen 3B/7B/14B (+ DeepSeek distills); Llama appears only
# in the v0.1 scripts, so the Llama candidates are listed under both taggings.
# --------------------------------------------------------------------------- #
CANDIDATES: List[Pair] = [
    # v0.3 completions of the Qwen grid
    Pair("v0.3", "7b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo-v0.3"),
    Pair("v0.3", "14b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3",
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-ppo-v0.3"),

    # Llama backbones, v0.3 tagging
    Pair("v0.3", "llama3.2-3b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-em-grpo-v0.3",
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-grpo-v0.3"),
    Pair("v0.3", "llama3.1-8b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-grpo-v0.3",
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-it-em-grpo-v0.3"),
    Pair("v0.3", "llama3.2-3b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-em-ppo-v0.3",
         "SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-ppo-v0.3"),
    Pair("v0.3", "llama3.1-8b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-ppo-v0.3",
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-it-em-ppo-v0.3"),

    # v0.1 gaps. The first release carries NO version suffix on the repo name --
    # `...-em-grpo` rather than `...-em-grpo-v0.1` -- and the set we were given
    # covered 3B GRPO plus 3B/7B PPO, leaving 7B GRPO and the 14B pairs unlisted.
    Pair("v0.1", "7b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo"),
    Pair("v0.1", "14b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo"),
    Pair("v0.1", "14b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo",
         "SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-ppo"),

    # Llama backbones, unversioned (v0.1) tagging, matching the scripts they
    # actually appear in
    Pair("v0.1", "llama3.1-8b", "grpo", True,
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-grpo",
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-it-em-grpo"),
    Pair("v0.1", "llama3.1-8b", "ppo", True,
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-em-ppo",
         "SearchR1-nq_hotpotqa_train-llama3.1-8b-it-em-ppo"),
]

# Pairs confirmed to exist by discover.py get merged in here, so nothing that
# 404s ever reaches the run.
_VERIFIED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verified_pairs.json")
if os.path.exists(_VERIFIED):
    try:
        with open(_VERIFIED) as _fh:
            _known = {p.pair_id for p in PAIRS}
            for _row in json.load(_fh).get("pairs", []):
                _p = Pair(_row["version"], _row["size"], _row["algo"],
                          bool(_row["with_search"]), _row["rl_base"], _row["rl_instruct"])
                if _p.pair_id not in _known:
                    PAIRS.append(_p)
                    _known.add(_p.pair_id)
    except Exception as _exc:
        # A bad file must not break the manifest -- and the warning must go to
        # stderr: run_all.sh captures this module's stdout as the pair list, so
        # anything printed there would be parsed as a pair id.
        import sys as _sys
        print(f"[warn] ignoring {_VERIFIED}: {_exc}", file=_sys.stderr)

PAIRS_BY_ID = {p.pair_id: p for p in PAIRS}
CANDIDATES_BY_ID = {p.pair_id: p for p in CANDIDATES}


# --------------------------------------------------------------------------- #
# Published reference numbers, used to validate the harness before any merged
# model is trusted.  Taken from the v0.1 paper, so expect only ballpark
# agreement for the v0.2 / v0.3 checkpoints.
# --------------------------------------------------------------------------- #
# The published tables cover exactly these seven, in this order. Spelled out
# rather than zipped against DATASETS: that list now also holds GPQA-D and
# SimpleQA, and a silent zip() truncation would misalign every reference number.
_REFERENCE_ORDER = ("nq", "triviaqa", "popqa", "hotpotqa",
                    "2wikimultihopqa", "musique", "bamboogle")


def _row(nq, tqa, pop, hqa, wiki, mus, bam):
    values = (nq, tqa, pop, hqa, wiki, mus, bam)
    assert len(values) == len(_REFERENCE_ORDER)
    return dict(zip(_REFERENCE_ORDER, values))


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
    if role in ("shadow", "base_baseline"):
        # No published direct-inference numbers exist for the raw base model.
        return None
    if role == "instruct_baseline":
        return REFERENCE.get(f"instruct_baseline/{pair.size}/direct")
    key = "nosearch" if not pair.with_search else pair.algo
    return REFERENCE.get(f"{key}/{pair.size}/{role}")


def reference_avg(pair: Pair, role: str, subset=None) -> Optional[float]:
    """Published average, optionally restricted to a subset of the datasets.

    When a run skips datasets, averaging the reference over all seven and ours
    over five would compare different things -- so the caller passes the subset
    it actually evaluated.
    """
    ref = reference_for(pair, role)
    if not ref:
        return None
    keys = [d for d in DATASETS if d in ref and (subset is None or d in subset)]
    return sum(ref[d] for d in keys) / len(keys) if keys else None


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
