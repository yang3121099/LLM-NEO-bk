"""Toy numeric tests for the accumulator + metric layer.

We feed hand-chosen tiny tensors whose stage deltas have known geometry, then
check that the derived metrics match closed-form values.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accumulators import Accumulator  # noqa: E402
from metrics import compute_group_metrics  # noqa: E402


def _feed(theta_base, theta_sft, theta_dpo, theta_rlvr):
    acc = Accumulator()
    thetas = {
        "base": np.asarray(theta_base, dtype=np.float64),
        "sft": np.asarray(theta_sft, dtype=np.float64),
        "dpo": np.asarray(theta_dpo, dtype=np.float64),
        "rlvr": np.asarray(theta_rlvr, dtype=np.float64),
    }
    vecs = {
        "sft": thetas["sft"] - thetas["base"],
        "dpo": thetas["dpo"] - thetas["sft"],
        "rlvr": thetas["rlvr"] - thetas["dpo"],
        "post": thetas["rlvr"] - thetas["base"],
    }
    acc.update(thetas, vecs)
    return acc


def test_orthogonal_path():
    # Steps along three orthogonal axes.
    acc = _feed([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])
    m = compute_group_metrics(acc)

    assert math.isclose(m["norm_sft"], 1.0, rel_tol=1e-9)
    assert math.isclose(m["norm_post"], math.sqrt(3), rel_tol=1e-9)

    # Orthogonal consecutive steps -> 90 deg turns -> curvature = pi.
    assert math.isclose(m["phi_sft_dpo"], math.pi / 2, rel_tol=1e-6)
    assert math.isclose(m["curvature"], math.pi, rel_tol=1e-6)

    # straightness = sqrt(3)/3.
    assert math.isclose(m["straightness"], math.sqrt(3) / 3, rel_tol=1e-6)
    assert math.isclose(m["cancellation"], 1 - math.sqrt(3) / 3, rel_tol=1e-6)

    # chord alignment of each unit step with the (1,1,1) chord.
    assert math.isclose(m["chord_align_sft"], 1 / math.sqrt(3), rel_tol=1e-6)
    assert math.isclose(m["residual_to_chord_sft"], math.sqrt(2 / 3), rel_tol=1e-6)

    # progress along chord.
    assert math.isclose(m["progress_sft"], 1 / 3, rel_tol=1e-6)

    # Each step orthogonal to the others -> full orthogonal residual.
    assert math.isclose(m["orth_residual_ratio_dpo"], 1.0, rel_tol=1e-6)
    assert math.isclose(m["orth_residual_ratio_rlvr"], 1.0, rel_tol=1e-6)

    # Identity gram -> uniform energy -> effective rank 3, pc1 = 1/3.
    assert math.isclose(m["pc1_energy"], 1 / 3, rel_tol=1e-6)
    assert math.isclose(m["effective_rank"], 3.0, rel_tol=1e-6)

    # Equal-magnitude steps -> equal stage fractions.
    assert math.isclose(m["stage_fraction_sft"], 1 / 3, rel_tol=1e-6)


def test_straight_path():
    # Three colinear steps along the same axis.
    acc = _feed([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0])
    m = compute_group_metrics(acc)

    # Tolerances here are loose because the eps in the cosine denominator nudges
    # a perfect cos=1 down by ~1e-12, which arccos amplifies to ~1e-6.
    assert math.isclose(m["straightness"], 1.0, rel_tol=1e-6)
    assert math.isclose(m["cancellation"], 0.0, abs_tol=1e-9)
    assert math.isclose(m["curvature"], 0.0, abs_tol=1e-4)
    assert math.isclose(m["chord_align_rlvr"], 1.0, rel_tol=1e-6)
    assert math.isclose(m["residual_to_chord_rlvr"], 0.0, abs_tol=1e-4)

    # Rank-1 stage subspace.
    assert math.isclose(m["pc1_energy"], 1.0, rel_tol=1e-6)
    assert math.isclose(m["effective_rank"], 1.0, rel_tol=1e-6)

    # Colinear -> no orthogonal residual.
    assert math.isclose(m["orth_residual_ratio_dpo"], 0.0, abs_tol=1e-6)
    assert math.isclose(m["orth_residual_ratio_rlvr"], 0.0, abs_tol=1e-6)


def test_sigma_and_norm_accumulation_across_tensors():
    # Two tensors fold additively into one group.
    acc = Accumulator()
    for theta in ([0.0], [2.0]):
        base = np.asarray([0.0])
        sft = np.asarray(theta)
        dpo = np.asarray(theta)
        rlvr = np.asarray(theta)
        thetas = {"base": base, "sft": sft, "dpo": dpo, "rlvr": rlvr}
        vecs = {"sft": sft - base, "dpo": dpo - sft,
                "rlvr": rlvr - dpo, "post": rlvr - base}
        acc.update(thetas, vecs)
    # sum(|v_sft|) over both tensors = 0 + 2 = 2.
    assert math.isclose(acc.l1["sft"], 2.0, rel_tol=1e-9)
    assert acc.numel == 2.0
    assert acc.tensor_count == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All metric tests passed.")
