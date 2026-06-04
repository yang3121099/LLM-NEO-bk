"""Streaming float64 accumulators for the static vector-field audit.

The audit never materialises whole-model vectors.  Instead, for every shared
tensor we compute the three adjacent stage deltas plus the post chord and fold
their sums (norms, dot products, sign agreement, sigma numerators/denominators,
top-k overlap) into per-group float64 accumulators.

All inputs to :meth:`Accumulator.update` are expected to be flat ``numpy``
``float64`` arrays so that the arithmetic is exact-ish and torch-independent.
The checkpoint loader converts the raw (possibly bfloat16) tensors to float32
and then to float64 before calling here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Stage delta vectors.
STAGES = ("sft", "dpo", "rlvr", "post")
# Raw checkpoints in lineage order.
CHECKPOINTS = ("base", "sft", "dpo", "rlvr")
# The previous checkpoint each stage delta is measured against (for rel-L2).
STAGE_PREV = {"sft": "base", "dpo": "sft", "rlvr": "dpo", "post": "base"}

# Ordered pairs of stage vectors used for dot products / cosine / sign / top-k.
STAGE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("sft", "dpo"),
    ("sft", "rlvr"),
    ("dpo", "rlvr"),
    ("sft", "post"),
    ("dpo", "post"),
    ("rlvr", "post"),
)

# Checkpoint pairs used for the sigma (relative gap ratio) metric.
SIGMA_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("base", "rlvr"),
)


@dataclass
class Accumulator:
    """Float64 accumulator for a single group."""

    numel: float = 0.0
    tensor_count: int = 0

    # stage -> sum(v**2), sum(|v|), max(|v|)
    norm_sq: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})
    l1: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})
    linf: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})

    # checkpoint -> sum(theta**2), sum(|theta|)
    theta_sq: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CHECKPOINTS})
    abs_sum: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CHECKPOINTS})

    # (a, b) -> sum(v_a * v_b)
    dot: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {p: 0.0 for p in STAGE_PAIRS}
    )

    # (a, b) -> sign agreement counts
    same_sign: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {p: 0.0 for p in STAGE_PAIRS}
    )
    valid_sign: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {p: 0.0 for p in STAGE_PAIRS}
    )

    # (a, b) -> approximate per-tensor top-k overlap (intersection count, total k)
    topk_inter: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {p: 0.0 for p in STAGE_PAIRS}
    )
    topk_total: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {p: 0.0 for p in STAGE_PAIRS}
    )

    # Bookkeeping flag — set when a tied embedding / lm_head tensor folds in.
    has_tied: bool = False

    def update(
        self,
        thetas: Dict[str, np.ndarray],
        vecs: Dict[str, np.ndarray],
        *,
        eps: float = 1e-12,
        topk_cfg: dict | None = None,
    ) -> None:
        """Fold one tensor's contribution into this group.

        Parameters
        ----------
        thetas : dict
            ``{checkpoint: flat_float64_array}`` for base/sft/dpo/rlvr.
        vecs : dict
            ``{stage: flat_float64_array}`` for sft/dpo/rlvr/post deltas.
        eps : float
            Threshold below which a coordinate is ignored for sign agreement.
        topk_cfg : dict | None
            If enabled, ``{"k_fraction", "min_k", "max_k"}`` controlling the
            per-tensor top-k overlap approximation.
        """
        n = vecs["sft"].size
        self.numel += float(n)
        self.tensor_count += 1

        for s in STAGES:
            v = vecs[s]
            self.norm_sq[s] += float(np.dot(v, v))
            self.l1[s] += float(np.abs(v).sum())
            m = float(np.abs(v).max()) if n else 0.0
            if m > self.linf[s]:
                self.linf[s] = m

        for c in CHECKPOINTS:
            t = thetas[c]
            self.theta_sq[c] += float(np.dot(t, t))
            self.abs_sum[c] += float(np.abs(t).sum())

        for a, b in STAGE_PAIRS:
            va, vb = vecs[a], vecs[b]
            self.dot[(a, b)] += float(np.dot(va, vb))

            mask = (np.abs(va) > eps) & (np.abs(vb) > eps)
            valid = int(mask.sum())
            if valid:
                agree = int((np.sign(va[mask]) == np.sign(vb[mask])).sum())
                self.same_sign[(a, b)] += agree
                self.valid_sign[(a, b)] += valid

        if topk_cfg and topk_cfg.get("enabled"):
            self._update_topk(vecs, topk_cfg)

    def _update_topk(self, vecs: Dict[str, np.ndarray], cfg: dict) -> None:
        n = vecs["sft"].size
        k = int(cfg.get("k_fraction", 0.001) * n)
        k = max(k, int(cfg.get("min_k", 0)))
        k = min(k, int(cfg.get("max_k", n)), n)
        if k <= 0:
            return
        # Indices of the top-k absolute coordinates per stage (unordered).
        top = {}
        for s in STAGES:
            a = np.abs(vecs[s])
            if k >= n:
                top[s] = set(range(n))
            else:
                idx = np.argpartition(a, n - k)[n - k:]
                top[s] = set(idx.tolist())
        for a, b in STAGE_PAIRS:
            inter = len(top[a] & top[b])
            self.topk_inter[(a, b)] += inter
            self.topk_total[(a, b)] += k

    def gram(self) -> np.ndarray:
        """Return the 3x3 Gram matrix of (sft, dpo, rlvr) stage vectors."""
        d = self.dot
        g = np.array(
            [
                [self.norm_sq["sft"], d[("sft", "dpo")], d[("sft", "rlvr")]],
                [d[("sft", "dpo")], self.norm_sq["dpo"], d[("dpo", "rlvr")]],
                [d[("sft", "rlvr")], d[("dpo", "rlvr")], self.norm_sq["rlvr"]],
            ],
            dtype=np.float64,
        )
        return g


class AccumulatorStore:
    """Holds one :class:`Accumulator` per ``(granularity, group)`` key."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], Accumulator] = {}

    def get(self, granularity: str, group: str) -> Accumulator:
        key = (granularity, group)
        acc = self._store.get(key)
        if acc is None:
            acc = Accumulator()
            self._store[key] = acc
        return acc

    def items(self) -> List[Tuple[Tuple[str, str], Accumulator]]:
        return sorted(self._store.items(), key=lambda kv: kv[0])

    def by_granularity(self, granularity: str) -> List[Tuple[str, Accumulator]]:
        out = [(g, a) for (gran, g), a in self._store.items() if gran == granularity]
        return sorted(out, key=lambda kv: kv[0])

    def __len__(self) -> int:
        return len(self._store)
