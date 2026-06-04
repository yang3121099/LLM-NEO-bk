"""Grouping rules for the static vector-field audit.

Given a HuggingFace parameter name (e.g. ``model.layers.12.self_attn.q_proj.weight``)
this module produces a list of ``(granularity, group)`` keys.  A single tensor can
belong to multiple groups; every accumulator/metric is computed per group.

This module is intentionally dependency-free (pure stdlib) so it can be unit
tested without torch / numpy.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

# Recognised submodule leaf names (longest / most specific first is not required
# because they are mutually exclusive substrings).
_SUBMODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "embed_tokens",
    "lm_head",
)

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")

# All granularities understood by this module.  ``analyze_lineage`` filters this
# down to whatever the config requests.
ALL_GRANULARITIES = (
    "global",
    "layer",
    "module",
    "layer_module",
    "submodule",
    "depth_bin",
)


def extract_layer_idx(name: str) -> int:
    """Return the transformer layer index encoded in ``name`` or ``-1``."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def classify_module(name: str) -> str:
    """Map a parameter name onto a coarse module bucket.

    Order matters: the ``norm`` test runs first so that ``post_attention_layernorm``
    (which contains the substring ``attention``) is correctly bucketed as a norm
    rather than attention. Attention / mlp weights never contain ``norm``.
    """
    if "norm" in name or "layernorm" in name:
        return "norm"
    if "self_attn" in name or "attention" in name:
        return "attn"
    if "mlp" in name:
        return "mlp"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def classify_submodule(name: str) -> str:
    """Return the leaf submodule name or ``"other"``."""
    for sub in _SUBMODULES:
        if sub in name:
            return sub
    return "other"


def classify_depth_bin(layer_idx: int, depth_bins: Dict[str, Sequence[int]]) -> str:
    """Return the depth-bin name for ``layer_idx`` (inclusive ranges) or ``""``."""
    if layer_idx < 0:
        return ""
    for bin_name, bounds in depth_bins.items():
        lo, hi = bounds[0], bounds[1]
        if lo <= layer_idx <= hi:
            return bin_name
    return ""


def assign_groups(
    name: str,
    granularities: Sequence[str],
    depth_bins: Dict[str, Sequence[int]] | None = None,
) -> List[Tuple[str, str]]:
    """Return ``(granularity, group)`` keys for a parameter ``name``.

    Only granularities present in ``granularities`` are emitted.  Groups that do
    not apply to a tensor (e.g. ``layer`` for a top-level ``lm_head``) are
    simply omitted.
    """
    depth_bins = depth_bins or {}
    layer_idx = extract_layer_idx(name)
    module = classify_module(name)
    groups: List[Tuple[str, str]] = []

    if "global" in granularities:
        groups.append(("global", "global:all"))

    if "layer" in granularities and layer_idx >= 0:
        groups.append(("layer", f"layer:{layer_idx}"))

    if "module" in granularities:
        groups.append(("module", f"module:{module}"))

    if "layer_module" in granularities and layer_idx >= 0:
        groups.append(("layer_module", f"layer_module:{layer_idx}:{module}"))

    if "submodule" in granularities:
        groups.append(("submodule", f"submodule:{classify_submodule(name)}"))

    if "depth_bin" in granularities:
        bin_name = classify_depth_bin(layer_idx, depth_bins)
        if bin_name:
            groups.append(("depth_bin", f"depth_bin:{bin_name}"))

    return groups
