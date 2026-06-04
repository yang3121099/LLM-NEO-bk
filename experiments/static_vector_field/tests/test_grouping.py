"""Unit tests for grouping rules (pure stdlib, no torch/numpy needed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grouping import (  # noqa: E402
    assign_groups,
    classify_depth_bin,
    classify_module,
    classify_submodule,
    extract_layer_idx,
)

ALL = ["global", "layer", "module", "layer_module", "submodule", "depth_bin"]
DEPTH = {"shallow": [0, 10], "middle": [11, 20], "deep": [21, 31]}


def test_layer_index_extraction():
    assert extract_layer_idx("model.layers.12.self_attn.q_proj.weight") == 12
    assert extract_layer_idx("model.layers.0.mlp.down_proj.weight") == 0
    assert extract_layer_idx("model.embed_tokens.weight") == -1
    assert extract_layer_idx("lm_head.weight") == -1


def test_module_classification():
    assert classify_module("model.layers.3.self_attn.q_proj.weight") == "attn"
    assert classify_module("model.layers.3.mlp.up_proj.weight") == "mlp"
    assert classify_module("model.layers.3.input_layernorm.weight") == "norm"
    assert classify_module("model.layers.3.post_attention_layernorm.weight") == "norm"
    assert classify_module("model.embed_tokens.weight") == "embed"
    assert classify_module("lm_head.weight") == "lm_head"
    assert classify_module("model.norm.weight") == "norm"


def test_submodule_classification():
    assert classify_submodule("model.layers.3.self_attn.k_proj.weight") == "k_proj"
    assert classify_submodule("model.layers.3.mlp.gate_proj.weight") == "gate_proj"
    assert classify_submodule("model.layers.3.post_attention_layernorm.weight") == \
        "post_attention_layernorm"
    assert classify_submodule("model.embed_tokens.weight") == "embed_tokens"


def test_depth_bins():
    assert classify_depth_bin(0, DEPTH) == "shallow"
    assert classify_depth_bin(10, DEPTH) == "shallow"
    assert classify_depth_bin(11, DEPTH) == "middle"
    assert classify_depth_bin(25, DEPTH) == "deep"
    assert classify_depth_bin(-1, DEPTH) == ""
    assert classify_depth_bin(99, DEPTH) == ""


def test_assign_groups_attn_weight():
    groups = dict(assign_groups(
        "model.layers.12.self_attn.q_proj.weight", ALL, DEPTH))
    assert groups["global"] == "global:all"
    assert groups["layer"] == "layer:12"
    assert groups["module"] == "module:attn"
    assert groups["layer_module"] == "layer_module:12:attn"
    assert groups["submodule"] == "submodule:q_proj"
    assert groups["depth_bin"] == "depth_bin:middle"


def test_assign_groups_embedding_has_no_layer():
    keys = [g for g, _ in assign_groups("model.embed_tokens.weight", ALL, DEPTH)]
    assert "layer" not in keys
    assert "layer_module" not in keys
    assert "depth_bin" not in keys  # no layer -> no depth bin
    groups = dict(assign_groups("model.embed_tokens.weight", ALL, DEPTH))
    assert groups["module"] == "module:embed"
    assert groups["submodule"] == "submodule:embed_tokens"


def test_granularity_filtering():
    groups = dict(assign_groups(
        "model.layers.5.mlp.down_proj.weight", ["global", "module"], DEPTH))
    assert set(groups.keys()) == {"global", "module"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All grouping tests passed.")
