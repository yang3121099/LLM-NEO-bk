#!/usr/bin/env python3
"""Tests for the streaming Shadow-FT merge.

Builds tiny sharded safetensors checkpoints that reproduce the awkward parts of
the real ones -- a bf16 instruct backbone against fp32 RL checkpoints, an
integer buffer, two shards with an index -- and checks the arithmetic, the
dtype policy, and the failure paths.

Run:  python shadow_rl/tests/test_merge.py
"""

import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE = os.path.join(os.path.dirname(HERE), "merge.py")

SHARDS = {
    "model-00001-of-00002.safetensors": ["model.layers.0.self_attn.q_proj.weight",
                                         "model.layers.0.mlp.down_proj.weight"],
    "model-00002-of-00002.safetensors": ["model.layers.1.self_attn.q_proj.weight",
                                         "model.norm.weight",
                                         "model.buffer.int_thing"],
}
FLOAT_KEYS = [k for ks in SHARDS.values() for k in ks if not k.endswith("int_thing")]
ALL_KEYS = [k for ks in SHARDS.values() for k in ks]


def write_ckpt(path, tensors, dtype, extra_files=True):
    os.makedirs(path, exist_ok=True)
    weight_map = {}
    for shard, keys in SHARDS.items():
        payload = {}
        for k in keys:
            t = tensors[k]
            payload[k] = t if not t.is_floating_point() else t.to(dtype)
            weight_map[k] = shard
        save_file(payload, os.path.join(path, shard), metadata={"format": "pt"})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, fh)
    if extra_files:
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump({"model_type": "qwen2", "marker": os.path.basename(path)}, fh)
        with open(os.path.join(path, "tokenizer_config.json"), "w") as fh:
            json.dump({"chat_template": "{{ marker }}"}, fh)


def random_tensors(seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in ALL_KEYS:
        if k.endswith("int_thing"):
            out[k] = torch.arange(8, dtype=torch.int64)
        elif k.endswith("norm.weight"):
            out[k] = torch.rand(16, generator=g)
        else:
            out[k] = torch.randn(16, 16, generator=g) * 0.02
    return out


def read_all(path):
    out = {}
    for shard in SHARDS:
        with safe_open(os.path.join(path, shard), framework="pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out


def run(args, expect_ok=True):
    proc = subprocess.run([sys.executable, MERGE] + args, capture_output=True, text=True)
    if expect_ok and proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise AssertionError("merge.py failed unexpectedly")
    return proc


def main():
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        base_t = random_tensors(0)
        # Instruct differs from base by a small amount, as a real pair does.
        inst_t = {k: (v + torch.randn(v.shape, generator=torch.Generator().manual_seed(1)) * 0.002
                      if v.is_floating_point() else v.clone())
                  for k, v in base_t.items()}
        # RL checkpoint differs from base by the learned update.
        delta_t = {k: (torch.randn(v.shape, generator=torch.Generator().manual_seed(2)) * 0.001
                       if v.is_floating_point() else torch.zeros_like(v))
                   for k, v in base_t.items()}
        rlb_t = {k: v + delta_t[k] for k, v in base_t.items()}

        p_base = os.path.join(tmp, "base")
        p_inst = os.path.join(tmp, "instruct")
        p_rlb = os.path.join(tmp, "rl_base")
        p_out = os.path.join(tmp, "shadow")

        # The realistic dtype mix: bf16 official weights, fp32 RL release.
        write_ckpt(p_base, base_t, torch.bfloat16)
        write_ckpt(p_inst, inst_t, torch.bfloat16)
        write_ckpt(p_rlb, rlb_t, torch.float32)

        print("\n[1] merge with bf16 backbone / fp32 RL checkpoint")
        proc = run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_rlb, "--out", p_out])

        merged = read_all(p_out)
        inst_disk = read_all(p_inst)
        base_disk = read_all(p_base)
        rlb_disk = read_all(p_rlb)

        check("all keys present", set(merged) == set(ALL_KEYS))
        check("output dtype follows W_I (bf16)",
              all(merged[k].dtype == torch.bfloat16 for k in FLOAT_KEYS))
        check("integer buffer copied verbatim",
              torch.equal(merged["model.buffer.int_thing"], base_t["model.buffer.int_thing"]))

        # Reference computed the way the script promises: fp32 throughout, cast at the end.
        worst = 0.0
        for k in FLOAT_KEYS:
            want = (inst_disk[k].float() + (rlb_disk[k].float() - base_disk[k].float())).to(torch.bfloat16)
            worst = max(worst, (want.float() - merged[k].float()).abs().max().item())
        check("matches fp32 reference arithmetic", worst == 0.0, f"max abs err {worst:g}")

        # The point of doing it in fp32: a bf16 subtraction visibly loses the delta.
        naive_err = 0.0
        for k in FLOAT_KEYS:
            naive = (inst_disk[k] + (rlb_disk[k].to(torch.bfloat16) - base_disk[k])).float()
            naive_err = max(naive_err, (naive - merged[k].float()).abs().max().item())
        check("fp32 path differs from naive bf16 path", naive_err > 0.0,
              f"max divergence {naive_err:g}")

        check("config.json taken from W_I",
              json.load(open(os.path.join(p_out, "config.json")))["marker"] == "instruct")
        check("tokenizer config copied", os.path.exists(os.path.join(p_out, "tokenizer_config.json")))
        check("index written", os.path.exists(os.path.join(p_out, "model.safetensors.index.json")))
        check("no stray weights copied from W_I",
              sorted(f for f in os.listdir(p_out) if f.endswith(".safetensors")) == sorted(SHARDS))

        stats = json.load(open(os.path.join(p_out, "shadow_merge_stats.json")))
        check("sigma reported", 0.0 < stats["sigma"] < 1.0, f"sigma={stats['sigma']:.5f}")
        check("relative delta reported", stats["rel_delta_base"] > 0.0,
              f"rel={stats['rel_delta_base']:.5f}")
        check("sigma printed to stdout", "sigma (base vs instruct)" in proc.stdout)

        print("\n[2] shadow == instruct when RL made no change")
        p_out2 = os.path.join(tmp, "shadow_noop")
        run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_base, "--out", p_out2])
        noop = read_all(p_out2)
        check("identity merge reproduces W_I",
              all(torch.equal(noop[k], inst_disk[k]) for k in ALL_KEYS))

        print("\n[3] --scale scales the update")
        p_out3 = os.path.join(tmp, "shadow_half")
        run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_rlb, "--out", p_out3,
             "--scale", "0.5"])
        half = read_all(p_out3)
        k0 = FLOAT_KEYS[0]
        want_half = (inst_disk[k0].float() + 0.5 * (rlb_disk[k0].float() - base_disk[k0].float())).to(torch.bfloat16)
        check("scale=0.5 applied", torch.equal(half[k0], want_half))

        print("\n[4] mismatched key set fails with a diff")
        p_bad = os.path.join(tmp, "bad_keys")
        write_ckpt(p_bad, rlb_t, torch.float32)
        with safe_open(os.path.join(p_bad, "model-00002-of-00002.safetensors"), framework="pt") as fh:
            payload = {k: fh.get_tensor(k) for k in fh.keys()}
        payload["lm_head.weight"] = torch.randn(16, 16)
        save_file(payload, os.path.join(p_bad, "model-00002-of-00002.safetensors"),
                  metadata={"format": "pt"})
        idx_path = os.path.join(p_bad, "model.safetensors.index.json")
        idx = json.load(open(idx_path))
        idx["weight_map"]["lm_head.weight"] = "model-00002-of-00002.safetensors"
        json.dump(idx, open(idx_path, "w"))

        proc = run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_bad,
                    "--out", os.path.join(tmp, "never")], expect_ok=False)
        out = proc.stdout + proc.stderr
        check("key mismatch is fatal", proc.returncode != 0)
        check("diff names the offending key", "lm_head.weight" in out)
        check("nothing written on failure", not os.path.exists(os.path.join(tmp, "never")))

        print("\n[5] --ignore-keys recovers the benign case")
        p_out5 = os.path.join(tmp, "shadow_ignored")
        run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_bad, "--out", p_out5,
             "--ignore-keys", "lm_head"])
        check("merge succeeds ignoring lm_head", os.path.exists(os.path.join(p_out5, "config.json")))

        print("\n[6] mismatched shape fails with a diff")
        p_shape = os.path.join(tmp, "bad_shape")
        wrong = {k: (torch.randn(8, 8) if k == FLOAT_KEYS[0] else v) for k, v in rlb_t.items()}
        write_ckpt(p_shape, wrong, torch.float32)
        proc = run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_shape,
                    "--out", os.path.join(tmp, "never2")], expect_ok=False)
        out = proc.stdout + proc.stderr
        check("shape mismatch is fatal", proc.returncode != 0)
        check("diff names the offending shape", FLOAT_KEYS[0] in out and "shapes do not match" in out)

        print("\n[7] --stats-only writes nothing")
        p_never3 = os.path.join(tmp, "never3")
        proc = run(["--base", p_base, "--instruct", p_inst, "--rl-base", p_rlb,
                    "--out", p_never3, "--stats-only", "--rl-instruct", p_rlb])
        check("stats-only produced no model", not os.path.exists(p_never3))
        check("stats-only reported both sides", "||RL(W_I)-W_I||" in proc.stdout)

        print("\n[8] out-of-range sigma is flagged")
        p_far = os.path.join(tmp, "far_instruct")
        far_t = {k: (v * 3.0 if v.is_floating_point() else v.clone()) for k, v in base_t.items()}
        write_ckpt(p_far, far_t, torch.bfloat16)
        proc = run(["--base", p_base, "--instruct", p_far, "--rl-base", p_rlb,
                    "--out", os.path.join(tmp, "shadow_far"), "--stats-only"])
        check("warns on sigma outside paper range", "WARNING" in proc.stdout)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
