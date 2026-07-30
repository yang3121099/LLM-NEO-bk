# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GPU generation profile.

scripts/gpu_profile.sh decides which torch, which attention kernel and which
batch shape every setup and generator script uses. Getting it wrong is
expensive in both directions: an H100 stack on a B300 fails at the first matmul,
and a B300 stack on an H100 silently reinstalls torch. These pin the mapping,
and pin the H100 row specifically, because that one has to keep matching the
runs already published.

Run:  pytest tests/env/test_gpu_profile.py
"""

import json
import os
import shutil
import subprocess
import sys

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILE = os.path.join(ROOT, "scripts", "gpu_profile.sh")
CHECK_ENV = os.path.join(ROOT, "shadow_rl")


def profile(cc=None, name=None, mem_gb=None, count=None, **extra):
    """Run gpu_profile.sh for a pretended device and return its JSON."""
    env = dict(os.environ)
    # Whatever the ambient shell exports must not leak in -- but an override
    # passed by the test itself has to survive, so clear first, then apply.
    for key in ("TORCH_INDEX", "TORCH_SPEC", "TORCH_CUDA_ARCH_LIST", "ATTN_IMPL", "VLLM_SPEC"):
        env.pop(key, None)
    env.update({k: str(v) for k, v in extra.items()})
    if cc is not None:
        env["FORCE_GPU_CC"] = cc
    if name is not None:
        env["FORCE_GPU_NAME"] = name
    if mem_gb is not None:
        env["FORCE_GPU_MEM_GB"] = str(mem_gb)
    if count is not None:
        env["FORCE_GPU_COUNT"] = str(count)
    out = subprocess.run(["bash", PROFILE, "--json"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


class TestGeneration:
    r"""Compute capability -> family."""

    @pytest.mark.parametrize(
        ("cc", "family", "sm"),
        [
            ("10.3", "blackwell-ultra", "103"),
            ("10.0", "blackwell", "100"),
            ("12.0", "blackwell-rtx", "120"),
            ("9.0", "hopper", "90"),
            ("8.0", "ampere", "80"),
        ],
    )
    def test_family(self, cc, family, sm):
        got = profile(cc=cc)
        assert got["family"] == family
        assert got["sm"] == sm

    @pytest.mark.parametrize(
        ("name", "sm"),
        [
            ("NVIDIA B300", "103"),
            ("NVIDIA GB300 NVL72", "103"),
            ("NVIDIA B200", "100"),
            ("NVIDIA H100 80GB HBM3", "90"),
            ("NVIDIA H200", "90"),
            ("NVIDIA A100-SXM4-80GB", "80"),
        ],
    )
    def test_family_from_name(self, name, sm):
        """Drivers older than 495 have no compute_cap query; fall back to the name."""
        assert profile(name=name)["sm"] == sm

    def test_unknown_hardware_keeps_the_legacy_stack(self):
        """An unrecognised GPU must not get an experimental stack silently."""
        got = profile(name="Some Future GPU")
        assert got["family"] == "unknown"
        assert got["torch_spec"] == "torch==2.6.0"


class TestStack:
    def test_h100_is_the_stack_the_results_were_produced_with(self):
        got = profile(cc="9.0", mem_gb=79)
        assert got["torch_spec"] == "torch==2.6.0"
        assert got["torch_index"].endswith("/cu126")
        assert got["arch_list"] == "9.0"
        assert got["attn_impl"] == "fa2"
        assert got["micro_bs"] == 1

    def test_b300_needs_a_torch_that_has_sm_103(self):
        got = profile(cc="10.3", mem_gb=279)
        # No released torch below 2.9 has sm_103 kernels, and the cu126/cu128
        # wheels predate sm_103 support in the CUDA toolkit.
        assert got["torch_spec"] == "torch>=2.9.0"
        assert got["torch_index"].endswith("/cu130")
        # Arch-conditional, and arch-exact: sm_100a would not run here.
        assert got["arch_list"] == "10.3a"

    def test_b200_and_b300_do_not_share_an_arch_list(self):
        """The trap: both are Blackwell, neither one's `a` kernels run on the other."""
        assert profile(cc="10.0")["arch_list"] != profile(cc="10.3")["arch_list"]

    def test_blackwell_gets_a_vllm_floor_and_hopper_does_not(self):
        assert profile(cc="10.3")["vllm_spec"].startswith("vllm>=")
        assert profile(cc="9.0")["vllm_spec"] == "vllm"

    @pytest.mark.parametrize(
        ("cc", "mem_gb", "micro_bs"),
        [
            ("10.3", 279, 4),  # B300
            ("10.0", 180, 2),  # B200 180GB
            ("10.0", 96, 1),  # a small Blackwell: no headroom, no change
            ("9.0", 141, 1),  # H200: not Blackwell, left alone deliberately
            ("9.0", 79, 1),  # H100
        ],
    )
    def test_micro_batch_hint(self, cc, mem_gb, micro_bs):
        assert profile(cc=cc, mem_gb=mem_gb)["micro_bs"] == micro_bs

    def test_micro_batch_hint_is_conservative_without_a_memory_reading(self):
        assert profile(cc="10.3")["micro_bs"] == 1

    def test_caller_overrides_win(self):
        got = profile(cc="10.3", TORCH_INDEX="https://example.invalid/cu129", ATTN_IMPL="fa2")
        assert got["torch_index"] == "https://example.invalid/cu129"
        assert got["attn_impl"] == "fa2"


class TestSourcingIsSafe:
    """A probe that cannot answer must leave the field empty, not exit.

    Every caller sources this under `set -euo pipefail`. On a node without
    nvidia-smi the query pipeline exited 127 and took run.sh with it.
    """

    # Enough of a PATH to run the script, and deliberately no nvidia-smi, no
    # nvcc and no python3 -- the state of a CPU login node.
    TOOLS = ("bash", "head", "sed", "tr", "grep", "cat", "env")

    def source_under_strict_mode(self, tmp_path, **force):
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        for tool in self.TOOLS:
            real = shutil.which(tool)
            if real and not (bindir / tool).exists():
                os.symlink(real, bindir / tool)

        script = (
            "set -euo pipefail\n"
            f"source {PROFILE!r}\n"
            'echo "$GPU_FAMILY $TORCH_SPEC $ATTN_IMPL $GPU_COUNT $MICRO_BS_HINT"\n'
        )
        env = dict(os.environ, **{k: str(v) for k, v in force.items()})
        env["PATH"] = str(bindir)
        return subprocess.run([str(bindir / "bash"), "-c", script], capture_output=True, text=True, env=env)

    def test_no_nvidia_smi_is_not_fatal(self, tmp_path):
        out = self.source_under_strict_mode(tmp_path)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["unknown", "torch==2.6.0", "fa2", "0", "1"]

    def test_forced_capability_still_works_without_a_driver(self, tmp_path):
        out = self.source_under_strict_mode(tmp_path, FORCE_GPU_CC="10.3", FORCE_GPU_MEM_GB="279")
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["blackwell-ultra", "torch>=2.9.0", "sdpa", "0", "4"]


class TestAgreesWithCheckEnv:
    """The bash table and the python one must not drift.

    shadow_rl/check_env.py keeps its own copy so it stays runnable standalone,
    and it is what prints the reinstall command the setup scripts then run.
    """

    @staticmethod
    def check_env_module():
        sys.path.insert(0, CHECK_ENV)
        import check_env

        return check_env

    @pytest.mark.parametrize("cc", ["10.3", "10.0", "12.0", "9.0", "8.0"])
    def test_same_wheel_index(self, cc):
        ce = self.check_env_module()
        sm = int(cc.replace(".", ""))
        index, _ = ce.stack_for_sm(sm)
        assert index == profile(cc=cc)["torch_index"]

    @pytest.mark.parametrize("cc", ["10.3", "10.0", "9.0"])
    def test_same_minimum_torch(self, cc):
        ce = self.check_env_module()
        sm = int(cc.replace(".", ""))
        _, min_torch = ce.stack_for_sm(sm)
        # bash carries the whole requirement string ("torch>=2.9.0"), python just
        # the version it floors at.
        assert min_torch in profile(cc=cc)["torch_spec"]


class TestArchSupport:
    """Which cubins can run on which device.

    This is the rule behind every "no kernel image is available for execution on
    the device" in this pipeline.
    """

    @staticmethod
    def arch_support(*args):
        sys.path.insert(0, CHECK_ENV)
        import check_env

        return check_env.arch_support(*args)

    def test_exact_cubin(self):
        assert self.arch_support(["sm_90", "sm_103"], 103) == "cubin"

    def test_arch_conditional_for_this_arch(self):
        assert self.arch_support(["sm_103a"], 103) == "cubin"

    def test_arch_conditional_for_a_sibling_does_not_count(self):
        """A wheel built for a B200 (sm_100a) has nothing a B300 can run."""
        assert self.arch_support(["sm_100a"], 103) is None

    def test_lower_minor_of_the_same_major_is_binary_compatible(self):
        assert self.arch_support(["sm_100"], 103) == "compat"

    def test_higher_minor_does_not_run_backwards(self):
        assert self.arch_support(["sm_103"], 100) is None

    def test_other_major_never_counts(self):
        assert self.arch_support(["sm_90", "sm_89"], 103) is None
        assert self.arch_support(["sm_120"], 103) is None

    def test_ptx_can_jit(self):
        assert self.arch_support(["compute_100"], 103) == "ptx"
        assert self.arch_support(["compute_103"], 103) == "ptx"

    def test_a_real_cubin_beats_ptx(self):
        assert self.arch_support(["compute_100", "sm_103"], 103) == "cubin"

    def test_a_hopper_only_torch_on_a_b300_is_nothing(self):
        """The actual H100 -> B300 failure: torch 2.6.0's arch list."""
        hopper_wheel = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
        assert self.arch_support(hopper_wheel, 103) is None
        assert self.arch_support(hopper_wheel, 90) == "cubin"

    def test_empty_and_junk_are_not_support(self):
        assert self.arch_support([], 103) is None
        assert self.arch_support(None, 103) is None
        assert self.arch_support(["not_an_arch", "sm_", "compute_x"], 103) is None


class TestGeneratedScripts:
    """Check what the generators actually write into scripts/.

    The attention kernel and the batch shape are baked in at generation time,
    and they are the two lines that fail on the wrong machine.
    """

    @staticmethod
    def generate(**force):
        env = dict(os.environ, **{k: str(v) for k, v in force.items()})
        env.pop("FLASH_ATTN", None)
        before = set(os.listdir(os.path.join(ROOT, "scripts")))
        out = subprocess.run(["bash", os.path.join(ROOT, "run.sh")], capture_output=True, text=True, env=env, cwd=ROOT)
        after = set(os.listdir(os.path.join(ROOT, "scripts")))
        new = sorted(after - before)
        try:
            assert out.returncode == 0, out.stdout + out.stderr
            assert new, "run.sh generated nothing"
            with open(os.path.join(ROOT, "scripts", new[0]), encoding="utf-8") as fh:
                return fh.read()
        finally:
            for path in new:
                os.remove(os.path.join(ROOT, "scripts", path))
            subprocess.run(["git", "checkout", "--", "opencompass/eval_generated.py"], cwd=ROOT, capture_output=True)

    def test_b300_scripts_do_not_ask_for_fa2(self):
        script = self.generate(FORCE_GPU_CC="10.3", FORCE_GPU_MEM_GB=279, FORCE_GPU_COUNT=8)
        assert "--flash_attn fa2" not in script
        assert "--flash_attn sdpa" in script

    def test_b300_scripts_keep_the_h100_effective_batch(self):
        script = self.generate(FORCE_GPU_CC="10.3", FORCE_GPU_MEM_GB=279, FORCE_GPU_COUNT=8)
        assert "--per_device_train_batch_size 4" in script
        assert "--gradient_accumulation_steps 64" in script  # 4 x 64 == 1 x 256

    def test_h100_scripts_are_what_they_always_were(self):
        script = self.generate(FORCE_GPU_CC="9.0", FORCE_GPU_MEM_GB=79, FORCE_GPU_COUNT=8)
        assert "--flash_attn fa2" in script
        assert "--per_device_train_batch_size 1" in script
        assert "--gradient_accumulation_steps 256" in script

    def test_the_script_records_the_hardware_it_was_generated_for(self):
        """Record the hardware, so a script run on the other machine is visible."""
        script = self.generate(FORCE_GPU_CC="10.3", FORCE_GPU_MEM_GB=279, FORCE_GPU_COUNT=8)
        assert "sm_103" in script
        assert "# Attention : sdpa" in script
