#!/usr/bin/env python3
"""Assert that train_grpo.sh's hyperparameters actually reach verl.

Hydra accepts any well-formed key. A setting can be spelled correctly, placed on
a key that exists, be printed back in the banner -- and still be read by nothing,
because verl moved it to a different group two releases ago. That is exactly what
happened to `custom_reward_function.path`: it survives as a compatibility stub
that main_ppo never consults, so training silently scores with the default reward
manager instead of math_reward.py. No error, just a wrong experiment.

So the check has to be against the *composed* config, not the command line. This
runs `DRY_RUN=1 train_grpo.sh base`, which asks hydra to print the config it
would train with, and looks up each value by its real path.

Requires verl to be installed (./shadow_rl/verl_math/setup_verl.sh); skips
otherwise.

Run:  python shadow_rl/verl_math/tests/test_train_config.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # shadow_rl/verl_math/tests
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TRAIN_SH = os.path.join(REPO, "shadow_rl", "verl_math", "train_grpo.sh")
assert os.path.exists(TRAIN_SH), TRAIN_SH

# The recipe as specified, plus the low-GPU offload run_all_math.sh adds. Values
# are what the *composed* config must hold, keyed by the path verl reads them
# from -- which is not always the path the override is written on.
EXPECTED = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.use_kl_in_reward": False,
    "data.train_batch_size": 64,
    "data.max_prompt_length": 1024,
    "data.max_response_length": 7168,
    "actor_rollout_ref.actor.optim.lr": 1e-6,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.kl_loss_coef": 0.0,
    "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.n": 8,
    "actor_rollout_ref.rollout.temperature": 1.0,
    "actor_rollout_ref.rollout.top_p": 1.0,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "trainer.nnodes": 1,
    "trainer.total_epochs": 1,
    "trainer.n_gpus_per_node": 1,
}

# Passed as extra overrides, the way run_all_math.sh does it on 1-2 GPUs.
OFFLOAD_ARGS = [
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
]
OFFLOAD_EXPECTED = {
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
    "actor_rollout_ref.ref.fsdp_config.param_offload": True,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
}

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def compose(extra_args, data_dir, ckpt_dir):
    env = dict(os.environ, DRY_RUN="1", N_GPUS="1",
               DATA_DIR=data_dir, CKPT_DIR=ckpt_dir)
    proc = subprocess.run(["bash", TRAIN_SH, "base", *extra_args],
                          capture_output=True, text=True, env=env, cwd=REPO)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError("DRY_RUN failed:\n  " + "\n  ".join(tail))
    out = proc.stdout
    # train_grpo.sh prints a banner before handing over to hydra.
    marker = out.find("actor_rollout_ref:")
    if marker < 0:
        raise RuntimeError("no composed config in output")
    from omegaconf import OmegaConf

    return OmegaConf.create(out[marker:])


def main():
    try:
        import verl  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: verl is not installed ({exc}).")
        print("      ./shadow_rl/verl_math/setup_verl.sh")
        return 0

    tmp = tempfile.mkdtemp(prefix="shadow-train-cfg-")
    try:
        data = os.path.join(tmp, "datasets")
        os.makedirs(os.path.join(data, "DAPO-Math-17k-Processed"))
        os.makedirs(os.path.join(data, "math_val"))
        # train_grpo.sh only checks these exist; hydra never opens them.
        open(os.path.join(data, "DAPO-Math-17k-Processed", "DAPO-Math.parquet"), "w").close()
        open(os.path.join(data, "math_val", "val.parquet"), "w").close()

        from omegaconf import OmegaConf

        print("\n[1] the specified hyperparameters reach verl")
        cfg = compose([], data, os.path.join(tmp, "ckpt"))
        for path, want in EXPECTED.items():
            got = OmegaConf.select(cfg, path)
            check(path, got == want, f"= {got!r}" + ("" if got == want else f" want {want!r}"))

        print("\n[2] the custom reward function is on the path verl reads")
        # The whole point: not "an override named custom_reward_function was
        # accepted", but "the reward loader will find math_reward.py".
        for path in ("reward.custom_reward_function.path", "custom_reward_function.path"):
            got = OmegaConf.select(cfg, path)
            if got:
                break
        check("points at math_reward.py",
              isinstance(got, str) and got.endswith("shadow_rl/verl_math/math_reward.py"),
              f"= {got!r}")
        name = (OmegaConf.select(cfg, "reward.custom_reward_function.name")
                or OmegaConf.select(cfg, "custom_reward_function.name"))
        check("function name is compute_score", name == "compute_score", f"= {name!r}")
        enable = OmegaConf.select(cfg, "reward.reward_model.enable")
        if enable is None:
            enable = OmegaConf.select(cfg, "reward_model.enable")
        check("no reward model (rule-based only)", enable is False, f"= {enable!r}")

        print("\n[3] logging does not require a wandb login")
        logger = list(OmegaConf.select(cfg, "trainer.logger") or [])
        check("wandb is off by default", "wandb" not in logger, f"= {logger}")
        check("console is on", "console" in logger, f"= {logger}")

        print("\n[4] the low-GPU offload overrides land")
        cfg2 = compose(OFFLOAD_ARGS, data, os.path.join(tmp, "ckpt2"))
        for path, want in OFFLOAD_EXPECTED.items():
            got = OmegaConf.select(cfg2, path)
            check(path, got == want, f"= {got!r}")

        print("\n[5] the demo overrides land")
        cfg3 = compose(["trainer.total_training_steps=8", "trainer.save_freq=4",
                        "trainer.test_freq=4", "data.max_response_length=1024",
                        "actor_rollout_ref.rollout.n=4"],
                       data, os.path.join(tmp, "ckpt3"))
        for path, want in (("trainer.total_training_steps", 8),
                           ("trainer.save_freq", 4),
                           ("data.max_response_length", 1024),
                           ("actor_rollout_ref.rollout.n", 4)):
            got = OmegaConf.select(cfg3, path)
            check(path, got == want, f"= {got!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
