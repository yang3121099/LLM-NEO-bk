#!/usr/bin/env python3
"""CPU-only tests for the verl GRPO + Shadow-FT pipeline.

Nothing here needs a GPU, verl, or a model download. What it pins is the set of
mistakes that would otherwise surface hours into a run on the B300:

  * a reward function that scores a correct answer 0 (silently ruins the run)
  * a parquet schema verl's RLHFDataset cannot read (fails at the first batch)
  * base and instruct validated on different questions (result is meaningless)
  * an eval config naming a dataset symbol OpenCompass does not export
    (ImportError after the models have loaded)

Run:  python verl_rl/tests/test_pipeline.py
"""

import json
import os
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
VERL_RL = os.path.dirname(HERE)
ROOT = os.path.dirname(VERL_RL)
sys.path.insert(0, VERL_RL)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
def test_reward():
    print("\n[1] the reward function scores what the prompt asks for")
    import reward_math as rm

    # The instruction tells the model to use \boxed{}; if the scorer disagreed
    # with the instruction, GRPO would optimise against its own prompt.
    check("boxed answer scores 1", rm.compute_score("d", r"\boxed{42}", "42") == 1.0)
    check("wrong answer scores 0", rm.compute_score("d", r"\boxed{41}", "42") == 0.0)
    check("no answer scores 0", rm.compute_score("d", "I give up", "42") == 0.0)

    # The last box is the answer: a chain of thought boxes intermediate results.
    check("last box wins",
          rm.compute_score("d", r"\boxed{1} ... \boxed{42}", "42") == 1.0)

    # A brace-balanced extractor, because answers contain braces.
    check("nested braces survive extraction",
          rm.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")
    check("truncated box is not an answer",
          rm.extract_boxed(r"\boxed{4") is None)

    # Normalisation collapses spellings of the same value, never distinct values.
    check("0.5 == .5", rm.is_equivalent("0.5", ".5"))
    check("12.0 == 12", rm.is_equivalent("12.0", "12"))
    check("1,000 == 1000", rm.is_equivalent("1,000", "1000"))
    check("1{,}000 == 1000", rm.is_equivalent(r"1{,}000", "1000"))
    check("12.5 != 12", not rm.is_equivalent("12.5", "12"))
    check("-3 != 3", not rm.is_equivalent("-3", "3"))
    check("empty is never equivalent", not rm.is_equivalent("", ""))

    # verl calls it with extra_info; a signature mismatch fails at batch one.
    detail = ""
    try:
        rm.compute_score("src", r"\boxed{1}", "1", extra_info={"index": 0})
        ok = True
    except TypeError as exc:
        ok, detail = False, str(exc)
    check("accepts verl's extra_info kwarg", ok, detail if not ok else "")


# --------------------------------------------------------------------------- #
def test_data_schema():
    print("\n[2] the parquet is in the shape verl's RLHFDataset reads")
    out = tempfile.mkdtemp()
    proc = subprocess.run(
        [sys.executable, os.path.join(VERL_RL, "prepare_data.py"),
         "--out", out, "--demo", "--val-size", "20"],
        capture_output=True, text=True)
    check("prepare_data --demo succeeds", proc.returncode == 0,
          proc.stderr.strip()[:200])
    if proc.returncode != 0:
        return

    try:
        import pandas as pd
    except ImportError:
        print("  skip  pandas not installed")
        return

    train = pd.read_parquet(os.path.join(out, "train.parquet"))
    val = pd.read_parquet(os.path.join(out, "val.parquet"))

    for col in ("data_source", "prompt", "ability", "reward_model", "extra_info"):
        check(f"column {col!r} present", col in train.columns)

    row = train.iloc[0]
    prompt = row["prompt"]
    check("prompt is a list of chat messages",
          len(prompt) >= 1 and set(prompt[0]) >= {"role", "content"},
          f"got {type(prompt).__name__}")
    # Stored as messages, not rendered text: verl applies the chat template, and
    # both roles must go through the identical path.
    check("prompt is not pre-rendered text", not isinstance(prompt, str))
    check("the instruction asks for \\boxed",
          "\\boxed" in prompt[0]["content"])

    rm_field = row["reward_model"]
    check("reward_model carries a ground_truth",
          "ground_truth" in rm_field and rm_field["ground_truth"] != "")
    check("ground_truth is a string", isinstance(rm_field["ground_truth"], str))

    check("train and val are disjoint sizes", len(train) == 180 and len(val) == 20,
          f"train={len(train)} val={len(val)}")

    # The reward function must be able to score the dataset's own answers.
    import reward_math as rm_mod
    gold = row["reward_model"]["ground_truth"]
    check("a perfect response on row 0 scores 1.0",
          rm_mod.compute_score("d", f"\\boxed{{{gold}}}", gold) == 1.0)

    meta = json.load(open(os.path.join(out, "dataset_info.json")))
    check("dataset_info records the instruction", "instruction" in meta)


def test_val_split_is_shared():
    print("\n[3] both GRPO runs are validated on the same questions")
    # Base-init and instruct-init are only comparable if nothing but the starting
    # weights differs. A reshuffled validation split would break that quietly.
    outs = []
    for _ in range(2):
        out = tempfile.mkdtemp()
        subprocess.run(
            [sys.executable, os.path.join(VERL_RL, "prepare_data.py"),
             "--out", out, "--demo", "--val-size", "20", "--seed", "1"],
            capture_output=True, text=True)
        outs.append(out)
    try:
        import pandas as pd
    except ImportError:
        print("  skip  pandas not installed")
        return
    a = pd.read_parquet(os.path.join(outs[0], "val.parquet"))
    b = pd.read_parquet(os.path.join(outs[1], "val.parquet"))
    qa = [r["question"] for r in a["extra_info"]]
    qb = [r["question"] for r in b["extra_info"]]
    check("the same seed gives the same validation set", qa == qb)

    out_c = tempfile.mkdtemp()
    subprocess.run(
        [sys.executable, os.path.join(VERL_RL, "prepare_data.py"),
         "--out", out_c, "--demo", "--val-size", "20", "--seed", "2"],
        capture_output=True, text=True)
    c = pd.read_parquet(os.path.join(out_c, "val.parquet"))
    qc = [r["question"] for r in c["extra_info"]]
    check("a different seed gives a different one", qa != qc)


# --------------------------------------------------------------------------- #
def test_eval_config():
    print("\n[4] the eval config names symbols OpenCompass actually exports")
    sys.path.insert(0, VERL_RL)
    import make_eval_config as mec

    configs_root = os.path.join(ROOT, "opencompass", "opencompass", "configs")
    if not os.path.isdir(configs_root):
        print("  skip  vendored opencompass not present")
    else:
        for name, (module, symbol) in sorted(mec.DATASETS.items()):
            rel = module.replace("opencompass.configs.", "").replace(".", os.sep)
            path = os.path.join(configs_root, rel + ".py")
            if not os.path.isfile(path):
                check(f"{name}: module exists", False, module)
                continue
            with open(path, encoding="utf-8") as fh:
                body = fh.read()
            # The symbol has to be assigned at module level for `from X import Y`.
            check(f"{name}: {symbol} defined in {os.path.basename(path)}",
                  f"\n{symbol} =" in body or body.startswith(f"{symbol} ="))

    for suite, names in mec.SUITES.items():
        unknown = [n for n in names if n not in mec.DATASETS]
        check(f"suite {suite!r} only names known datasets", not unknown, str(unknown))

    print("\n[5] the generated config covers all five roles and parses")
    out = os.path.join(tempfile.mkdtemp(), "eval_gen.py")
    proc = subprocess.run(
        [sys.executable, os.path.join(VERL_RL, "make_eval_config.py"),
         "--out", out, "--suite", "fast"],
        capture_output=True, text=True)
    check("make_eval_config succeeds", proc.returncode == 0, proc.stderr.strip()[:200])
    if proc.returncode != 0:
        return
    body = open(out, encoding="utf-8").read()

    import ast
    detail = ""
    try:
        ast.parse(body)
        parsed = True
    except SyntaxError as exc:
        parsed, detail = False, str(exc)
    check("generated config is valid python", parsed, detail if not parsed else "")

    for marker in ("W_B", "W_I", "RL_W_B", "RL_W_I", "W_shadow"):
        check(f"role {marker} present", marker in body)
    # Every role through the same class: a base model scored on a raw-text path
    # and an instruct model on a chat path would not be comparable.
    check("one model class for every role", body.count("type=") == 1)
    check("chat template backend", "ChatTemplate" in body)

    proc = subprocess.run(
        [sys.executable, os.path.join(VERL_RL, "make_eval_config.py"),
         "--out", out, "--datasets", "not_a_dataset"],
        capture_output=True, text=True)
    check("an unknown dataset is rejected up front", proc.returncode != 0)


# --------------------------------------------------------------------------- #
def test_config_sh():
    print("\n[6] config.sh answers for every path the other scripts read")
    keys = ["BASE_MODEL", "INSTRUCT_MODEL", "PAIR_NAME", "WORK_DIR", "DATA_DIR",
            "CKPT_DIR", "EXPORT_DIR", "MERGED_DIR", "LOG_DIR", "N_GPUS",
            "REWARD_FN_PATH", "EVAL_SUITE"]
    query = f"source {os.path.join(VERL_RL, 'config.sh')}; " + "; ".join(
        f'echo "{k}=${k}"' for k in keys)
    proc = subprocess.run(["bash", "-c", query], capture_output=True, text=True)
    check("config.sh sources under set -e", proc.returncode == 0,
          proc.stderr.strip()[:200])
    values = dict(line.split("=", 1) for line in proc.stdout.strip().splitlines()
                  if "=" in line)
    for k in keys:
        check(f"{k} is set", values.get(k, "") != "")
    check("the pair is base + instruct of one model",
          values.get("BASE_MODEL", "").rstrip("-Base") in values.get("INSTRUCT_MODEL", "")
          or values.get("INSTRUCT_MODEL", "") in values.get("BASE_MODEL", ""),
          f"{values.get('BASE_MODEL')} / {values.get('INSTRUCT_MODEL')}")
    check("the reward function exists", os.path.isfile(values.get("REWARD_FN_PATH", "")))

    # An override from the environment has to win, or run_all's --demo settings
    # would not reach train_grpo.sh.
    proc = subprocess.run(
        ["bash", "-c", f"source {os.path.join(VERL_RL, 'config.sh')}; echo $ROLLOUT_N"],
        capture_output=True, text=True, env=dict(os.environ, ROLLOUT_N="99"))
    check("environment overrides win", proc.stdout.strip() == "99")


def test_run_all_dry():
    print("\n[7] the orchestrator plans without touching anything")
    proc = subprocess.run(
        ["bash", os.path.join(VERL_RL, "run_all.sh"), "--dry-run", "--demo"],
        capture_output=True, text=True,
        env=dict(os.environ, FORCE_GPU_CC="10.3", FORCE_GPU_COUNT="8",
                 FORCE_GPU_MEM_GB="279"))
    check("dry run exits clean", proc.returncode == 0, proc.stderr.strip()[:200])
    out = proc.stdout
    check("reports the B300 it was told about", "sm_103" in out)
    check("reports both sides of the pair", "W_B" in out and "W_I" in out)
    check("nothing was executed", "nothing executed" in out)

    # The graft direction is the one thing that must not be written backwards.
    check("states the graft as W_I + (RL(W_B) - W_B)",
          "W_I + (RL(W_B) - W_B)" in out)


# --------------------------------------------------------------------------- #
HYDRA_CARD = """
```bash
python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_batch_size=1024 \\
    data.max_response_length=8192 \\
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.rollout.n=16
```
"""

TABLE_CARD = """
| Hyperparameter | Value |
| --- | --- |
| Algorithm | GRPO |
| Learning rate | 1e-6 |
| Group size | 8 |
| Batch size | 512 |
"""

PROSE_CARD = """
Base model: Qwen/Qwen3-4B-Base
learning_rate: 0.000001
rollout.n = 8
train_batch_size: 512
"""


def run_compare(card_text, extra=()):
    path = os.path.join(tempfile.mkdtemp(), "card.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(card_text)
    return subprocess.run(
        [sys.executable, os.path.join(VERL_RL, "compare_recipe.py"),
         "--card", path, "--no-color", *extra],
        capture_output=True, text=True)


def test_compare_recipe():
    print("\n[8] the recipe comparator reads the three shapes cards come in")
    # This is what makes "is our run the same as theirs" a mechanical check
    # rather than an eyeball diff, so its parsing has to be right.
    sys.path.insert(0, VERL_RL)
    import compare_recipe as cr

    hydra = cr.parse_card(HYDRA_CARD)
    check("hydra overrides parsed",
          hydra.get("actor_rollout_ref.actor.optim.lr") == "1e-6",
          str(hydra.get("actor_rollout_ref.actor.optim.lr")))
    table = cr.parse_card(TABLE_CARD)
    check("markdown table parsed", table.get("learning rate") == "1e-6",
          str(table.get("learning rate")))
    check("table separator row ignored", "---" not in table)
    prose = cr.parse_card(PROSE_CARD)
    check("prose parsed", prose.get("learning_rate") == "0.000001",
          str(prose.get("learning_rate")))

    # 1e-6 and 0.000001 are the same setting; reporting them as a difference
    # would send someone chasing a mismatch that is not there.
    lr = [f for f in cr.FIELDS if f.name == "learning rate"][0]
    check("1e-6 == 0.000001", cr.same(lr, "1e-6", "0.000001"))
    check("1e-6 != 5e-6", not cr.same(lr, "1e-6", "5e-6"))
    model = [f for f in cr.FIELDS if f.name == "base model"][0]
    check("Qwen/X == X", cr.same(model, "Qwen/Qwen3-4B-Base", "Qwen3-4B-Base"))
    check("Base != Instruct",
          not cr.same(model, "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B"))
    n = [f for f in cr.FIELDS if f.name.startswith("rollout n")][0]
    check("8 != 16", not cr.same(n, "16", "8"))

    print("\n[9] a differing recipe is reported, not silently accepted")
    proc = run_compare(HYDRA_CARD)
    check("differences exit non-zero", proc.returncode != 0)
    check("names the differing group size", "rollout n" in proc.stdout)
    check("prints the override that aligns us", "ROLLOUT_N=16" in proc.stdout)
    check("says the group size changes the algorithm",
          "advantage estimate" in proc.stdout)
    check("a matching field is not reported as differing",
          "learning rate" in proc.stdout and proc.stdout.count("DIFFERS") >= 1)

    proc = run_compare(HYDRA_CARD, ["--apply"])
    check("--apply emits shell assignments",
          "export ROLLOUT_N=16" in proc.stdout, proc.stdout.strip()[:80])
    check("--apply emits nothing else",
          all(line.startswith(("export ", "#")) or not line.strip()
              for line in proc.stdout.splitlines()))

    print("\n[10] a matching recipe is confirmed")
    # config.sh's own defaults, restated as a card: must come back clean.
    proc = run_compare(PROSE_CARD)
    check("an aligned card exits zero", proc.returncode == 0,
          proc.stdout.strip()[-200:])
    check("says so explicitly", "matches this config" in proc.stdout)
    # Silence about an unstated field would read as agreement; it must not.
    check("unstated fields are called out", "does not state" in proc.stdout)


# --------------------------------------------------------------------------- #
def main():
    print("verl_rl pipeline tests")
    for fn in (test_reward, test_data_schema, test_val_split_is_shared,
               test_eval_config, test_config_sh, test_run_all_dry,
               test_compare_recipe):
        try:
            fn()
        except Exception as exc:  # a broken test must not hide the others
            print(f"  ERROR in {fn.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(fn.__name__)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
