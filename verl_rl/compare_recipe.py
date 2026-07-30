#!/usr/bin/env python3
r"""Diff a published training recipe against this repo's config.sh.

Written because the recipe we are reproducing lives on a page this environment
cannot reach (see README, "Read this before you start"). Rather than eyeball two
lists of hyperparameters, paste the model card in and let this say exactly which
knobs differ and what to set.

    python verl_rl/compare_recipe.py --card card.md
    pbpaste | python verl_rl/compare_recipe.py --card -
    python verl_rl/compare_recipe.py --card card.md --apply > recipe.env

It reads three shapes, because model cards use all of them:

  * hydra overrides pasted from the launch command
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.rollout.n=8
  * markdown tables
        | learning rate | 1e-6 |
  * plain prose
        learning_rate: 1e-6

Exit status is 0 only when every field the card mentions matches ours, so it can
gate a run:

    python verl_rl/compare_recipe.py --card card.md && ./verl_rl/run_all.sh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


RESET, RED, GREEN, YELLOW, DIM = (
    "\033[0m", "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[2m",
)


class Field:
    """One knob: what we call it, what they might call it, how to compare it."""

    def __init__(self, name, config_var, hydra=(), aliases=(), kind="str",
                 note="", host=False):
        self.name = name
        self.host = host                  # value comes from the machine, not config
        self.config_var = config_var      # the variable in config.sh, "" if fixed
        self.hydra = hydra                # verl override keys, matched by suffix
        self.aliases = aliases            # prose / table spellings, lowercased
        self.kind = kind                  # str | num | bool | model
        self.note = note


def tokens(key: str):
    """Split a key into comparable words.

    Cards write the same knob as `ppo_mini_batch_size`, `ppo mini-batch size`
    and `PPO Mini Batch Size`. Enumerating every spelling per field does not
    scale and quietly reports 'not stated' for a value the card did state --
    which is the one thing this tool must not do. Normalise instead.
    """
    return [t for t in re.split(r"[^a-z0-9]+", str(key).lower()) if t]


def is_subsequence(needle, haystack) -> bool:
    """Are `needle`'s tokens present in `haystack`, in order?

    So "mini batch size" matches "ppo mini batch size", and "batch size"
    matches it too -- the caller resolves that by preferring the longer match.
    """
    it = iter(haystack)
    return all(token in it for token in needle)


# Ordered by how much a mismatch would change the result: the first block
# changes the algorithm, the second changes the optimisation, the third is
# throughput only.
FIELDS = [
    Field("base model", "BASE_MODEL",
          hydra=("actor_rollout_ref.model.path",),
          aliases=("base model", "base_model", "model", "model_name_or_path"),
          kind="model"),
    Field("algorithm", "",
          hydra=("algorithm.adv_estimator",),
          aliases=("algorithm", "adv_estimator", "rl algorithm", "method"),
          note="this pipeline is GRPO; a DAPO/PPO card needs different code, "
               "not just different values"),
    Field("train dataset", "RL_DATASET",
          hydra=("data.train_files",),
          aliases=("train", "dataset", "training data", "train_files",
                   "train file", "train dataset"),
          kind="model"),
    Field("val datasets", "VAL_DATASETS",
          hydra=("data.val_files",),
          aliases=("val", "validation", "val_files", "eval set", "val dataset"),
          kind="list"),
    Field("KL loss", "USE_KL_LOSS",
          hydra=("actor_rollout_ref.actor.use_kl_loss",),
          aliases=("kl_loss", "kl loss", "use_kl_loss", "kl"),
          kind="bool",
          note="disabling it also drops the reference policy, which frees a "
               "model's worth of memory"),
    Field("loss aggregation", "LOSS_AGG_MODE",
          hydra=("actor_rollout_ref.actor.loss_agg_mode",),
          aliases=("loss aggregation", "loss_agg_mode", "loss agg mode",
                   "aggregation")),
    Field("rollout n (group size)", "ROLLOUT_N",
          hydra=("actor_rollout_ref.rollout.n",),
          aliases=("n responses per prompt", "responses per prompt",
                   "samples per prompt", "rollout.n", "num_generations",
                   "group size", "group_size", "n samples", "n rollouts",
                   "num_return_sequences"),
          kind="num",
          note="the GRPO group size -- changes the advantage estimate itself"),
    Field("train batch size (prompts)", "TRAIN_BATCH_SIZE",
          hydra=("data.train_batch_size",),
          aliases=("train_batch_size", "batch size", "batch_size",
                   "global batch size", "rollout batch size"),
          kind="num"),
    Field("mini batch size", "PPO_MINI_BATCH_SIZE",
          hydra=("actor_rollout_ref.actor.ppo_mini_batch_size",),
          aliases=("ppo mini batch size", "mini batch size", "mini_batch_size"),
          kind="num"),
    Field("micro batch per GPU", "MICRO_BATCH_PER_GPU",
          hydra=("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",),
          aliases=("ppo micro batch gpu", "micro batch per gpu",
                   "ppo_micro_batch_size_per_gpu", "micro batch size per gpu",
                   "micro batch"),
          kind="num",
          note="gradient-accumulation chunking, not part of the update; raise it "
               "on a 279 GB card for throughput once the run is matched"),
    Field("learning rate", "LEARNING_RATE",
          hydra=("actor_rollout_ref.actor.optim.lr",),
          aliases=("learning rate", "learning_rate", "lr", "actor lr"),
          kind="num"),
    Field("KL coefficient", "KL_LOSS_COEF",
          hydra=("actor_rollout_ref.actor.kl_loss_coef", "algorithm.kl_ctrl.kl_coef"),
          aliases=("kl_loss_coef", "kl coefficient", "kl coef", "kl_coef",
                   "beta", "kl penalty"),
          kind="num"),
    Field("KL loss type", "KL_LOSS_TYPE",
          hydra=("actor_rollout_ref.actor.kl_loss_type",),
          aliases=("kl_loss_type", "kl type")),
    Field("entropy coefficient", "ENTROPY_COEF",
          hydra=("actor_rollout_ref.actor.entropy_coeff",),
          aliases=("entropy_coeff", "entropy coefficient", "entropy coef"),
          kind="num"),
    Field("rollout temperature", "ROLLOUT_TEMPERATURE",
          hydra=("actor_rollout_ref.rollout.temperature",),
          aliases=("temperature", "rollout temperature"),
          kind="num"),
    Field("max prompt length", "MAX_PROMPT_LEN",
          hydra=("data.max_prompt_length",),
          aliases=("max_prompt_length", "max prompt length", "prompt length"),
          kind="num"),
    Field("max response length", "MAX_RESPONSE_LEN",
          hydra=("data.max_response_length",),
          aliases=("max_response_length", "max response length",
                   "response length", "max_new_tokens", "generation length"),
          kind="num"),
    Field("val response length", "VAL_RESPONSE_LEN",
          aliases=("val response length", "validation response length",
                   "val max response length"),
          kind="num",
          note="validation generates far longer than training here, so it needs "
               "its own budget"),
    Field("max model length", "MAX_MODEL_LEN",
          hydra=("actor_rollout_ref.rollout.max_model_len",),
          aliases=("max model length", "max_model_len", "context length"),
          kind="num"),
    Field("save frequency", "SAVE_FREQ",
          hydra=("trainer.save_freq",),
          aliases=("save freq", "save frequency", "save_freq", "checkpoint freq"),
          kind="num"),
    Field("test frequency", "TEST_FREQ",
          hydra=("trainer.test_freq",),
          aliases=("test freq", "test frequency", "test_freq", "eval freq",
                   "val freq"),
          kind="num"),
    Field("epochs", "TOTAL_EPOCHS",
          hydra=("trainer.total_epochs",),
          aliases=("epochs", "total_epochs", "num_train_epochs"),
          kind="num"),
    Field("training steps", "TOTAL_STEPS",
          hydra=("trainer.total_training_steps",),
          aliases=("steps", "total_training_steps", "training steps",
                   "max_steps", "global steps"),
          kind="num",
          note="0 here means 'derive from epochs'"),
    Field("rollout tensor parallel", "ROLLOUT_TP",
          hydra=("actor_rollout_ref.rollout.tensor_model_parallel_size",),
          aliases=("tensor_model_parallel_size", "tensor parallel", "tp"),
          kind="num",
          note="throughput only; does not change the update"),
    Field("GPUs per node", "N_GPUS",
          hydra=("trainer.n_gpus_per_node",),
          aliases=("n gpus", "n_gpus_per_node", "gpus", "num_gpus", "world size"),
          kind="num", host=True,
          note="read from the machine the run starts on, so it cannot be "
               "checked from a box with no GPUs"),
]

# Fields whose value we hold fixed in code rather than in config.sh.
FIXED = {"algorithm": "grpo"}


# --------------------------------------------------------------------------- #
# parsing the card
# --------------------------------------------------------------------------- #
HYDRA_RE = re.compile(r"([A-Za-z_][\w.]*)\s*=\s*([^\s\\|]+)")
TABLE_RE = re.compile(r"^\s*\|([^|]+)\|([^|]+)\|")
PROSE_RE = re.compile(r"^\s*[-*]?\s*\*{0,2}([A-Za-z][\w .()/_-]{1,40}?)\*{0,2}\s*[:=]\s*`?([^`\n]+?)`?\s*$")


def parse_card(text: str) -> dict:
    """Every key/value pair the card states, lowercased keys.

    Deliberately greedy: collecting a key we do not care about is free, while
    missing one means silently reporting "not stated" for a field the card
    actually pins.
    """
    found = {}

    def put(key, value):
        key = key.strip().strip("`*").lower()
        value = str(value).strip().strip("`,;\"'")
        if key and value and key not in found:
            found[key] = value

    for key, value in HYDRA_RE.findall(text):
        put(key, value)

    for line in text.splitlines():
        m = TABLE_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            # Skip the header separator row.
            if set(value.strip()) <= set("-: "):
                continue
            put(key, value)
            continue
        m = PROSE_RE.match(line)
        if m:
            put(m.group(1), m.group(2))

    return found


def match_score(field: Field, key: str):
    """How well a card key names this field. Higher is better, 0 is no match.

    Scored rather than first-match-wins because aliases overlap: "batch size"
    is a subsequence of "ppo mini batch size", so whichever field is checked
    first would otherwise swallow the other's value.
    """
    key_tokens = tokens(key)

    for hydra in field.hydra:                       # full hydra path: unambiguous
        if key == hydra:
            return 1000
    for hydra in field.hydra:                       # its tail (rollout.n, optim.lr)
        tail = tokens(hydra.split(".")[-1])
        if key_tokens == tail or (len(tail) > 1 and is_subsequence(tail, key_tokens)):
            return 500 + len(tail)

    best = 0
    for alias in field.aliases:
        alias_tokens = tokens(alias)
        if not alias_tokens:
            continue
        if key_tokens == alias_tokens:
            best = max(best, 100 + len(alias_tokens))
        elif is_subsequence(alias_tokens, key_tokens):
            # A partial name is weaker evidence than an exact one, and a longer
            # partial beats a shorter one.
            best = max(best, 10 + len(alias_tokens))
    return best


def assign(fields, card: dict):
    """Best card key for each field, resolving overlapping aliases globally."""
    scored = []
    for field in fields:
        for key in card:
            score = match_score(field, key)
            if score:
                scored.append((score, field.name, key))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    taken_keys, chosen = set(), {}
    for score, field_name, key in scored:
        if field_name in chosen or key in taken_keys:
            continue
        chosen[field_name] = (card[key], key)
        taken_keys.add(key)
    return chosen


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
NUMBER_IN_PROSE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

TRUTHY = {"true", "1", "yes", "on", "enabled", "enable"}
FALSY = {"false", "0", "no", "off", "disabled", "disable", "none"}


def as_number(value):
    try:
        return float(str(value).replace("_", "").rstrip("."))
    except (TypeError, ValueError):
        pass
    # "every 20 steps" -- cards write frequencies as prose.
    match = NUMBER_IN_PROSE.search(str(value))
    return float(match.group()) if match else None


def as_bool(value):
    v = str(value).strip().lower()
    if v in TRUTHY:
        return True
    if v in FALSY:
        return False
    # "kl_loss: 0.0" states the same thing as "kl_loss: disabled".
    n = as_number(v)
    if n is not None:
        return n != 0
    return None


def same(field: Field, theirs, ours) -> bool:
    if theirs is None or ours is None:
        return False
    t, o = str(theirs).strip(), str(ours).strip()
    if field.kind == "bool":
        tb, ob = as_bool(t), as_bool(o)
        return tb is not None and tb == ob
    if field.kind == "num":
        tn, on = as_number(t), as_number(o)
        if tn is not None and on is not None:
            # 1e-6 vs 0.000001 vs 1E-6 are the same setting.
            return abs(tn - on) <= 1e-12 * max(1.0, abs(tn), abs(on))
    if field.kind == "list":
        # "AIME24, AIME25, AMC23" and "AIME24,AIME25,AMC23" are one setting, and
        # the order they are listed in is not meaningful.
        split = lambda v: {p.strip().lower() for p in re.split(r"[,\s]+", v) if p.strip()}
        return split(t) == split(o)
    if field.kind == "model":
        # "Qwen/Qwen3-4B-Base" vs "Qwen3-4B-Base" vs a local path ending in it --
        # and a card that gives a file inside a directory named for the dataset,
        # datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet, is naming the same
        # thing as "DAPO-Math-17k-Processed".
        def segments(value):
            parts = [p for p in re.split(r"[/\\]", value.strip().rstrip("/")) if p]
            return [re.sub(r"\.(parquet|json|jsonl|arrow)$", "", p).lower()
                    for p in parts]
        t_parts, o_parts = segments(t), segments(o)
        if not t_parts or not o_parts:
            return False
        # One side's leaf name must appear *whole* as a segment of the other.
        # Intersecting all segments would be wrong: Qwen/Qwen3-4B-Base and
        # Qwen/Qwen3-4B share "qwen", and calling a base/instruct mix-up a match
        # is the single worst thing this tool could do.
        return t_parts[-1] in o_parts or o_parts[-1] in t_parts
    return t.lower() == o.lower()


def read_our_config(config_path: str) -> dict:
    keys = [f.config_var for f in FIELDS if f.config_var] + ["GPU_COUNT"]
    query = f"source {config_path} >/dev/null 2>&1; " + "; ".join(
        f'echo "${k}"' for k in keys)
    out = subprocess.run(["bash", "-c", query], capture_output=True, text=True)
    values = out.stdout.split("\n")
    ours = {k: (values[i].strip() if i < len(values) else "") for i, k in enumerate(keys)}
    ours.update(FIXED)
    return ours


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True,
                    help="file holding the model card text, or '-' for stdin")
    ap.add_argument("--config", default=os.path.join(here, "config.sh"))
    ap.add_argument("--apply", action="store_true",
                    help="print the env assignments that would align us, "
                         "nothing else (safe to redirect to a file)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    if args.no_color or not sys.stdout.isatty():
        globals().update(RESET="", RED="", GREEN="", YELLOW="", DIM="")

    text = sys.stdin.read() if args.card == "-" else open(args.card, encoding="utf-8").read()
    card = parse_card(text)
    ours = read_our_config(args.config)

    chosen = assign(FIELDS, card)
    rows, mismatches, missing = [], [], []
    for field in FIELDS:
        theirs, stated_as = chosen.get(field.name, (None, None))
        our_value = ours.get(field.config_var or field.name, "")
        if theirs is None:
            verdict = "not stated"
            missing.append(field)
        elif field.host and ours.get("GPU_COUNT", "0") in ("", "0"):
            # No GPU here, so config.sh reported its fallback. Calling that a
            # mismatch would send someone to fix a value that is correct on the
            # machine that matters.
            verdict = "host"
        elif same(field, theirs, our_value):
            verdict = "match"
        else:
            verdict = "DIFFERS"
            mismatches.append((field, theirs, our_value))
        rows.append((field, theirs, our_value, verdict, stated_as))

    if args.apply:
        for field, theirs, _ in mismatches:
            if field.config_var:
                print(f"export {field.config_var}={theirs}")
            else:
                print(f"# {field.name}: card says {theirs!r} -- not a config value, "
                      f"see the note in compare_recipe.py")
        return 0

    width = max(len(f.name) for f in FIELDS) + 2
    print(f"\n  {'field'.ljust(width)}{'theirs'.ljust(26)}{'ours'.ljust(26)}verdict")
    print("  " + "-" * (width + 60))
    for field, theirs, our_value, verdict, stated_as in rows:
        colour = {"match": GREEN, "DIFFERS": RED, "not stated": YELLOW,
                  "host": YELLOW}[verdict]
        t = (theirs if theirs is not None else "-")[:24]
        o = (our_value or "-")[:24]
        print(f"  {field.name.ljust(width)}{t.ljust(26)}{o.ljust(26)}{colour}{verdict}{RESET}")
        if stated_as and verdict != "match":
            print(f"  {' ' * width}{DIM}card states it as {stated_as!r}{RESET}")

    host_rows = [r for r in rows if r[3] == "host"]
    if host_rows:
        print(f"\n{YELLOW}host{RESET} = taken from the machine at run time; this box "
              f"has no GPU, so it could not be checked here.")

    print()
    if mismatches:
        print(f"{RED}{len(mismatches)} field(s) differ.{RESET} To align:\n")
        env = " ".join(f"{f.config_var}={t}" for f, t, _ in mismatches if f.config_var)
        if env:
            print(f"  {env} \\")
            print("    ./verl_rl/run_all.sh")
            print()
            print("  (or edit verl_rl/config.sh, which is where the defaults live)")
        for field, theirs, our_value in mismatches:
            if field.note:
                print(f"\n  {field.name}: {field.note}")
            if not field.config_var:
                print(f"\n  {field.name}: card says {theirs!r}, we do {our_value!r} -- "
                      f"this one is not a config knob.")
    else:
        print(f"{GREEN}every field the card states matches this config.{RESET}")

    if missing:
        print(f"\n{YELLOW}{len(missing)} field(s) the card does not state:{RESET} "
              f"{', '.join(f.name for f in missing)}")
        print("  Ours are used as-is for those. Check the card by eye before "
              "calling the reproduction exact.")

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
