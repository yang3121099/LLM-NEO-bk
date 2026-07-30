#!/usr/bin/env python3
"""Rule-based reward for math RL, in verl's custom-reward-function shape.

verl resolves a scorer from the row's ``data_source`` against a fixed registry,
and a name the registry does not know raises at the first batch rather than at
config time. Passing this file as ``custom_reward_function.path`` removes that
coupling: the contract is one function, and the dataset can be called anything.

The score is binary -- 1.0 if the extracted answer is equivalent to the ground
truth, 0.0 otherwise. GRPO normalises advantages within a prompt group, so a
graded reward mostly adds variance to the baseline rather than signal; a
verifiable answer is exactly the case where binary is the right choice.

Equivalence uses ``math_verify`` when it is installed (setup_env.sh installs it
for the OpenCompass evaluators) and falls back to string normalisation, which
handles the common LaTeX spellings of the same number. The fallback is stricter,
never looser: it can score a correct answer 0, never an incorrect one 1.

Run it directly to see the extraction and comparison on a few examples::

    python verl_rl/reward_math.py --self-test
"""

from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# answer extraction
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    r"""Content of the LAST ``\boxed{...}`` in the text, brace-balanced.

    A regex cannot do this: answers legitimately contain braces
    (``\boxed{\frac{1}{2}}``), and a non-greedy match stops at the first ``}``
    while a greedy one runs to the end of the response. The last box is the one
    that counts -- a chain of thought often boxes intermediate results.
    """
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        # \boxed 42 -- rare, but seen from base models that have not learned the
        # brace convention yet.
        rest = text[i:].strip().split("\n")[0].strip()
        return rest or None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None  # unbalanced: the response was truncated mid-answer


_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER = re.compile(
    r"(?:final answer|the answer)\s*(?:is|:)\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def extract_answer(solution_str: str) -> str | None:
    r"""The model's answer, by the first convention that matches.

    Order matters: \boxed is what the prompt asks for and what the eval
    harnesses read, so it wins over a prose "the answer is" that may appear
    earlier in the reasoning.
    """
    if not solution_str:
        return None
    boxed = extract_boxed(solution_str)
    if boxed is not None:
        return boxed
    tagged = _ANSWER_TAG.findall(solution_str)
    if tagged:
        return tagged[-1].strip()
    stated = _FINAL_ANSWER.findall(solution_str)
    if stated:
        return stated[-1].strip()
    return None


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #
_STRIP_PREFIXES = ("the answer is", "answer:", "answer is", "$", "\\$")
_STRIP_WRAPPERS = (r"\text{", r"\mathrm{", r"\mbox{")


def normalise(ans: str) -> str:
    r"""Reduce an answer to a comparable form.

    Deliberately conservative. Every rule here collapses two spellings of the
    *same* value (``0.5`` vs ``.5``, ``1,000`` vs ``1000``); none of them make
    different values compare equal.
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\text\s*\{\s*([^}]*)\s*\}", r"\1", s)
    for wrapper in _STRIP_WRAPPERS:
        s = s.replace(wrapper, "")
    s = s.strip().strip("$").strip()
    low = s.lower()
    for prefix in _STRIP_PREFIXES:
        if low.startswith(prefix):
            s = s[len(prefix):].strip()
            low = s.lower()
    s = s.rstrip(".").strip()
    s = s.replace("%", "").replace("\\%", "")
    # \boxed{1{,}000} -- the LaTeX idiom for a thousands separator.
    s = s.replace("{,}", ",")
    # 1,000,000 -> 1000000, but leave (1,2) tuples alone.
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.startswith(".") and len(s) > 1 and s[1].isdigit():
        s = "0" + s
    # 12.0 and 12 are the same answer; 12.5 is not 12.
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _math_verify_equal(pred: str, gold: str) -> bool | None:
    """Symbolic comparison, or None when math_verify is not installed."""
    try:
        from math_verify import parse, verify
    except Exception:
        return None
    try:
        gold_parsed = parse(f"${gold}$")
        pred_parsed = parse(f"${pred}$")
        if not gold_parsed or not pred_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        # A parse failure is not evidence of inequality -- let the string path
        # answer instead of scoring a correct response 0.
        return None


def is_equivalent(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    p, g = normalise(pred), normalise(gold)
    if p == g and p != "":
        return True
    pn, gn = _numeric(p), _numeric(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    verified = _math_verify_equal(pred, gold)
    return bool(verified)


# --------------------------------------------------------------------------- #
# verl entry point
# --------------------------------------------------------------------------- #
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return the reward for one rollout.

    Signature matches what verl's reward manager calls. `solution_str` is the
    decoded response; `ground_truth` comes from the row's
    ``reward_model.ground_truth``.
    """
    answer = extract_answer(solution_str)
    if answer is None:
        # No parseable answer at all. Kept at 0.0 rather than a small negative:
        # GRPO subtracts the group mean, so a constant offset changes nothing,
        # and a format penalty would need tuning to be worth its complexity.
        return 0.0
    return 1.0 if is_equivalent(answer, ground_truth) else 0.0


# --------------------------------------------------------------------------- #
def _self_test() -> int:
    cases = [
        # (response, gold, expected)
        (r"so the answer is \boxed{42}.", "42", 1.0),
        (r"\boxed{42}", "43", 0.0),
        (r"first \boxed{7} then \boxed{42}", "42", 1.0),        # last box wins
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", 1.0),
        (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", 1.0),          # dfrac == frac
        (r"\boxed{0.5}", ".5", 1.0),
        (r"\boxed{1{,}000}", "1000", 1.0),
        (r"\boxed{1,000}", "1000", 1.0),
        (r"\boxed{12.0}", "12", 1.0),
        (r"\boxed{\text{yes}}", "yes", 1.0),
        (r"\boxed{60^\circ}", "60", 1.0),
        ("The final answer is 42", "42", 1.0),
        ("<answer>42</answer>", "42", 1.0),
        ("no answer here", "42", 0.0),
        (r"truncated \boxed{4", "4", 0.0),                       # unbalanced brace
        ("", "42", 0.0),
    ]
    failures = 0
    for response, gold, want in cases:
        got = compute_score("test", response, gold)
        status = "PASS" if got == want else "FAIL"
        if got != want:
            failures += 1
        print(f"  {status}  {response[:40]!r} vs {gold!r} -> {got} (want {want})")
    print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    ap.print_help()
