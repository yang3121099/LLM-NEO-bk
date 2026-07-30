#!/usr/bin/env python3
"""Rule-based math reward for verl GRPO on DAPO-Math.

Binary correctness on the final answer: 1.0 if the extracted answer matches the
reference, 0.0 otherwise. No format shaping and no partial credit -- GRPO
normalises advantages within a prompt group, so a sparse binary signal is what
the recipe assumes, and shaping terms would change what is being compared
between the base and instruct runs.

Answer extraction prefers \\boxed{...}, then "answer is X", then a trailing
number, because DAPO-Math responses are free-form chain-of-thought.

verl calls `compute_score(data_source, solution_str, ground_truth, extra_info)`.

Self-test:  python shadow_rl/verl_math/math_reward.py
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
_ANSWER_PATTERNS = [
    re.compile(r"answer\s*is[:\s]*\$?([^\n\.\$]+)", re.IGNORECASE),
    re.compile(r"answer\s*[:=]\s*\$?([^\n\.\$]+)", re.IGNORECASE),
]
_NUMBER = re.compile(r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, brace-balanced.

    A regex cannot do this: answers like \\boxed{\\frac{1}{2}} contain nested
    braces, and a non-greedy match would stop at the first closing brace.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None                      # unterminated brace


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    # Search the tail first: the final answer is what matters, and the working
    # above it is full of intermediate numbers.
    tail = text[-600:]
    for pattern in _ANSWER_PATTERNS:
        found = pattern.findall(tail)
        if found:
            return found[-1].strip()
    numbers = _NUMBER.findall(tail)
    return numbers[-1] if numbers else None


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\:|\\ "), ""),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\$"), ""),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"\s+"), ""),
]
_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")


def normalise(value: str) -> str:
    if value is None:
        return ""
    out = str(value).strip()
    out = _FRAC.sub(r"(\1)/(\2)", out)
    for pattern, repl in _STRIP:
        out = pattern.sub(repl, out)
    out = out.rstrip(".").strip("{}")
    # Trailing units/words that do not change the value.
    out = re.sub(r"(degrees|degree|cm|mm|km|m|kg|g|s)$", "", out, flags=re.IGNORECASE)
    return out.lower()


def as_number(value: str) -> Optional[Fraction]:
    """Exact numeric value if the string is one, else None.

    Fraction rather than float so 1/3 compares equal to 2/6 without tolerance
    games, and 0.5 equals 1/2.
    """
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return Fraction(text)                      # ints, decimals, "3/4"
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"\((-?[\d.]+)\)/\((-?[\d.]+)\)", text)
    if m:
        try:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def is_correct(prediction: str, reference: str) -> bool:
    if prediction is None or reference is None:
        return False
    p, r = normalise(prediction), normalise(reference)
    if p == r and p != "":
        return True
    pn, rn = as_number(p), as_number(r)
    return pn is not None and rn is not None and pn == rn


# --------------------------------------------------------------------------- #
# verl entry point
# --------------------------------------------------------------------------- #
def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs) -> float:
    """Binary correctness. Signature is positional-compatible across verl versions."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("answer"))
    if isinstance(ground_truth, (list, tuple)):
        references = list(ground_truth)
    else:
        references = [ground_truth]

    predicted = extract_answer(solution_str or "")
    if predicted is None:
        return 0.0
    return 1.0 if any(is_correct(predicted, r) for r in references) else 0.0


if __name__ == "__main__":
    cases = [
        # (response, reference, expected)
        (r"so the answer is \boxed{42}.", "42", 1.0),
        (r"\boxed{\frac{1}{2}}", "0.5", 1.0),
        (r"\boxed{\frac{2}{6}}", "1/3", 1.0),
        (r"The answer is 17", "17", 1.0),
        (r"answer: -3", "-3", 1.0),
        (r"...therefore 1024", "1024", 1.0),
        (r"\boxed{1,024}", "1024", 1.0),
        (r"\boxed{60degrees}", "60", 1.0),
        (r"\boxed{42}", "43", 0.0),
        (r"no answer here at all", "42", 0.0),
        ("", "42", 0.0),
        (r"\boxed{\frac{1}{3}}", "0.333", 0.0),      # not equal, correctly rejected
        # Nested braces: a non-greedy regex would return "\frac{1" here.
        (r"\boxed{\frac{\sqrt{2}}{2}}", r"\frac{\sqrt{2}}{2}", 1.0),
        # Last boxed wins, not the first.
        (r"\boxed{1} ... on reflection \boxed{2}", "2", 1.0),
    ]
    failures = 0
    for response, reference, want in cases:
        got = compute_score("dapo", response, reference)
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            failures += 1
        print(f"  {flag}  {response[:44]!r:<48} vs {reference!r:<12} -> {got}")
    print("\n" + ("all checks passed" if not failures else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)
