#!/usr/bin/env python3
"""Tests for the terminal report and the FINDINGS verdict.

The report is what gets read to decide whether the experiment worked, so the
things pinned here are: columns stay aligned once ANSI colour is involved, the
win/loss call matches the arithmetic, and a partial row is flagged rather than
quietly averaged against complete ones.

Run:  python shadow_rl/tests/test_report.py
"""

import csv
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FIELDS = ["version", "size", "algo", "with_search", "model_role", "dataset", "em", "n_questions"]
DS = ["nq", "hotpotqa", "musique", "bamboogle"]
FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def write_csv(path, pairs):
    """pairs: list of (version, size, algo, with_search, {role: {dataset: em}})."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for version, size, algo, search, roles in pairs:
            for role, per in roles.items():
                for d, em in per.items():
                    w.writerow({"version": version, "size": size, "algo": algo,
                                "with_search": search, "model_role": role,
                                "dataset": d, "em": em, "n_questions": 200})


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *args],
                          capture_output=True, text=True)


def flat(em):
    return {d: em for d in DS}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "r.csv")

        print("\n[1] shadow wins -> reported as a win")
        write_csv(csv_path, [("v0.2", "3b", "ppo", False, {
            "base_baseline": flat(0.05), "instruct_baseline": flat(0.11),
            "rl_on_instruct": flat(0.21), "rl_on_base": flat(0.23),
            "shadow": flat(0.25)})])
        p = run("report.py", "--results", csv_path, "--no-color")
        out = p.stdout
        check("exits cleanly", p.returncode == 0, p.stderr.strip()[:200])
        check("all five roles listed",
              all(s in out for s in ("W_B", "W_I", "RL(W_I)", "RL(W_B)", "W_shadow")))
        check("win against RL(W_I)", "WIN" in out and "+0.0400" in out)
        check("in-domain datasets starred", "NQ*" in out and "HotpotQA*" in out)
        check("OOD datasets not starred", "Musique*" not in out)
        check("dataset coverage stated", "4/7 datasets" in out)
        check("sample size stated", "n=200/dataset" in out)
        check("subsample warning shown", "subsampled run" in out)

        print("\n[2] shadow loses -> reported as a loss, not hidden")
        write_csv(csv_path, [("v0.3", "3b", "grpo", True, {
            "rl_on_instruct": flat(0.34), "rl_on_base": flat(0.31),
            "shadow": flat(0.32)})])
        out = run("report.py", "--results", csv_path, "--no-color").stdout
        check("loss against RL(W_I)", "LOSS" in out and "-0.0200" in out)
        check("win against RL(W_B) still shown", "+0.0100" in out)

        print("\n[3] columns stay aligned with colour on")
        write_csv(csv_path, [("v0.2", "3b", "ppo", False, {
            "base_baseline": flat(0.05), "instruct_baseline": flat(0.11),
            "rl_on_instruct": flat(0.21), "rl_on_base": flat(0.23),
            "shadow": flat(0.25)})])
        plain = run("report.py", "--results", csv_path, "--no-color").stdout
        # Force colour by unsetting NO_COLOR and asking the module directly.
        sys.path.insert(0, ROOT)
        import report as rp
        st_colour, st_plain = rp.Style(True), rp.Style(False)
        check("width() ignores ANSI escapes",
              st_colour.width(st_colour.green("0.250")) == 5)
        check("pad() aligns coloured text",
              st_colour.width(st_colour.pad(st_colour.green("0.250"), 10)) == 10)
        check("plain and coloured pad to the same width",
              st_plain.width(st_plain.pad("0.250", 10))
              == st_colour.width(st_colour.pad(st_colour.green("0.250"), 10)))

        # Every data row in the plain render must have the same column count.
        # Require an EM number on the line: the header subtitle also begins with
        # "  W_shadow = ..." and would otherwise be counted as a row.
        rows = [ln for ln in plain.splitlines()
                if re.match(r"^  (W_B|W_I|RL\(W_|W_shadow)", ln)
                and re.search(r"\d\.\d{3}", ln)]
        check("one row per role", len(rows) == 5, f"{len(rows)} rows")
        widths = {len(ln.rstrip()) for ln in rows}
        numbers = [len(re.findall(r"\d\.\d{3}", ln)) for ln in rows]
        check("every row has 4 datasets + Avg", set(numbers) == {5}, f"{numbers}")

        print("\n[4] a role missing a dataset is flagged")
        write_csv(csv_path, [("v0.2", "3b", "ppo", False, {
            "rl_on_instruct": flat(0.21),
            "rl_on_base": flat(0.23),
            "shadow": {"nq": 0.25, "hotpotqa": 0.25}})])   # missing 2 of 4
        out = run("report.py", "--results", csv_path, "--no-color").stdout
        check("incomplete row marked with !", "!" in out)

        print("\n[5] cross-pair overview appears only with 2+ pairs")
        out1 = run("report.py", "--results", csv_path, "--no-color").stdout
        check("single pair -> no overview", "overview" not in out1)
        write_csv(csv_path, [
            ("v0.2", "3b", "ppo", False, {"rl_on_instruct": flat(0.21),
                                          "rl_on_base": flat(0.23), "shadow": flat(0.25)}),
            ("v0.3", "3b", "grpo", True, {"rl_on_instruct": flat(0.34),
                                          "rl_on_base": flat(0.31), "shadow": flat(0.32)}),
        ])
        out2 = run("report.py", "--results", csv_path, "--no-color").stdout
        check("two pairs -> overview", "overview" in out2)
        check("overview tallies wins", "beats RL(W_I) on 1/2" in out2)

        print("\n[6] FINDINGS.md leads with a verdict")
        md_path = os.path.join(tmp, "F.md")
        p = run("aggregate.py", "--results", csv_path, "--out", md_path)
        check("aggregate.py succeeded", p.returncode == 0, p.stderr.strip()[:200])
        md = open(md_path).read()
        check("verdict section first", md.index("## Verdict") < md.index("## Summary"))
        check("win counted", "**1 of 2**" in md)
        check("win listed with a tick", "✅" in md and "+0.040" in md)
        check("loss listed with a cross", "❌" in md and "-0.020" in md)
        check("W_B column present", "| W_B |" in md)

        print("\n[7] --pair filters to one pair")
        out = run("report.py", "--results", csv_path, "--no-color",
                  "--pair", "grpo-search-3b-v0.3").stdout
        check("selected pair shown", "grpo-search-3b-v0.3" in out)
        check("other pair hidden", "ppo-nosearch-3b-v0.2" not in out)

        print("\n[8] empty results does not crash")
        empty = os.path.join(tmp, "e.csv")
        with open(empty, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
        p = run("report.py", "--results", empty, "--no-color")
        check("exits cleanly", p.returncode == 0)
        check("says there is nothing yet", "no results yet" in p.stdout)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
