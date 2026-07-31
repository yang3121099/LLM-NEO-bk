#!/usr/bin/env python3
"""Spread one pair's evaluation across the available GPUs.

A 3B model does not need tensor parallelism, so on a multi-GPU box the win comes
from running independent (role, dataset) jobs side by side rather than from
splitting one model. Each worker gets its own GPU via CUDA_VISIBLE_DEVICES, runs
`evaluate.py` for a single role and dataset, and writes to its own CSV shard.
The shards are merged at the end.

Separate shards rather than a shared file on purpose: concurrent appends to one
CSV are only atomic for small writes and only on some filesystems, and a torn
row would be a silently corrupt result.

Failures are reported where they happen. Most things that break a multi-GPU run
break every job identically -- a vLLM engine that will not start, a model that
will not load, a retriever that is not listening -- so the first job is run alone
as a canary before fanning out, and a run of consecutive failures aborts the rest
instead of reproducing the same error thirty times. Every failure prints the
error from its log inline; you should never have to go looking for it.

    python shadow_rl/run_eval_parallel.py --pair grpo-search-3b-v0.3 \
        --search-r1-root third_party/Search-R1 --sample 200
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import CSV_FIELDS  # noqa: E402
from pairs import DATASETS, MODEL_ROLES, PAIRS_BY_ID  # noqa: E402
from paths import search_r1_root as _default_search_r1_root  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Lines that mean "this is the actual problem" rather than progress chatter.
_ERROR_HINTS = (
    "Traceback (most recent call last)",
    "Error", "error:", "ERROR",
    "out of memory", "OutOfMemory", "CUDA error",
    "Connection refused", "ConnectionError", "Max retries exceeded",
    "AssertionError", "RuntimeError", "ValueError", "OSError",
    "No module named", "not found", "Killed", "core dumped",
    "EngineCore", "EngineDeadError",
)
# Noise that matches the hints above but never explains anything.
_ERROR_NOISE = (
    "error_count", "errors=0", "ERROR:root:  ",
)


def extract_error(log_path: str, max_lines: int = 12) -> List[str]:
    """Pull the part of a worker log that explains why it died.

    A vLLM log is thousands of lines of progress bars and engine chatter; the
    cause is usually a python traceback near the end, or a single line about
    memory or a refused connection. Prefer the last traceback if there is one,
    otherwise the last lines that look like errors, otherwise just the tail --
    something is always better than a bare "FAIL".
    """
    try:
        with open(log_path, errors="replace") as fh:
            lines = [l.rstrip() for l in fh.readlines()]
    except OSError as exc:
        return [f"(could not read {log_path}: {exc})"]
    lines = [l for l in lines if l.strip()]
    if not lines:
        return ["(log is empty -- the worker died before writing anything, "
                "which usually means the process was killed: OOM or a signal)"]

    # A traceback is the most informative thing available, so prefer the last one.
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most recent call last)")]
    if starts:
        block = lines[starts[-1]:]
        if len(block) <= max_lines:
            return block
        # Keep the head (where it was raised) and the tail (what was raised).
        head, tail = max_lines // 3, max_lines - max_lines // 3 - 1
        return block[:head] + [f"    ... {len(block) - head - tail} line(s) omitted ..."] \
            + block[-tail:]

    hits = [l for l in lines
            if any(h in l for h in _ERROR_HINTS)
            and not any(n in l for n in _ERROR_NOISE)]
    if hits:
        return hits[-max_lines:]
    return lines[-max_lines:]


def failure_signature(error_lines: List[str]) -> str:
    """A short key for grouping failures that are really the same failure."""
    for line in reversed(error_lines):
        stripped = line.strip()
        # The last line of a traceback is "ExceptionType: message".
        if stripped and not stripped.startswith(("File \"", "...", "    ")):
            return stripped[:120]
    return (error_lines[-1].strip()[:120] if error_lines else "unknown")


def visible_gpus() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return len([x for x in env.split(",") if x.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


def done_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {(r["model_role"], r["dataset"]) for r in csv.DictReader(fh)}


def merge_shards(shards: List[str], out: str) -> int:
    """Fold shard rows into `out`, keeping one row per (pair, role, dataset)."""
    rows = {}
    for path in [out] + shards:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["version"], r["size"], r["algo"], r["with_search"],
                       r["model_role"], r["dataset"])
                rows[key] = r
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({k: rows[key].get(k, "") for k in CSV_FIELDS})
    for path in shards:
        if os.path.exists(path):
            os.unlink(path)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS_BY_ID))
    ap.add_argument("--search-r1-root", default=_default_search_r1_root())
    ap.add_argument("--out", default="shadow_rl/results.csv")
    ap.add_argument("--roles", default=",".join(MODEL_ROLES))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-datasets", default="")
    ap.add_argument("--model-path", default=None, help="for --roles shadow")
    ap.add_argument("--shadow-path", default=None, help="alias of --model-path")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--tp", type=int, default=1, help="GPUs per worker")
    ap.add_argument("--jobs", default="auto",
                    help="concurrent workers; 'auto' = visible GPUs // tp")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", type=int, default=3, metavar="N",
                    help="abort after N consecutive failures (0 disables). "
                         "Most multi-GPU breakage is systematic, and running the "
                         "remaining jobs only reproduces the same error")
    ap.add_argument("--no-canary", action="store_true",
                    help="skip the single-job trial run and fan out immediately")
    ap.add_argument("--timeout", type=float, default=0, metavar="MIN",
                    help="kill a job that has run this long (0 = no limit)")
    ap.add_argument("--stagger", type=float, default=15.0, metavar="SEC",
                    help="delay between worker starts; keeps N vLLM engines from "
                         "initialising and downloading the same weights at once")
    args = ap.parse_args()

    pair = PAIRS_BY_ID[args.pair]
    shadow_path = args.shadow_path or args.model_path

    n_gpu = visible_gpus()
    if args.jobs == "auto":
        jobs = max(1, n_gpu // max(1, args.tp))
    else:
        jobs = max(1, int(args.jobs))
    if n_gpu and jobs * args.tp > n_gpu:
        sys.exit(f"[fail] {jobs} job(s) x tp={args.tp} needs {jobs * args.tp} GPUs, "
                 f"but only {n_gpu} are visible")

    skip = {d.strip() for d in args.skip_datasets.split(",") if d.strip()}
    datasets = [d.strip() for d in args.datasets.split(",")
                if d.strip() and d.strip() not in skip]
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    already = set() if args.force else done_keys(args.out)
    work: List[Tuple[str, str]] = [(r, d) for r in roles for d in datasets
                                   if (r, d) not in already]

    # ---- preflight: things that are cheap to check and fatal to every job --- #
    # Each of these otherwise fails identically in all N workers, several minutes
    # in, after every one of them has loaded a model.
    problems = []
    if "shadow" in roles and shadow_path and not os.path.isdir(shadow_path):
        problems.append(f"--shadow-path {shadow_path} does not exist "
                        "(run the merge stage first)")
    if "shadow" in roles and not shadow_path:
        problems.append("role 'shadow' was requested but --shadow-path was not given")
    if pair.with_search:
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(
                args.retriever_url,
                data=b'{"queries":["ping"],"topk":1,"return_scores":false}',
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read(1)
        except urllib.error.HTTPError:
            pass          # answered, just not happily -- it is listening, which is the test
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"retriever at {args.retriever_url} is not answering ({exc}). "
                f"{args.pair} is a search pair, so every job needs it: "
                "./shadow_rl/launch_bm25_retriever.sh, or pass --auto-retriever "
                "to run_all.sh")
    if problems:
        print("[parallel] preflight failed:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)

    print(f"[parallel] pair={args.pair}  gpus={n_gpu}  workers={jobs}  tp={args.tp}")
    print(f"[parallel] {len(roles)} role(s) x {len(datasets)} dataset(s) = "
          f"{len(roles) * len(datasets)} cell(s); {len(work)} to run, "
          f"{len(roles) * len(datasets) - len(work)} already done")
    if not work:
        print("[parallel] nothing to do")
        return

    if jobs == 1:
        print("[parallel] single worker: running roles sequentially, one model load each")

    pending: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    for item in work:
        pending.put(item)

    shards = [f"{args.out}.part{i}" for i in range(jobs)]
    for path in shards:
        if os.path.exists(path):
            os.unlink(path)

    failures: List[Tuple[str, str, List[str]]] = []   # (cell, log path, error lines)
    lock = threading.Lock()
    started = time.time()
    completed = [0]
    consecutive_failures = [0]
    aborted = threading.Event()

    def build_cmd(role: str, dataset: str, shard: str) -> List[str]:
        cmd = [sys.executable, os.path.join(HERE, "evaluate.py"),
               "--search-r1-root", args.search_r1_root,
               "--pair", args.pair, "--role", role,
               "--datasets", dataset,
               "--out", shard,
               "--retriever-url", args.retriever_url,
               "--tensor-parallel-size", str(args.tp),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        if role == "shadow":
            cmd += ["--model-path", shadow_path]
        if args.sample:
            cmd += ["--sample", str(args.sample)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        return cmd

    def run_cell(role: str, dataset: str, gpus: str, shard: str) -> Tuple[int, str, List[str]]:
        """Run one (role, dataset) on `gpus`. Returns (rc, log path, error lines)."""
        log = os.path.join(os.path.dirname(os.path.abspath(args.out)), "logs",
                           f"{args.pair}.{role}.{dataset}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
        cmd = build_cmd(role, dataset, shard)
        with open(log, "w") as fh:
            fh.write(f"# CUDA_VISIBLE_DEVICES={gpus}\n# {' '.join(cmd)}\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=args.timeout * 60 if args.timeout else None)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return 124, log, [f"(killed after {args.timeout:g} minute(s) by --timeout)"]
        return rc, log, ([] if rc == 0 else extract_error(log))

    def report(role: str, dataset: str, gpus: str, rc: int, log: str,
               error: List[str]) -> None:
        """Print one result. Failures carry their reason; never a bare FAIL."""
        completed[0] += 1
        tag = "ok" if rc == 0 else "FAIL"
        print(f"[gpu {gpus}] {tag:<4}   {role} / {dataset}   "
              f"({completed[0]}/{len(work)}, {(time.time() - started) / 60:.1f}m)",
              flush=True)
        if rc == 0:
            consecutive_failures[0] = 0
            return
        consecutive_failures[0] += 1
        print(f"           exit {rc}, log: {log}")
        for line in error:
            print(f"           | {line}")
        failures.append((f"{role}/{dataset}", log, error))
        if args.fail_fast and consecutive_failures[0] >= args.fail_fast and not aborted.is_set():
            aborted.set()
            print(f"\n[parallel] {consecutive_failures[0]} failures in a row — stopping "
                  f"rather than reproducing this {pending.qsize()} more time(s).",
                  flush=True)
            print("[parallel] rerun with --fail-fast 0 to push through anyway.",
                  flush=True)

    def worker(slot: int) -> None:
        # Pin this worker to its own GPU(s). vLLM then sees exactly these and
        # numbers them from zero, so tp>1 works without further coordination.
        gpus = ",".join(str(slot * args.tp + i) for i in range(args.tp))
        # Stagger the starts: N vLLM engines initialising at the same instant
        # contend for the same NCCL ports, the same HuggingFace cache lock and
        # the same host RAM during weight load, which turns a working
        # configuration into a flaky one.
        if args.stagger > 0 and slot > 0:
            if aborted.wait(timeout=args.stagger * slot):
                return
        while not aborted.is_set():
            try:
                role, dataset = pending.get_nowait()
            except queue.Empty:
                return
            if role == "shadow" and not shadow_path:
                with lock:
                    report(role, dataset, gpus, 2, "(not run)",
                           ["--shadow-path was not given, so there is no model to load"])
                continue
            with lock:
                print(f"[gpu {gpus}] start  {role} / {dataset}", flush=True)
            rc, log, error = run_cell(role, dataset, gpus, shards[slot])
            with lock:
                report(role, dataset, gpus, rc, log, error)

    # ---- canary ------------------------------------------------------------ #
    # Almost everything that breaks a multi-GPU run breaks it identically on
    # every GPU: an engine that will not start, a model that will not load, a
    # retriever that is not listening. Prove one job works before paying for
    # eight of them to fail.
    if not args.no_canary and jobs > 1 and len(work) > 1:
        role, dataset = pending.get_nowait()
        gpus = ",".join(str(i) for i in range(args.tp))
        print(f"[parallel] canary: {role} / {dataset} on gpu {gpus} "
              f"(one job, to check the setup before fanning out)", flush=True)
        rc, log, error = run_cell(role, dataset, gpus, shards[0])
        report(role, dataset, gpus, rc, log, error)
        if rc != 0:
            print("\n[parallel] the canary failed, so the other "
                  f"{pending.qsize()} job(s) would too. Nothing else was started.")
            print(f"[parallel] full log: {log}")
            merge_shards(shards, args.out)
            sys.exit(1)
        consecutive_failures[0] = 0
        print("[parallel] canary passed; fanning out", flush=True)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = merge_shards(shards, args.out)
    minutes = (time.time() - started) / 60
    print(f"\n[parallel] {completed[0]} cell(s) in {minutes:.1f}m -> {args.out} "
          f"({total} row(s) total)")
    if aborted.is_set() and not pending.empty():
        print(f"[parallel] {pending.qsize()} cell(s) were not attempted (aborted early). "
              "They will be picked up on the next run.")
    if failures:
        # Group by what actually went wrong: thirty jobs failing for one reason
        # is one problem to fix, and reading it thirty times does not help.
        grouped: dict = {}
        for cell, log, error in failures:
            grouped.setdefault(failure_signature(error), []).append((cell, log))
        print(f"\n[parallel] {len(failures)} failure(s), {len(grouped)} distinct cause(s):")
        for sig, cells in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            names = ", ".join(c for c, _ in cells[:4])
            more = f" (+{len(cells) - 4} more)" if len(cells) > 4 else ""
            print(f"\n  {len(cells)}x  {sig}")
            print(f"      cells: {names}{more}")
            print(f"      log  : {cells[0][1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
