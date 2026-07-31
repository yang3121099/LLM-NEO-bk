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

    failures: List[str] = []
    lock = threading.Lock()
    started = time.time()
    completed = [0]

    def worker(slot: int) -> None:
        # Pin this worker to its own GPU(s). vLLM then sees exactly these and
        # numbers them from zero, so tp>1 works without further coordination.
        gpus = ",".join(str(slot * args.tp + i) for i in range(args.tp))
        while True:
            try:
                role, dataset = pending.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, os.path.join(HERE, "evaluate.py"),
                   "--search-r1-root", args.search_r1_root,
                   "--pair", args.pair, "--role", role,
                   "--datasets", dataset,
                   "--out", shards[slot],
                   "--retriever-url", args.retriever_url,
                   "--tensor-parallel-size", str(args.tp),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
            if role == "shadow":
                if not shadow_path:
                    with lock:
                        failures.append(f"{role}/{dataset}: --shadow-path not given")
                    continue
                cmd += ["--model-path", shadow_path]
            if args.sample:
                cmd += ["--sample", str(args.sample)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]

            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
            with lock:
                print(f"[gpu {gpus}] start  {role} / {dataset}", flush=True)
            log = os.path.join(os.path.dirname(os.path.abspath(args.out)), "logs",
                               f"{args.pair}.{role}.{dataset}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w") as fh:
                proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            with lock:
                completed[0] += 1
                tag = "ok" if proc.returncode == 0 else "FAIL"
                print(f"[gpu {gpus}] {tag:<4}   {role} / {dataset}   "
                      f"({completed[0]}/{len(work)}, {(time.time()-started)/60:.1f}m)",
                      flush=True)
                if proc.returncode != 0:
                    failures.append(f"{role}/{dataset} (see {log})")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = merge_shards(shards, args.out)
    minutes = (time.time() - started) / 60
    print(f"\n[parallel] {completed[0]} cell(s) in {minutes:.1f}m -> {args.out} "
          f"({total} row(s) total)")
    if failures:
        print(f"[parallel] {len(failures)} failure(s):")
        for f in failures:
            print(f"    {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
