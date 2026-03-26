#!/usr/bin/env python3
"""
EvalScope tool-use evaluation script (BFCL + ToolBench).

Prerequisites:
    pip install 'evalscope[all]' vllm

Usage:
    # Evaluate all models (starts vLLM server automatically):
    python3 scripts/eval_tooluse_evalscope.py

    # Evaluate a single model against an already-running vLLM server:
    python3 scripts/eval_tooluse_evalscope.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --model-name Llama-3.1-8B-Instruct \
        --api-url http://localhost:8000/v1

    # Override result date prefix:
    RESULT_MMDD=0727 python3 scripts/eval_tooluse_evalscope.py
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_DIR / "results"
EVALSCOPE_OUT = RESULTS_DIR / "evalscope"

# Date prefix for result directories (from generate_dataset_scripts.sh)
MMDD = os.environ.get("RESULT_MMDD", "0326")


def get_model_list():
    """Return list of (abbr, model_path) tuples."""
    llama_base = "Llama-3.1-8B"
    qwen_base = "Qwen3-8B-Base"
    llama_rel = f"{MMDD}/result-{llama_base}-{MMDD}"
    qwen_rel = f"{MMDD}/result-{qwen_base}-{MMDD}"

    def merged(rel, tag, k, suffix, merge):
        return str(
            RESULTS_DIR
            / rel
            / f"{tag}-{k}-lora-rank128-lr0.0002-{suffix}"
            / f"merged-{merge}"
        )

    models = [
        # HF baselines
        ("Llama-3.1-8B-Instruct-hf", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Qwen3-8B-hf", "Qwen/Qwen3-8B"),
    ]

    # dolci_mix and nemotron_if, B2I and I2I, for both model families
    for base_name, rel in [(llama_base, llama_rel), (qwen_base, qwen_rel)]:
        for suffix in ["dolci_mix", "nemotron_if"]:
            models.append(
                (f"{base_name}-{suffix}-2k-B2I", merged(rel, "B", "2k", suffix, "B2I"))
            )
            models.append(
                (f"{base_name}-{suffix}-2k-I2I", merged(rel, "I", "2k", suffix, "I2I"))
            )

    return models


def start_vllm_server(model_path, port=8234, tp=1, gpu_util=0.9, max_len=16384):
    """Start a vLLM OpenAI-compatible server, return (process, port)."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
        "--trust-remote-code",
        "--dtype", "auto",
    ]
    log_path = EVALSCOPE_OUT / "vllm_server.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    import urllib.request
    max_wait = 300
    for elapsed in range(0, max_wait, 5):
        time.sleep(5)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3)
            print(f"  vLLM server ready ({elapsed + 5}s)")
            return proc, port
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {proc.returncode}")
            continue

    proc.kill()
    raise RuntimeError(f"vLLM server did not start within {max_wait}s")


def stop_vllm_server(proc):
    """Gracefully stop the vLLM server."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_bfcl_eval(model_name, api_url, out_dir):
    """Run BFCL evaluation via evalscope."""
    print(f"  Running BFCL evaluation ...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model=model_name,
            api_url=api_url,
            eval_type="service",
            datasets=["bfcl"],
            work_dir=str(out_dir),
        )
        results = run_task(task_cfg)
        print(f"  BFCL results: {results}")
        return results
    except ImportError:
        print("  evalscope not available, trying bfcl-eval CLI ...")
        subprocess.run(
            [
                "bfcl", "evaluate",
                "--model", model_name,
                "--api-base", api_url,
                "--output-dir", str(out_dir),
            ],
            check=False,
        )
    except Exception as e:
        print(f"  BFCL eval error: {e}")
        return None


def run_toolbench_eval(model_name, api_url, out_dir):
    """Run ToolBench evaluation via evalscope."""
    print(f"  Running ToolBench evaluation ...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model=model_name,
            api_url=api_url,
            eval_type="service",
            datasets=["toolbench"],
            work_dir=str(out_dir),
        )
        results = run_task(task_cfg)
        print(f"  ToolBench results: {results}")
        return results
    except ImportError:
        print("  evalscope not installed. Run: pip install 'evalscope[all]'")
    except Exception as e:
        print(f"  ToolBench eval error: {e}")
        return None


def evaluate_model(abbr, model_path, api_url=None, port=8234):
    """Evaluate a single model on BFCL + ToolBench."""
    out_dir = EVALSCOPE_OUT / abbr

    # Check local path existence
    if model_path.startswith("/") and not Path(model_path).is_dir():
        print(f"  SKIP: {abbr} — path not found: {model_path}")
        return

    print(f"\n{'=' * 64}")
    print(f"  Model: {abbr}")
    print(f"  Path:  {model_path}")
    print(f"{'=' * 64}")

    server_proc = None
    try:
        if api_url is None:
            # Start vLLM server
            server_proc, port = start_vllm_server(model_path, port=port)
            api_url_used = f"http://localhost:{port}/v1"
        else:
            api_url_used = api_url

        # Run evaluations
        run_bfcl_eval(abbr, api_url_used, out_dir / "bfcl")
        run_toolbench_eval(abbr, api_url_used, out_dir / "toolbench")

    finally:
        if server_proc:
            print("  Stopping vLLM server ...")
            stop_vllm_server(server_proc)
            time.sleep(2)

    print(f"  Done: {abbr} -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="EvalScope tool-use evaluation")
    parser.add_argument("--model-path", help="Single model path to evaluate")
    parser.add_argument("--model-name", help="Model abbreviation (for results)")
    parser.add_argument("--api-url", help="Use existing vLLM server URL")
    parser.add_argument("--port", type=int, default=8234, help="vLLM server port")
    parser.add_argument("--list", action="store_true", help="List models and exit")
    args = parser.parse_args()

    EVALSCOPE_OUT.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("Models to evaluate:")
        for abbr, path in get_model_list():
            exists = "OK" if not path.startswith("/") or Path(path).is_dir() else "NOT FOUND"
            print(f"  [{exists}] {abbr}")
            print(f"         {path}")
        return

    if args.model_path:
        # Single model mode
        name = args.model_name or Path(args.model_path).name
        evaluate_model(name, args.model_path, api_url=args.api_url, port=args.port)
    else:
        # Batch mode: all models
        models = get_model_list()
        print(f"EvalScope Tool-Use Evaluation (BFCL + ToolBench)")
        print(f"Models: {len(models)}")
        print(f"Output: {EVALSCOPE_OUT}/\n")

        for abbr, model_path in models:
            try:
                evaluate_model(abbr, model_path, port=args.port)
            except Exception as e:
                print(f"  ERROR: {abbr}: {e}")
                continue

    print(f"\nAll evaluations complete!")
    print(f"Results: {EVALSCOPE_OUT}/")


if __name__ == "__main__":
    main()
