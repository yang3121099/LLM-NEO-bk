#!/usr/bin/env python3
"""
EvalScope tool-use evaluation script (BFCL + ToolBench).

Model list uses the SAME (abbr, path) tuple format as OpenCompass eval configs,
so you can copy model entries directly between scripts.

Prerequisites:
    pip install 'evalscope[all]' vllm

Usage:
    # Evaluate all models:
    python3 scripts/eval_tooluse_evalscope.py

    # Evaluate a single model:
    python3 scripts/eval_tooluse_evalscope.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --model-name Llama-3.1-8B-Instruct

    # Use an existing vLLM server:
    python3 scripts/eval_tooluse_evalscope.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --api-url http://localhost:8000/v1

    # List models only:
    python3 scripts/eval_tooluse_evalscope.py --list
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (same layout as generate_dataset_scripts.sh)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = _SCRIPT_DIR.parent
RESULTS_DIR = WORKSPACE_DIR / "results"
EVALSCOPE_OUT = RESULTS_DIR / "evalscope"

# ---------------------------------------------------------------------------
# Auto-detect GPU count (same logic as OpenCompass eval configs)
# ---------------------------------------------------------------------------
_NUM_GPUS = 1
try:
    _NUM_GPUS = max(1, len(
        subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
    ))
except Exception:
    _NUM_GPUS = 1

# ---------------------------------------------------------------------------
# Model list — SAME (abbr, path) format as OpenCompass eval configs
#
# Copy entries directly from eval_2k_*.py or eval_full_*.py.
# For local merged models, use $RESULTS_DIR which is resolved automatically.
# ---------------------------------------------------------------------------

# ======= Original Instruct baselines =======
HF_baselines = [
    ('Llama-3.1-8B-Instruct-hf', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('Qwen3-8B-Instruct-hf', 'Qwen/Qwen3-8B'),
]

# ======= Trained models (B2I Shadow-FT, I2I baseline) =======
# 2k experiments
Baseline_settings = [
    ('Llama-3.1-8B-dolci_mix-2k-B2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I'),
    ('Llama-3.1-8B-dolci_mix-2k-I2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I'),
    ('Llama-3.1-8B-nemotron_if-2k-B2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I'),
    ('Llama-3.1-8B-nemotron_if-2k-I2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I'),
    ('Qwen3-8B-dolci_mix-2k-B2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-2k-lora-rank128-lr0.0002-dolci_mix/merged-B2I'),
    ('Qwen3-8B-dolci_mix-2k-I2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-2k-lora-rank128-lr0.0002-dolci_mix/merged-I2I'),
    ('Qwen3-8B-nemotron_if-2k-B2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-2k-lora-rank128-lr0.0002-nemotron_if/merged-B2I'),
    ('Qwen3-8B-nemotron_if-2k-I2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-2k-lora-rank128-lr0.0002-nemotron_if/merged-I2I'),
    # Tool-use experiments (dolci_instruct_tool_use, 2k samples)
    ('Llama-3.1-8B-dolci_tool-2k-B2I', '$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I'),
    ('Llama-3.1-8B-dolci_tool-2k-I2I', '$RESULTS_DIR/0330/result-Llama-3.1-8B-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I'),
    ('Qwen3-8B-dolci_tool-2k-B2I', '$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/B-2k-lora-rank128-lr0.0002-dolci_tool/merged-B2I'),
    ('Qwen3-8B-dolci_tool-2k-I2I', '$RESULTS_DIR/0330/result-Qwen3-8B-Base-0330/I-2k-lora-rank128-lr0.0002-dolci_tool/merged-I2I'),
    # Full-scale experiments (uncomment after training completes)
    # ('Llama-3.1-8B-openr1-220k-B2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-220k-lora-rank128-lr0.0002-openr1/merged-B2I'),
    # ('Llama-3.1-8B-openr1-220k-I2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-220k-lora-rank128-lr0.0002-openr1/merged-I2I'),
    # ('Llama-3.1-8B-deepmath-309k-B2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/B-309k-lora-rank128-lr0.0002-deepmath/merged-B2I'),
    # ('Llama-3.1-8B-deepmath-309k-I2I', '$RESULTS_DIR/0326/result-Llama-3.1-8B-0326/I-309k-lora-rank128-lr0.0002-deepmath/merged-I2I'),
    # ('Qwen3-8B-openr1-220k-B2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-220k-lora-rank128-lr0.0002-openr1/merged-B2I'),
    # ('Qwen3-8B-openr1-220k-I2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-220k-lora-rank128-lr0.0002-openr1/merged-I2I'),
    # ('Qwen3-8B-deepmath-309k-B2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/B-309k-lora-rank128-lr0.0002-deepmath/merged-B2I'),
    # ('Qwen3-8B-deepmath-309k-I2I', '$RESULTS_DIR/0326/result-Qwen3-8B-0326/I-309k-lora-rank128-lr0.0002-deepmath/merged-I2I'),
]


def resolve_path(path_str):
    """Resolve $RESULTS_DIR in path strings (same as OpenCompass configs)."""
    return path_str.replace("$RESULTS_DIR", str(RESULTS_DIR))


def get_all_models():
    """Return list of (abbr, resolved_path) tuples."""
    models = []
    for abbr, path in HF_baselines:
        models.append((abbr, path))
    for abbr, path in Baseline_settings:
        models.append((abbr, resolve_path(path)))
    return models


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def _get_tool_call_parser(model_path):
    """Determine the vLLM tool-call parser based on model family."""
    model_lower = model_path.lower()
    if "llama" in model_lower:
        return "llama3_json"
    elif "qwen" in model_lower:
        return "hermes"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        return "hermes"  # reasonable default


def start_vllm_server(model_path, port=8234, tp=None, gpu_util=0.9, max_len=8192):
    """Start a vLLM OpenAI-compatible server, return (process, port)."""
    if tp is None:
        tp = _NUM_GPUS

    tool_parser = _get_tool_call_parser(model_path)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
        "--trust-remote-code",
        "--dtype", "auto",
        "--enable-auto-tool-choice",
        "--tool-call-parser", tool_parser,
    ]
    log_path = EVALSCOPE_OUT / "vllm_server.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

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


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_bfcl_eval(model_name, api_url, out_dir):
    """Run BFCL evaluation via evalscope."""
    print("  Running BFCL evaluation ...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model=model_name,
            api_url=api_url,
            eval_type="server",
            datasets=["bfcl_v3"],
            work_dir=str(out_dir),
        )
        results = run_task(task_cfg)
        print(f"  BFCL results: {results}")
        return results
    except ImportError:
        print("  evalscope not available for BFCL.")
        print("  Install: pip install 'evalscope[all]' bfcl-eval==2025.10.27.1")
    except Exception as e:
        print(f"  BFCL eval error: {e}")
        return None


def run_toolbench_eval(model_name, api_url, out_dir):
    """Run ToolBench evaluation via evalscope."""
    print("  Running ToolBench evaluation ...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model=model_name,
            api_url=api_url,
            eval_type="server",
            datasets=["tool_bench"],
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
    """Evaluate a single model on BFCL + ToolBench.

    Restarts vLLM server between benchmarks to avoid OOM / KV cache
    fragmentation on long-running evaluations.
    """
    out_dir = EVALSCOPE_OUT / abbr

    if model_path.startswith("/") and not Path(model_path).is_dir():
        print(f"  SKIP: {abbr} — path not found: {model_path}")
        return

    print(f"\n{'=' * 64}")
    print(f"  Model: {abbr}")
    print(f"  Path:  {model_path}")
    print(f"  GPUs:  {_NUM_GPUS} (tp={_NUM_GPUS})")
    print(f"{'=' * 64}")

    benchmarks = [
        ("bfcl", run_bfcl_eval),
        ("toolbench", run_toolbench_eval),
    ]

    for bench_name, bench_fn in benchmarks:
        server_proc = None
        try:
            if api_url is None:
                server_proc, port = start_vllm_server(model_path, port=port)
                api_url_used = f"http://localhost:{port}/v1"
            else:
                api_url_used = api_url

            bench_fn(model_path, api_url_used, out_dir / bench_name)
        except Exception as e:
            print(f"  ERROR in {bench_name}: {e}")
        finally:
            if server_proc:
                print(f"  Stopping vLLM server (after {bench_name}) ...")
                stop_vllm_server(server_proc)
                time.sleep(5)  # give GPU memory time to free

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
        print(f"Models to evaluate (GPUs={_NUM_GPUS}):")
        for abbr, path in get_all_models():
            exists = "OK" if not path.startswith("/") or Path(path).is_dir() else "NOT FOUND"
            print(f"  [{exists}] {abbr}")
            print(f"         {path}")
        return

    if args.model_path:
        name = args.model_name or Path(args.model_path).name
        evaluate_model(name, args.model_path, api_url=args.api_url, port=args.port)
    else:
        models = get_all_models()
        print(f"EvalScope Tool-Use Evaluation (BFCL + ToolBench)")
        print(f"Models: {len(models)}, GPUs: {_NUM_GPUS}")
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
