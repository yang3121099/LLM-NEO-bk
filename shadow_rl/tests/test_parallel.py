#!/usr/bin/env python3
"""The multi-GPU runner has to explain its failures.

Eight workers failing identically used to print eight bare "FAIL" lines and a
list of log paths at the very end. That is the worst possible output: the run
looks broken in a way that says nothing about why, and the one interesting piece
of information -- the exception -- is buried in a log full of vLLM progress bars.

These check the parsing that turns a worker log into a one-line cause, and the
grouping that collapses N identical failures into one problem.

Run:  python shadow_rl/tests/test_parallel.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import run_eval_parallel as R  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def write(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


VLLM_NOISE = "\n".join(
    [f"INFO {i:02d}-01 loading weights... {i * 5}%" for i in range(20)]
    + ["Processed prompts:  37%|###   | 74/200 [00:12<00:20]"]
)

TRACEBACK_LOG = VLLM_NOISE + """
Traceback (most recent call last):
  File "/repo/shadow_rl/evaluate.py", line 411, in main
    qa_em = load_qa_em(args.search_r1_root)
  File "/repo/shadow_rl/evaluate.py", line 106, in load_qa_em
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
ModuleNotFoundError: No module named 'vllm'
"""

OOM_LOG = VLLM_NOISE + """
INFO 01-01 engine starting
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

REFUSED_LOG = VLLM_NOISE + """
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000):
Max retries exceeded with url: /retrieve (Connection refused)
"""


def main():
    tmp = tempfile.mkdtemp(prefix="shadow-parallel-")

    print("\n[1] a traceback is preferred over anything else in the log")
    lines = R.extract_error(write(tmp, "tb.log", TRACEBACK_LOG))
    check("starts at the traceback", lines[0].startswith("Traceback"), lines[0][:40])
    check("ends with the exception",
          lines[-1] == "ModuleNotFoundError: No module named 'vllm'", lines[-1])
    check("progress bars excluded",
          not any("Processed prompts" in l for l in lines))

    print("\n[2] a long traceback keeps both ends")
    long_tb = ("Traceback (most recent call last):\n"
               + "".join(f'  File "f{i}.py", line {i}, in g\n    call()\n' for i in range(40))
               + "RuntimeError: boom\n")
    lines = R.extract_error(write(tmp, "long.log", long_tb), max_lines=10)
    check("respects max_lines", len(lines) <= 10, str(len(lines)))
    check("keeps the head", lines[0].startswith("Traceback"))
    check("keeps the exception", lines[-1] == "RuntimeError: boom", lines[-1])
    check("says what it dropped", any("omitted" in l for l in lines))

    print("\n[3] error lines are found without a traceback")
    lines = R.extract_error(write(tmp, "oom.log", OOM_LOG))
    check("finds the OOM", any("out of memory" in l for l in lines), str(lines[-1:]))
    lines = R.extract_error(write(tmp, "refused.log", REFUSED_LOG))
    check("finds the refused connection",
          any("Connection refused" in l or "Max retries" in l for l in lines))

    print("\n[4] a log with nothing but noise still yields the tail")
    lines = R.extract_error(write(tmp, "noise.log", VLLM_NOISE))
    check("returns something", bool(lines))
    check("it is the tail", "Processed prompts" in lines[-1], lines[-1][:40])

    print("\n[5] an empty log says what an empty log means")
    # This is the OOM-killer case: the process dies without writing anything, and
    # "FAIL, log is empty" is a real diagnosis, not a shrug.
    lines = R.extract_error(write(tmp, "empty.log", ""))
    check("explains the emptiness", "killed" in " ".join(lines).lower(), lines[0][:60])
    lines = R.extract_error(os.path.join(tmp, "does-not-exist.log"))
    check("missing file does not raise", bool(lines))

    print("\n[6] identical failures share a signature, different ones do not")
    a = R.failure_signature(R.extract_error(os.path.join(tmp, "tb.log")))
    b = R.failure_signature(R.extract_error(os.path.join(tmp, "long.log")))
    c = R.failure_signature(R.extract_error(os.path.join(tmp, "tb.log")))
    check("stable for the same failure", a == c, a)
    check("differs across failures", a != b, f"{a!r} vs {b!r}")
    check("is the exception line", a == "ModuleNotFoundError: No module named 'vllm'", a)
    check("never a File line", not b.startswith('File "'), b)

    print("\n[7] the CLI exposes the robustness controls")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_eval_parallel.py"),
                          "--help"], capture_output=True, text=True).stdout
    for flag in ("--fail-fast", "--no-canary", "--timeout", "--stagger"):
        check(f"{flag} documented", flag in out)

    print("\n[8] preflight refuses a shadow role with no model")
    # Cheap to check, fatal to every worker: without this the run spends minutes
    # per GPU discovering the same missing directory.
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "run_eval_parallel.py"),
                           "--pair", "ppo-nosearch-3b-v0.2", "--roles", "shadow",
                           "--out", os.path.join(tmp, "r.csv")],
                          capture_output=True, text=True)
    check("exits non-zero", proc.returncode != 0, str(proc.returncode))
    check("names the cause", "shadow-path" in proc.stdout + proc.stderr,
          (proc.stdout + proc.stderr).strip().splitlines()[-1][:70])

    print("\n[9] diagnose.py groups an existing log directory")
    logs = os.path.join(tmp, "logs")
    os.makedirs(logs, exist_ok=True)
    for i in range(5):
        write(logs, f"p.role{i}.nq.log", TRACEBACK_LOG)
    write(logs, "p.shadow.nq.log", OOM_LOG)
    write(logs, "p.ok.nq.log", VLLM_NOISE + "\n[eval] wrote 1 row\n")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "diagnose.py"),
                           "--logs", logs], capture_output=True, text=True)
    out = proc.stdout
    check("exits 1 when something failed", proc.returncode == 1, str(proc.returncode))
    check("counts the distinct causes", "2 distinct cause" in out,
          out.splitlines()[0] if out else "")
    check("collapses the five identical ones", "5 log(s):" in out)
    check("does not count the clean log as a failure", "1 with no error" in out)
    check("suggests a fix", "pip install vllm" in out)
    check("names the OOM remedy", "gpu-memory-utilization" in out)

    print("\n[10] a traceback ends at its exception, not at the end of the file")
    # A server that keeps running after logging a traceback appends hundreds of
    # INFO lines. Taking the tail of the file then reports "Finished server
    # process" as the cause, which is how a JVM failure got mislabelled.
    noisy = ("INFO:     Started server process\n"
             "Traceback (most recent call last):\n"
             '  File "retrieval_server.py", line 389, in <module>\n'
             "    retriever = get_retriever(config)\n"
             "SystemError: JVM failed to start: -1\n"
             + "".join(f'INFO:     127.0.0.1:{6500 + i} - "POST /retrieve" 200 OK\n'
                       for i in range(650))
             + "INFO:     Finished server process\n")
    lines = R.extract_error(write(tmp, "noisy_tb.log", noisy))
    check("stops at the exception line",
          lines[-1] == "SystemError: JVM failed to start: -1", lines[-1])
    check("no post-traceback noise", not any("200 OK" in l for l in lines))
    check("signature is the exception",
          R.failure_signature(lines) == "SystemError: JVM failed to start: -1")

    print("\n[11] chained tracebacks are kept whole")
    chained = ("Traceback (most recent call last):\n"
               '  File "a.py", line 1, in f\n'
               "ValueError: inner\n"
               "\n"
               "During handling of the above exception, another exception occurred:\n"
               "\n"
               "Traceback (most recent call last):\n"
               '  File "b.py", line 2, in g\n'
               "RuntimeError: outer\n"
               "INFO: still running\n")
    lines = R.extract_error(write(tmp, "chained.log", chained), max_lines=30)
    check("ends at the outermost exception", lines[-1] == "RuntimeError: outer", lines[-1])
    check("keeps the inner cause", any("ValueError: inner" in l for l in lines))

    print("\n[12] retriever dependency causes have a named fix")
    logs2 = os.path.join(tmp, "logs2")
    os.makedirs(logs2, exist_ok=True)
    write(logs2, "r.faiss.log", "Traceback (most recent call last):\n"
                                '  File "retrieval_server.py", line 7, in <module>\n'
                                "    import faiss\n"
                                "ModuleNotFoundError: No module named 'faiss'\n")
    write(logs2, "r.jvm.log", noisy)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "diagnose.py"),
                          "--logs", logs2], capture_output=True, text=True).stdout
    check("faiss points at the interpreter mismatch", "different\n       python" in out
          or "different" in out)
    check("faiss suggests --check", "launch_retriever.sh --check" in out)
    check("JVM names the conda/apt trap", "CONDA_PREFIX/lib/jvm" in out)
    check("JVM offers the no-Java escape", "--retriever e5-hnsw" in out)

    print("\n[13] the faiss-GPU probe agrees with what the server needs")
    # The trap this exists for: faiss-cpu *does* define index_cpu_to_all_gpus
    # (a pure-python wrapper), so probing that attribute reports GPU support on
    # a build that has none, and retrieval_server.py then dies on
    # GpuMultipleClonerOptions. get_num_gpus() is the honest test.
    try:
        import faiss

        launcher = open(os.path.join(ROOT, "launch_retriever.sh")).read()
        check("probe uses get_num_gpus", "get_num_gpus" in launcher)
        check("probe checks GpuMultipleClonerOptions",
              "GpuMultipleClonerOptions" in launcher)
        gpu_real = getattr(faiss, "get_num_gpus", lambda: 0)() > 0
        server_ok = hasattr(faiss, "GpuMultipleClonerOptions")
        check("probe verdict matches the symbol the server calls",
              gpu_real == server_ok, f"get_num_gpus>0={gpu_real} "
                                     f"GpuMultipleClonerOptions={server_ok}")
        if not gpu_real:
            # Documents the exact discrepancy that caused the bug.
            check("index_cpu_to_all_gpus alone would have lied",
                  hasattr(faiss, "index_cpu_to_all_gpus"),
                  "present on faiss-cpu, hence unusable as a probe")
    except ImportError:
        print("     (skipped: faiss not installed)")

    print("\n[14] check_retriever distinguishes the retriever's failure modes")
    # "Uvicorn running on ..." only means a port was bound. These are the states
    # that look identical from the outside and are not.
    import http.server
    import json as _json
    import threading

    class Stub(http.server.BaseHTTPRequestHandler):
        mode = "ok"

        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path != "/retrieve":
                self.send_error(404)
                return
            req = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if Stub.mode == "empty":
                body = [[] for _ in req["queries"]]
            else:
                body = [[{"document": {"contents": f"about {q}"}, "score": 0.9}]
                        for q in req["queries"]]
            data = _json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/retrieve"
    script = os.path.join(ROOT, "check_retriever.py")

    def run_check(u):
        return subprocess.run([sys.executable, script, "--url", u, "--timeout", "20"],
                              capture_output=True, text=True)

    Stub.mode = "ok"
    proc = run_check(url)
    check("healthy server -> exit 0", proc.returncode == 0, str(proc.returncode))
    check("shows the passages", "about who wrote" in proc.stdout)

    Stub.mode = "empty"
    proc = run_check(url)
    check("empty results -> exit 1", proc.returncode == 1, str(proc.returncode))
    check("says the index is not answering",
          "returned nothing" in proc.stdout, proc.stdout.strip()[-60:])

    proc = run_check(url.replace("/retrieve", "/json"))
    check("wrong path -> exit 1", proc.returncode == 1, str(proc.returncode))
    check("explains that 404 is the server behaving",
          "only endpoint is POST /retrieve" in proc.stdout)

    proc = run_check("http://127.0.0.1:1/retrieve")
    check("nothing listening -> exit 1", proc.returncode == 1, str(proc.returncode))
    check("says how to start it", "launch_retriever.sh" in proc.stdout)
    srv.shutdown()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
