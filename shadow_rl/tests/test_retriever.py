#!/usr/bin/env python3
"""The retriever has to stay up, and has to say so honestly.

Run in the foreground, the server dies with its shell -- Ctrl-C, a closed
terminal, a dropped SSH session -- and the next command reports "Connection
refused" with no hint that the process simply went away. --daemon exists to make
that not happen, and the lifecycle around it (pidfile, status, stop, and failing
loudly when the server dies while loading) is easy to break silently.

These drive launch_retriever.sh against a stand-in Search-R1 checkout: same
layout, same API, no 60GB index.

Run:  python shadow_rl/tests/test_retriever.py
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # shadow_rl/
REPO = os.path.dirname(ROOT)
LAUNCHER = os.path.join(ROOT, "launch_retriever.sh")

FAILURES = []

HEALTHY_SERVER = '''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--topk") + 2]) if False else PORT
print("loading index...", flush=True)
time.sleep(1)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        if self.path != "/retrieve":
            self.send_error(404); return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps([[{"document": {"contents": "about " + q}, "score": 0.9}]
                           for q in req["queries"]]).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
print("Uvicorn running", flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
'''

CRASHING_SERVER = '''
print("loading index...", flush=True)
raise RuntimeError("index file is corrupt")
'''


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    tmp = tempfile.mkdtemp(prefix="shadow-retriever-")
    port = free_port()
    sr1 = os.path.join(tmp, "Search-R1", "search_r1", "search")
    os.makedirs(sr1)
    corpus = os.path.join(tmp, "corpus")
    os.makedirs(corpus)
    # The launcher only checks these exist before serving.
    open(os.path.join(corpus, "wiki-18.jsonl"), "w").close()
    open(os.path.join(corpus, "e5_HNSW64.index"), "w").close()

    server_py = os.path.join(sr1, "retrieval_server.py")

    def install(source):
        with open(server_py, "w") as fh:
            fh.write(f"PORT = {port}\n" + source)

    env = dict(os.environ,
               SEARCH_R1_ROOT=os.path.join(tmp, "Search-R1"),
               CORPUS_DIR=corpus,
               LOG_DIR=os.path.join(tmp, "logs"),
               PORT=str(port))

    def run(*args, timeout=180):
        return subprocess.run(["bash", LAUNCHER, *args], capture_output=True,
                              text=True, env=env, cwd=REPO, timeout=timeout)

    try:
        print("\n[1] --status and --stop are safe when nothing is running")
        proc = run("--status")
        check("status exits 1", proc.returncode == 1, str(proc.returncode))
        check("says there is no process", "no retriever process" in proc.stderr)
        proc = run("--stop")
        check("stop exits 0", proc.returncode == 0, str(proc.returncode))
        check("says there was nothing to stop", "nothing to stop" in proc.stderr)

        print("\n[2] --daemon waits for the server to actually answer")
        install(HEALTHY_SERVER)
        proc = run("--retriever", "e5-hnsw", "--daemon")
        check("exits 0", proc.returncode == 0,
              (proc.stdout + proc.stderr).strip()[-160:])
        check("reports the pid and url", "retriever up" in proc.stdout)
        pidfile = os.path.join(tmp, "logs", "retriever.pid")
        check("wrote a pidfile", os.path.exists(pidfile))

        print("\n[3] it outlives the shell that started it")
        # The whole point of setsid: the launcher has already exited above, and
        # the server is still there.
        time.sleep(1)
        proc = run("--status")
        check("still running", proc.returncode == 0,
              (proc.stdout + proc.stderr).strip()[-120:])
        check("reports answering", "answering" in proc.stdout)

        print("\n[4] starting again is a no-op, not a second server")
        proc = run("--retriever", "e5-hnsw", "--daemon")
        check("exits 0", proc.returncode == 0)
        check("says it is already up", "already running" in proc.stdout)

        print("\n[5] --stop really stops it")
        proc = run("--stop")
        check("exits 0", proc.returncode == 0)
        check("removed the pidfile", not os.path.exists(pidfile))
        proc = run("--status")
        check("status now exits 1", proc.returncode == 1, str(proc.returncode))

        print("\n[6] a server that dies while loading fails loudly")
        # Otherwise --daemon returns success and the failure surfaces much later
        # as a refused connection from the evaluation.
        install(CRASHING_SERVER)
        proc = run("--retriever", "e5-hnsw", "--daemon")
        check("exits non-zero", proc.returncode != 0, str(proc.returncode))
        check("prints the server's own error",
              "index file is corrupt" in proc.stderr,
              proc.stderr.strip().splitlines()[-1][:70] if proc.stderr else "")
        check("does not leave a stale pidfile", not os.path.exists(pidfile))
    finally:
        subprocess.run(["bash", LAUNCHER, "--stop"], capture_output=True, env=env,
                       cwd=REPO)
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
