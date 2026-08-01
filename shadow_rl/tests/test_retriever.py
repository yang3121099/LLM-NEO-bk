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

# The stand-in mirrors the real retrieval_server.py in the ways the launcher
# reasons about: argparse with no --port, a FastAPI app, and a hardcoded
# uvicorn.run(app, port=8000) at the end. Serving on any other port goes through
# the launcher's shim, which works by intercepting that call -- so a stand-in
# that merely serves HTTP some other way would not exercise the same code.
SERVER = '''
import argparse
from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    # Upstream ends with `return {"result": resp}` -- the envelope matters.
    return {"result": [[{"document": {"contents": "about " + q}, "score": 0.9}]
                       for q in request.queries]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path")
    parser.add_argument("--corpus_path")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    args = parser.parse_args()
    print("loading index...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)   # hardcoded, exactly like upstream
'''

# Dies while loading the index. It still contains a uvicorn.run so the launcher
# takes the same path it would for the real thing -- the point is the crash, not
# a differently-shaped file.
CRASHING_SERVER = '''
import uvicorn
print("loading index...", flush=True)
raise RuntimeError("index file is corrupt")
uvicorn.run(None, port=8000)
'''


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bind_low_port():
    """Occupy a port below the ephemeral range, and keep it.

    Not socket(0): the kernel allocates those from ~32768 up, which is also
    where it draws *source* ports for outgoing connections. Basing the search
    there means the launcher can pick a port that looks free and then lose it to
    one of its own health-check connections a moment later -- a race that only
    exists in the test, since PORT_BASE is 8000 in real use.
    """
    for candidate in range(8300, 9000):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", candidate))
            sock.listen(1)
            return sock, candidate
        except OSError:
            sock.close()
    raise RuntimeError("no free port in 8300-8999")


def main():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: the stand-in server needs fastapi + uvicorn ({exc}).")
        return 0

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
            fh.write(source)

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
        install(SERVER)
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

        print("\n[6] --port picks a free one when the default is taken")
        # 8000 is a busy address in practice -- vLLM's OpenAI server uses it -- so
        # binding it blindly is how a run ends up talking to someone else's service.
        install(SERVER)
        blocker, busy = bind_low_port()
        env_auto = dict(env, PORT="auto", PORT_BASE=str(busy))
        proc = subprocess.run(["bash", LAUNCHER, "--retriever", "e5-hnsw", "--daemon"],
                              capture_output=True, text=True, env=env_auto, cwd=REPO,
                              timeout=180)
        check("starts anyway", proc.returncode == 0,
              (proc.stdout + proc.stderr).strip()[-140:])
        check("says it moved off the busy port", "is taken" in proc.stdout)
        urlfile = os.path.join(tmp, "logs", "retriever.url")
        check("recorded the url it chose", os.path.exists(urlfile))
        chosen = open(urlfile).read().strip() if os.path.exists(urlfile) else ""
        check("and it is not the busy one", f":{busy}/" not in chosen, chosen)

        print("\n[7] --status and --stop find that port without being told")
        proc = subprocess.run(["bash", LAUNCHER, "--status"], capture_output=True,
                              text=True, env=env_auto, cwd=REPO, timeout=60)
        check("status finds it", proc.returncode == 0,
              (proc.stdout + proc.stderr).strip()[-120:])
        check("reports the chosen url", chosen.split("/retrieve")[0] in proc.stdout,
              proc.stdout.strip()[-80:])
        subprocess.run(["bash", LAUNCHER, "--stop"], capture_output=True, env=env_auto,
                       cwd=REPO, timeout=60)

        print("\n[8] an explicit --port that is occupied is refused, not guessed at")
        proc = subprocess.run(["bash", LAUNCHER, "--retriever", "e5-hnsw", "--daemon",
                               "--port", str(busy)],
                              capture_output=True, text=True, env=env, cwd=REPO,
                              timeout=180)
        check("exits non-zero", proc.returncode != 0, str(proc.returncode))
        check("names the conflict", "in use" in proc.stderr,
              proc.stderr.strip().splitlines()[0][:70] if proc.stderr else "")
        blocker.close()

        print("\n[9] a server that dies while loading fails loudly")
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
