#!/usr/bin/env python3
"""Ask the retrieval server a real question and show what comes back.

"Uvicorn running on http://0.0.0.0:8000" only means the process bound a port.
It says nothing about whether the index loaded usefully, whether queries return
passages, or whether a query takes 200ms or 200 seconds -- and the last one
decides whether an evaluation is feasible at all.

    python shadow_rl/check_retriever.py
    python shadow_rl/check_retriever.py --url http://127.0.0.1:8000/retrieve
    python shadow_rl/check_retriever.py --query "who wrote hamlet" --topk 5

Exit status is 0 only if the server answered with usable passages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Questions with unmistakable answers, so a wrong-looking result is obvious
# rather than merely unfamiliar.
DEFAULT_QUERIES = [
    "who wrote the play hamlet",
    "what is the capital of france",
]


def default_url() -> str:
    """The URL the launcher actually chose, not a guess.

    The port is picked at start-up (8000 is often taken -- vLLM's own server
    uses it), so hardcoding it here would send the check somewhere else.
    RETRIEVER_URL still wins, for a server on another host entirely.
    """
    env = os.environ.get("RETRIEVER_URL")
    if env:
        return env
    recorded = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "logs", "retriever.url")
    try:
        with open(recorded) as fh:
            url = fh.read().strip()
        if url:
            return url
    except OSError:
        pass
    return "http://127.0.0.1:8000/retrieve"


def post(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body), time.time() - started


def passages(result) -> list:
    """Normalise the several shapes the server returns per query."""
    out = []
    for item in result:
        if isinstance(item, dict):
            # return_scores=True wraps each hit as {"document": {...}, "score": x}
            doc = item.get("document", item)
            if isinstance(doc, dict):
                out.append(doc.get("contents") or doc.get("text") or json.dumps(doc)[:200])
            else:
                out.append(str(doc)[:200])
        else:
            out.append(str(item)[:200])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=default_url(),
                    help="default: whatever launch_retriever.sh recorded in "
                         "logs/retriever.url, else :8000")
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300,
                    help="seconds; a flat index on CPU really can take minutes")
    ap.add_argument("--quiet", action="store_true", help="verdict only")
    args = ap.parse_args()

    queries = args.query or DEFAULT_QUERIES
    print(f"POST {args.url}")
    print(f"  queries: {queries}  topk={args.topk}\n")

    try:
        result, elapsed = post(args.url, {"queries": queries, "topk": args.topk,
                                          "return_scores": True}, args.timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode(errors="replace")
        print(f"[fail] HTTP {exc.code} from the server")
        print(f"       {body}")
        if exc.code == 404:
            print("       The only endpoint is POST /retrieve. A 404 on some other\n"
                  "       path is the server working correctly and something else\n"
                  "       asking for the wrong thing.")
        elif exc.code == 422:
            print("       422 means the payload shape is wrong; it must be\n"
                  '       {"queries": [...], "topk": N, "return_scores": bool}')
        return 1
    except urllib.error.URLError as exc:
        print(f"[fail] could not reach it: {exc.reason}")
        print("       Nothing is listening, or it is on another host/port.")
        print("       Start it:  ./shadow_rl/launch_retriever.sh")
        print("       Check the port:  ss -ltnp | grep 8000")
        return 1
    except TimeoutError:
        print(f"[fail] no answer within {args.timeout:g}s.")
        print("       A flat e5 index searched on the CPU behaves exactly like this:\n"
              "       every query scans the whole 21M-passage index. Either give\n"
              "       faiss a GPU (pip uninstall -y faiss-cpu && pip install faiss-gpu)\n"
              "       or use the index built for CPU: --retriever e5-hnsw")
        return 1

    # retrieval_server.py ends with `return {"result": resp}` -- the list of
    # per-query hits is wrapped. evaluate.py already unwraps it (`.json()["result"]`);
    # this checker did not, and so called a perfectly healthy server broken.
    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    # One list of hits per query.
    if not isinstance(result, list) or len(result) != len(queries):
        print(f"[fail] unexpected response shape: {type(result).__name__}, "
              f"{len(result) if hasattr(result, '__len__') else '?'} item(s) "
              f"for {len(queries)} query/queries")
        print(f"       {json.dumps(result)[:400]}")
        print("       Expected either [[hit, ...], ...] or {\"result\": [[hit, ...], ...]},\n"
              "       one list per query. If the server changed shape, evaluate.py's\n"
              "       Retriever._to_string needs the same update.")
        return 1

    empty = 0
    for query, hits in zip(queries, result):
        texts = passages(hits)
        if not texts:
            empty += 1
        if args.quiet:
            continue
        print(f"  {query!r} -> {len(texts)} passage(s)")
        for text in texts[:args.topk]:
            snippet = " ".join(str(text).split())[:160]
            print(f"      - {snippet}")
        print()

    per_query = elapsed / max(1, len(queries))
    print(f"[info] {elapsed:.2f}s for {len(queries)} query/queries "
          f"({per_query:.2f}s each)")

    if empty:
        print(f"[fail] {empty} of {len(queries)} queries returned nothing.")
        print("       The server is up but the index is not answering -- check that\n"
              "       --index_path and --corpus_path point at matching files.")
        return 1

    # An evaluation issues on the order of 10k-100k queries. At a second each
    # that is hours of pure retrieval, which is worth knowing before starting.
    if per_query > 1.0:
        print(f"[warn] {per_query:.1f}s per query is slow. A full evaluation issues\n"
              "       tens of thousands of them, so this would dominate the run.\n"
              "       Usually a flat index being searched on the CPU: either give\n"
              "       faiss a GPU, or switch to --retriever e5-hnsw.")

    print("[ok]   the retriever is answering with passages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
