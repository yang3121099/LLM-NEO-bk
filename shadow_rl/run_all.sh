#!/usr/bin/env bash
# One-click driver: similarity -> merge -> evaluate -> aggregate, over any subset
# of the 12 pairs.
#
#   ./shadow_rl/run_all.sh --pairs nosearch              # start here, no retrieval needed
#   ./shadow_rl/run_all.sh --pairs all --cleanup         # everything, freeing disk as it goes
#   ./shadow_rl/run_all.sh --pairs ppo-nosearch-3b-v0.2 --stages similarity
#   ./shadow_rl/run_all.sh --fast --yes                  # start here
#   ./shadow_rl/run_all.sh --pairs all --sample 500 --cleanup --auto-retriever --yes
#   ./shadow_rl/run_all.sh --pairs nosearch --limit 200  # quick smoke run
#
# Every stage is resumable: finished work is detected and skipped, so re-running
# after an interruption picks up where it stopped. Each pair is isolated -- one
# failure is logged and the run continues to the next.
#
# Options
#   --pairs X      groups: all demo nosearch search grpo ppo 3b 7b 14b qwen
#                          llama v0.1 v0.2 v0.3 latest
#                  version groups union: 'v0.1,v0.3' = all v0.1 + v0.3 pairs
#                  other groups intersect: 'v0.3,grpo' = the v0.3 GRPO pairs
#                  combined: 'v0.1,v0.3,grpo' = (v0.1 OR v0.3) AND grpo
#                  a leading - excludes: 'all,-v0.2' = everything but v0.2
#                  or explicit ids: <pair_id>[,<pair_id>...]
#   --stages X     comma list of: similarity,merge,eval,aggregate
#                  default 'merge,eval,aggregate' -- similarity (per-parameter
#                  sigma) is diagnostic and costs a full pass over four
#                  checkpoints per pair, so ask for it explicitly.
#   --roles X      comma list of roles to evaluate  (default all four)
#   --limit N      first N questions per dataset; biased, smoke runs only
#   --sample N     deterministic random N per dataset, identical across roles.
#                  Recommended for multi-pair sweeps: the full test sets are
#                  ~51,700 questions per role, and --pairs all over them is on
#                  the order of two weeks of single-GPU time.
#   --datasets X   evaluate only these datasets (comma list)
#   --skip-datasets X comma list of datasets to omit entirely
#   --auto-retriever  start and stop the retrieval server automatically
#   --port N       retrieval server port. Default: whatever the server chose and
#                  recorded in logs/retriever.url, else the first free port from
#                  8000 up. 8000 is popular -- vLLM's OpenAI server uses it.
#   --retriever X  auto (default) | e5 | e5-hnsw | bm25. 'auto' picks the dense
#                  e5 index -- exact on GPU if faiss-gpu is installed, the
#                  approximate HNSW one on CPU otherwise. Only bm25 needs a JVM,
#                  and it is never chosen for you.
#   --cleanup      delete a pair's RL checkpoints and merged model once it is evaluated
#   --tp N         GPUs per worker (default 1; a 3B/7B model needs only 1)
#   --jobs N       concurrent eval workers, one GPU group each.
#                  'auto' (default) = visible GPUs / tp, so 8 GPUs run 8
#                  (role, dataset) cells at once instead of one at a time.
#   --fast         smallest useful run: one 3B pair, 4 datasets (2 in-domain +
#                  2 OOD), 200 questions, all five models. A few minutes.
#   --force        redo work that is already complete (merge, smoke test)
#   --fail-fast N  stop after N consecutive cell failures (default 3, 0 = never).
#                  Multi-GPU breakage is almost always systematic; reproducing
#                  it on every remaining cell wastes hours and tells you nothing.
#   --no-canary    fan out immediately instead of proving one cell works first
#   --stagger SEC  delay between worker starts (default 15). Simultaneous vLLM
#                  engine inits contend for ports, the HF cache lock and host RAM.
#   --job-timeout MIN  kill any single cell that runs longer than this
#   --skip-env-check  do not run the environment check at all
#   --strict-env      abort if the environment check reports problems
#   --dry-run      print the plan and exit
#   --yes          skip the confirmation prompt
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults -------------------------------------------------------------- #
PAIRS_SEL="nosearch"
# similarity is NOT in the default: it streams four whole checkpoints per pair
# to compute per-parameter sigma, which is diagnostic rather than part of the
# result. Ask for it explicitly with --stages similarity,merge,eval,aggregate.
STAGES="merge,eval,aggregate"
ROLES="base_baseline,instruct_baseline,rl_on_instruct,rl_on_base,shadow"
FAST=0
LIMIT=""
SAMPLE=""
SKIP_DATASETS=""
DATASETS_SEL=""
CLEANUP=0
AUTO_RETRIEVER=0
RETRIEVER_PID=""
# Which index to serve. Dense by default: bm25 is the only option that drags in
# pyserini -> Lucene -> a JVM, and e5 is what Search-R1 itself used, so the
# absolute numbers are comparable to the published ones.
RETRIEVER="${RETRIEVER:-auto}"
TP="${TP:-1}"
JOBS="auto"
# Multi-GPU robustness. Most breakage on a multi-GPU box hits every worker
# identically, so the default is to prove one job works before fanning out and
# to stop after a few identical failures rather than reproduce them N times.
FAIL_FAST="${FAIL_FAST:-3}"
STAGGER="${STAGGER:-15}"
JOB_TIMEOUT="${JOB_TIMEOUT:-}"
NO_CANARY=0
DRY=0
ASSUME_YES=0
FORCE=0
SKIP_ENV_CHECK=0
STRICT_ENV=0

source "$REPO_ROOT/shadow_rl/paths.sh"
SEARCH_R1_ROOT="$(shadow_rl_search_r1_root)"
MERGED_DIR="${MERGED_DIR:-$REPO_ROOT/shadow_rl/merged}"
RESULTS="${RESULTS:-$REPO_ROOT/shadow_rl/results.csv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/shadow_rl/logs}"
# The retriever picks its own port (8000 is commonly taken -- vLLM's server uses
# it) and records the URL. Read that rather than assuming, so a server started
# separately is found without anything being kept in sync by hand.
if [[ -z "${RETRIEVER_URL:-}" && -f "$LOG_DIR/retriever.url" ]]; then
    RETRIEVER_URL="$(cat "$LOG_DIR/retriever.url")"
fi
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_PORT=""
# Models go to the standard HuggingFace cache by default, so they are shared
# with anything else on the machine rather than duplicated. Only export the
# override when the caller actually set one.
if [[ -n "${SHADOW_RL_MODEL_DIR:-}" ]]; then
    MODEL_DIR="$SHADOW_RL_MODEL_DIR"
    export SHADOW_RL_MODEL_DIR
else
    MODEL_DIR=$(python3 -c "
try:
    from huggingface_hub.constants import HF_HUB_CACHE
    print(HF_HUB_CACHE)
except Exception:
    import os
    print(os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub'))
" 2>/dev/null || echo "$HOME/.cache/huggingface/hub")
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pairs)   PAIRS_SEL="$2"; shift 2 ;;
        --stages)  STAGES="$2";    shift 2 ;;
        --roles)   ROLES="$2";     shift 2 ;;
        --limit)   LIMIT="$2";     shift 2 ;;
        --sample)  SAMPLE="$2";    shift 2 ;;
        --skip-datasets) SKIP_DATASETS="$2"; shift 2 ;;
        --datasets) DATASETS_SEL="$2"; shift 2 ;;
        --auto-retriever) AUTO_RETRIEVER=1; shift ;;
        --retriever) RETRIEVER="$2"; shift 2 ;;
        --port)      RETRIEVER_PORT="$2"; shift 2 ;;
        --tp)      TP="$2";        shift 2 ;;
        --jobs)    JOBS="$2";      shift 2 ;;
        --fail-fast)  FAIL_FAST="$2";   shift 2 ;;
        --stagger)    STAGGER="$2";     shift 2 ;;
        --job-timeout) JOB_TIMEOUT="$2"; shift 2 ;;
        --no-canary)  NO_CANARY=1;      shift ;;
        --cleanup) CLEANUP=1;      shift ;;
        --fast)    FAST=1;         shift ;;
        --force)   FORCE=1;        shift ;;
        --skip-env-check) SKIP_ENV_CHECK=1; shift ;;
        --strict-env)     STRICT_ENV=1;     shift ;;
        --dry-run) DRY=1;          shift ;;
        --yes|-y)  ASSUME_YES=1;   shift ;;
        -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# An explicit --port overrides whatever URL was recorded earlier; otherwise a
# stale logs/retriever.url would silently win over the port just asked for.
if [[ -n "$RETRIEVER_PORT" ]]; then
    RETRIEVER_URL="http://127.0.0.1:$RETRIEVER_PORT/retrieve"
fi

# --fast: one 3B pair, two in-domain datasets, 200 sampled questions, and the
# five models the question is actually about. Enough to prove the pipeline runs
# and to see the sign of the effect; not enough to publish.
if [[ $FAST -eq 1 ]]; then
    [[ "$PAIRS_SEL" == "nosearch" ]] && PAIRS_SEL="ppo-nosearch-3b-v0.2"
    [[ -z "$SAMPLE" && -z "$LIMIT" ]] && SAMPLE=200
    :   # --fast leaves the stage list alone; similarity is already opt-in
    # nq + hotpotqa are in-domain; musique + bamboogle are out-of-domain and
    # small (2.4k and 125 rows), so they add generalisation signal almost free.
    # Skipped: triviaqa/popqa/2wiki, the three largest, which say little that
    # the cheaper OOD sets do not.
    SKIP_DATASETS="${SKIP_DATASETS:-triviaqa,popqa,2wikimultihopqa}"
fi

if [[ -n "$DATASETS_SEL" ]]; then
    SKIP_DATASETS=$(python3 -c "
import sys; sys.path.insert(0,'shadow_rl')
from pairs import DATASETS
want={d.strip() for d in '$DATASETS_SEL'.split(',') if d.strip()}
bad=want-set(DATASETS)
if bad:
    sys.exit('unknown dataset(s): '+', '.join(sorted(bad)))
print(','.join(d for d in DATASETS if d not in want))") \
        || { echo "[fail] bad --datasets" >&2; exit 1; }
fi

# Which commit is actually running. A long session accumulates pulls, and
# "did that fix land in what I am running?" is otherwise unanswerable from the
# output alone -- the symptom being a run that behaves like an older version.
GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=""
git -C "$REPO_ROOT" diff --quiet 2>/dev/null || GIT_DIRTY=" (uncommitted changes)"

mkdir -p "$LOG_DIR" "$MERGED_DIR" "$MODEL_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

log()  { printf '\033[1;34m[run]\033[0m  %s\n'  "$*" | tee -a "$RUN_LOG"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n'  "$*" | tee -a "$RUN_LOG"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" | tee -a "$RUN_LOG" >&2; }
err()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" | tee -a "$RUN_LOG" >&2; }
die()  { err "$*"; exit 1; }

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# A merged model counts as done only if the merge ran to completion: merge.py
# writes shadow_merge_stats.json last, after the shards and the config, and every
# shard named in the index must actually exist. Checking only for config.json
# would happily reuse a model left half-written by an interrupted run.
merged_complete() {
    local dir="$1"
    [[ -f "$dir/shadow_merge_stats.json" ]] || return 1
    python3 - "$dir" <<'MERGEDPY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    shards = set(json.load(open(idx))["weight_map"].values())
else:
    shards = {f for f in os.listdir(d) if f.endswith(".safetensors")}
    if not shards:
        sys.exit(1)
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
sys.exit(1 if missing else 0)
MERGEDPY
}

# The smoke test loads a 7B model; no reason to repeat it once it has passed for
# a merged model that has not changed since.
smoke_passed() {
    local dir="$1"
    [[ -f "$dir/.smoke_ok" ]] || return 1
    [[ "$dir/.smoke_ok" -nt "$dir/shadow_merge_stats.json" ]]
}

# ---- resolve the pair selection ------------------------------------------- #
# Command substitution, not a process substitution into mapfile: mapfile reports
# its own exit status, so a failure in the resolver would be silently swallowed.
PAIR_LIST_RAW=$(python3 - "$PAIRS_SEL" <<'PY'
import sys, os
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS, PAIRS_BY_ID

sel = sys.argv[1]
groups = {
    "all":      lambda p: True,
    "nosearch": lambda p: not p.with_search,
    "search":   lambda p: p.with_search,
    "grpo":     lambda p: p.algo == "grpo",
    "ppo":      lambda p: p.algo == "ppo",
    "3b":       lambda p: p.size == "3b",
    "7b":       lambda p: p.size == "7b",
    "14b":      lambda p: p.size == "14b",
    "qwen":     lambda p: not p.size.startswith("llama"),
    "llama":    lambda p: p.size.startswith("llama"),
    # Version tags: v0.3 is the current recipe, so "latest" aliases it.
    "v0.1":     lambda p: p.version == "v0.1",
    "v0.2":     lambda p: p.version == "v0.2",
    "v0.3":     lambda p: p.version == "v0.3",
    "latest":   lambda p: p.version == "v0.3",
    # The single pair to sanity-check a change against.
    "demo":     lambda p: p.pair_id == "grpo-search-3b-v0.3",
}
# Comma-separated group names intersect on version and union otherwise, e.g.
# "v0.3,grpo" means the v0.3 GRPO pairs.
parts = [x.strip() for x in sel.split(",") if x.strip()]
# A leading "-" excludes: "all,-v0.2" is everything except the v0.2 pairs.
include = [x for x in parts if not x.startswith("-")]
exclude = [x[1:] for x in parts if x.startswith("-")]
unknown_ex = [x for x in exclude if x not in groups]
if unknown_ex:
    sys.exit(f"unknown group to exclude: {', '.join(unknown_ex)}")

if include and all(x in groups for x in include):
    # Version groups (v0.1, v0.2, v0.3, latest) union with each other — a pair
    # cannot belong to two versions, so intersecting them always gives zero.
    # Other groups intersect: "v0.3,grpo" = v0.3 AND grpo.
    # Combined: "v0.1,v0.3,grpo" = (v0.1 OR v0.3) AND grpo.
    _VERSIONS = {"v0.1", "v0.2", "v0.3", "latest"}
    ver_terms = [x for x in include if x in _VERSIONS]
    other_terms = [x for x in include if x not in _VERSIONS]

    def _match(p):
        if ver_terms and not any(groups[v](p) for v in ver_terms):
            return False
        if other_terms and not all(groups[o](p) for o in other_terms):
            return False
        return True

    chosen = [p.pair_id for p in PAIRS
              if _match(p)
              and not any(groups[x](p) for x in exclude)]
elif any(x in groups for x in include):
    sys.exit(f"do not mix group names with pair ids: {sel}")
elif exclude and not include:
    sys.exit("an exclusion needs something to exclude from, e.g. 'all,-v0.2'")
else:
    parts = include
    chosen = []
    for name in parts:
        if name not in PAIRS_BY_ID:
            sys.exit(f"unknown pair '{name}'. Known: {', '.join(sorted(PAIRS_BY_ID))}")
        chosen.append(name)
if not chosen:
    sys.exit(f"selection '{sel}' matched no pairs")
print("\n".join(chosen))
PY
) || die "could not resolve --pairs '$PAIRS_SEL'"
mapfile -t PAIR_LIST <<< "$PAIR_LIST_RAW"
[[ ${#PAIR_LIST[@]} -gt 0 && -n "${PAIR_LIST[0]}" ]] || die "--pairs '$PAIRS_SEL' matched no pairs"

NEEDS_SEARCH=0
for p in "${PAIR_LIST[@]}"; do [[ "$p" == *"-search-"* ]] && NEEDS_SEARCH=1; done

# ---- preflight ------------------------------------------------------------- #
log "preflight"

command -v python3 >/dev/null || die "python3 not found"

# A real import check, not just "is it installed": a torch/torchvision CUDA
# mismatch only shows up when a model class is actually resolved, and finding
# that out after a 7-minute merge is a waste of everyone's time.
# Advisory by default. The checker's job is to name the problem early, and it
# has been wrong before; a bug in it must not be able to block a working box.
# --strict-env makes it fatal, --skip-env-check skips it entirely.
if [[ $SKIP_ENV_CHECK -eq 1 ]]; then
    warn "environment check skipped (--skip-env-check)"
else
    ENV_ARGS=()
    has_stage eval && ENV_ARGS+=(--full --search-r1-root "$SEARCH_R1_ROOT")
    if python3 shadow_rl/check_env.py "${ENV_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"; then
        ok "environment check passed"
    elif [[ $STRICT_ENV -eq 1 ]]; then
        die "environment check failed (--strict-env). Apply the fixes above."
    else
        warn "environment check reported problems -- continuing anyway."
        warn "If a model fails to load later, the fixes printed above are why."
    fi
fi

[[ -f "$SEARCH_R1_ROOT/verl/utils/reward_score/qa_em.py" ]] \
    || die "Search-R1 not found at SEARCH_R1_ROOT=$SEARCH_R1_ROOT. Run ./shadow_rl/setup.sh"

# CUDA_VISIBLE_DEVICES wins over nvidia-smi: it is what the workers will
# actually see, and pinning is how the parallel runner divides the machine.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
elif command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
else
    NGPU=0
fi
if has_stage eval; then
    [[ "$NGPU" -gt 0 ]] || die "no GPU visible, but the eval stage needs one"
    [[ "$TP" -le "$NGPU" ]] || die "--tp $TP exceeds the $NGPU visible GPU(s)"
    log "$NGPU GPU(s), tensor_parallel_size=$TP"
fi

# Disk: the RL checkpoints are frequently fp32, so roughly double the bf16 size.
AVAIL_GB=$(df -PBG "$MODEL_DIR" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
NEED_GB=$(python3 - "${PAIR_LIST[@]}" <<'PY'
import sys
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS_BY_ID
import re


def billions(size: str) -> float:
    """Parameter count in billions, from the manifest key (3b, llama3.1-8b, ...)."""
    m = re.search(r"(\d+(?:\.\d+)?)b$", size)
    return float(m.group(1)) if m else 8.0    # unknown backbone: assume mid-size


# bf16 is ~2 bytes/param; the released RL checkpoints are frequently fp32, so
# ~4. Per pair: two RL checkpoints (fp32) + the merged model (bf16).
def pair_gb(size):
    b = billions(size)
    return 2 * (4 * b) + (2 * b)


def originals_gb(size):
    return 2 * (2 * billions(size))           # base + instruct, bf16, shared


need = sum(pair_gb(PAIRS_BY_ID[p].size) for p in sys.argv[1:])
need += sum(originals_gb(s) for s in {PAIRS_BY_ID[p].size for p in sys.argv[1:]})
print(int(need))
PY
)
log "disk: ~${NEED_GB}GB needed under $MODEL_DIR, ${AVAIL_GB:-?}GB free"
if [[ -n "${AVAIL_GB:-}" && "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    if [[ $CLEANUP -eq 1 ]]; then
        warn "less free space than the full total, but --cleanup frees each pair as it finishes"
    else
        warn "NOT enough free space for all pairs at once. Re-run with --cleanup,"
        warn "or work through the pairs in smaller batches."
    fi
fi

retriever_alive() {
    curl -fsS --max-time 5 -X POST "$RETRIEVER_URL" \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":3,"return_scores":true}' >/dev/null 2>&1
}

# Started by us -> ours to stop, on any exit path including Ctrl-C.
stop_retriever() {
    if [[ -n "$RETRIEVER_PID" ]] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        log "stopping the retrieval server we started (pid $RETRIEVER_PID)"
        kill "$RETRIEVER_PID" 2>/dev/null || true
        wait "$RETRIEVER_PID" 2>/dev/null || true
    fi
}
trap stop_retriever EXIT INT TERM

# Results from two different retrievers are not comparable, and nothing in
# results.csv records which one produced a row. Keep a marker beside it and say
# so loudly rather than letting a mixed table look like a real finding.
RETRIEVER_MARK="${RESULTS%.csv}.retriever"
if [[ $NEEDS_SEARCH -eq 1 ]] && has_stage eval; then
    if [[ -f "$RETRIEVER_MARK" ]]; then
        PREV="$(cat "$RETRIEVER_MARK")"
        if [[ "$PREV" != "$RETRIEVER" && "$RETRIEVER" != auto ]]; then
            warn "$RESULTS already holds rows retrieved with '$PREV', and this run"
            warn "uses '$RETRIEVER'. Search-pair EM is not comparable across"
            warn "retrievers. Start a fresh --out/RESULTS file, or re-run the"
            warn "existing rows with --force."
        fi
    fi
    if retriever_alive; then
        ok "retrieval server responding at $RETRIEVER_URL"
        python3 shadow_rl/check_retriever.py --url "$RETRIEVER_URL" --quiet \
            || die "the retriever is listening but not returning passages (see above)"
        # Started outside this script, so its retriever cannot be interrogated --
        # the server exposes no such endpoint. Record only what was asserted.
        [[ "$RETRIEVER" != auto ]] && echo "$RETRIEVER" > "$RETRIEVER_MARK"
    elif [[ $AUTO_RETRIEVER -eq 1 ]]; then
        # Check faiss/pyserini/JVM before the download, not after it. Otherwise a
        # missing JDK is discovered ~70GB and an hour later, and only as a
        # dlopen error buried in the server's own log.
        RETR_ARGS=(--retriever "$RETRIEVER")
        [[ -n "$RETRIEVER_PORT" ]] && RETR_ARGS+=(--port "$RETRIEVER_PORT")
        ./shadow_rl/launch_retriever.sh "${RETR_ARGS[@]}" --check \
            || die "the retriever cannot start; see the errors above"
        RETR_LOG="$LOG_DIR/retriever.log"
        log "starting the $RETRIEVER retriever (log: $RETR_LOG)"
        ./shadow_rl/launch_retriever.sh "${RETR_ARGS[@]}" >>"$RETR_LOG" 2>&1 &
        RETRIEVER_PID=$!
        log "waiting for it to come up (first run downloads ~70GB of corpus+index)"
        for i in $(seq 1 720); do
            if retriever_alive; then break; fi
            if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
                die "the retriever exited during startup. Last lines of $RETR_LOG:
$(tail -20 "$RETR_LOG" 2>/dev/null)"
            fi
            sleep 10
            [[ $((i % 30)) -eq 0 ]] && log "  still waiting ($((i / 6))m)..."
        done
        retriever_alive || die "retriever did not become ready. See $RETR_LOG"
        [[ -f "$LOG_DIR/retriever.url" ]] && RETRIEVER_URL="$(cat "$LOG_DIR/retriever.url")"
        ok "retrieval server up at $RETRIEVER_URL (pid $RETRIEVER_PID)"
        echo "$RETRIEVER" > "$RETRIEVER_MARK"
        # Binding a port is not the same as answering usefully. Ask it a real
        # question once: an index that loads but returns nothing, or takes
        # seconds per query, would otherwise be discovered dataset by dataset.
        python3 shadow_rl/check_retriever.py --url "$RETRIEVER_URL" --quiet \
            || die "the retriever is listening but not returning passages (see above)"
    else
        die "selection includes search pairs, but no retrieval server at $RETRIEVER_URL.
       Start one that outlives its shell:
         ./shadow_rl/launch_retriever.sh --retriever $RETRIEVER --daemon
       (without --daemon it runs in the foreground and dies with the terminal,
        which is the usual reason this message appears the second time)
       or re-run with --auto-retriever to have this script manage it.
       To check its dependencies without downloading anything:
         ./shadow_rl/launch_retriever.sh --retriever $RETRIEVER --check"
    fi
fi

# ---- runtime estimate ------------------------------------------------------ #
# Rough, from vLLM throughput on one H200 and the multi-turn rollout cost. Meant
# to prevent a two-week surprise, not to be accurate to the hour.
EST_WORKERS=$( if [[ "$JOBS" == "auto" ]]; then j=$(( NGPU / TP )); [[ $j -lt 1 ]] && j=1; echo $j; else echo "$JOBS"; fi )
EST_HOURS=$( { python3 - "$SAMPLE" "$LIMIT" "$ROLES" "$TP" "$SKIP_DATASETS" "$EST_WORKERS" "${PAIR_LIST[@]}" <<'ESTPY'
import sys
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS_BY_ID

sample, limit, roles, tp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 1)
skip = {d.strip() for d in sys.argv[5].split(",") if d.strip()}
workers = max(1, int(sys.argv[6] or 1))
pair_ids = sys.argv[7:]

FULL = {"nq": 3610, "triviaqa": 11313, "popqa": 14267, "hotpotqa": 7405,
        "2wikimultihopqa": 12576, "musique": 2417, "bamboogle": 125,
        "gpqa_diamond": 198, "simpleqa": 4326}
# Any dataset added to the registry without a row count here would silently
# vanish from the estimate, so fall back to a placeholder and say so.
from pairs import DATASETS as _ALL
_missing = [d for d in _ALL if d not in FULL]
for d in _missing:
    FULL[d] = 1000
if _missing:
    print(f"[warn] no row count for {', '.join(_missing)}; estimate assumes 1000 each",
          file=sys.stderr)

n = int(sample) if sample else (int(limit) if limit else None)
per_role = sum(min(n, v) if n else v for d, v in FULL.items()
               if d in _ALL and d not in skip)
print(f"{per_role}", file=sys.stderr)   # picked up by the plan line

import re


def billions(size):
    m = re.search(r"(\d+(?:\.\d+)?)b$", size)
    return float(m.group(1)) if m else 8.0


# Anchors measured on one H200: ~14 q/s for a 3B without search, ~2.5 q/s for a
# 7B with it. Throughput falls roughly with parameter count, and search costs
# extra turns per question.
def rate_for(size, with_search):
    b = billions(size)
    base = 42.0 / b                 # 3B -> 14 q/s, 7B -> 6 q/s
    return base / 2.4 if with_search else base


role_list = [r.strip() for r in roles.split(",") if r.strip()]
# W_B and W_I are shared by every pair of a size, so they are scored once per
# size rather than once per pair -- charge them to the first pair of that size.
shared = {"base_baseline", "instruct_baseline"}
per_pair_roles = [r for r in role_list if r not in shared]
shared_roles = [r for r in role_list if r in shared]
speedup = min(tp, 2.0)

total = 0.0
seen_sizes = set()
for pid in pair_ids:
    p = PAIRS_BY_ID[pid]
    rate = rate_for(p.size, p.with_search) * speedup
    n = len(per_pair_roles)
    if p.size not in seen_sizes:
        n += len(shared_roles)
        seen_sizes.add(p.size)
    total += n * per_role / rate
# Independent (role, dataset) cells run concurrently, one per worker. Not
# perfectly linear -- the last wave is ragged -- so allow a little overhead.
if workers > 1:
    total /= workers * 0.85
h = total / 3600
if h >= 24:
    print(f"~{h:.0f}h (~{h/24:.1f} days)")
elif h >= 1:
    print(f"~{h:.1f}h")
else:
    print(f"~{max(1, round(h * 60))}min")
ESTPY
} 2>"$LOG_DIR/.per_role" )
PER_ROLE=$(cat "$LOG_DIR/.per_role" 2>/dev/null | tail -1)
PER_ROLE="${PER_ROLE:-?}"
read -r N_DATASETS N_DATASETS_ALL < <(python3 -c "
import sys; sys.path.insert(0,'shadow_rl')
from pairs import DATASETS
skip={d.strip() for d in '$SKIP_DATASETS'.split(',') if d.strip()}
print(len([d for d in DATASETS if d not in skip]), len(DATASETS))")
rm -f "$LOG_DIR/.per_role"


# ---- plan ------------------------------------------------------------------ #
cat <<EOF | tee -a "$RUN_LOG"

==============================================================
 pairs    : ${#PAIR_LIST[@]}  ($(IFS=,; echo "${PAIR_LIST[*]}"))
 stages   : $STAGES
 roles    : $ROLES
 models   : $MODEL_DIR$( [[ -z "${SHADOW_RL_MODEL_DIR:-}" ]] && echo "  (HuggingFace cache)" )
 merged   : $MERGED_DIR
 results  : $RESULTS
 version  : $GIT_REV$GIT_DIRTY
 log      : $RUN_LOG
 gpus     : $NGPU visible, tp=$TP, workers=$( if [[ "$JOBS" == "auto" ]]; then
   j=$(( NGPU / TP )); [[ $j -lt 1 ]] && j=1; echo "$j (auto)"; else echo "$JOBS"; fi )
 datasets : $N_DATASETS of $N_DATASETS_ALL$( [[ -n "$SKIP_DATASETS" ]] && echo " (skipping $SKIP_DATASETS)" )
 questions: $( if [[ -n "$SAMPLE" ]]; then echo "$SAMPLE sampled per dataset, $PER_ROLE per role"; elif [[ -n "$LIMIT" ]]; then echo "first $LIMIT per dataset, $PER_ROLE per role (biased; smoke only)"; else echo "FULL test sets, $PER_ROLE per role"; fi )
 est. time: $EST_HOURS
 cleanup  : $( [[ $CLEANUP -eq 1 ]] && echo "yes (frees each pair after eval)" || echo "no (models kept)" )
 reuse    : $( if [[ $FORCE -eq 1 ]]; then echo "nothing (--force)"; else
   done_n=0; for _p in "${PAIR_LIST[@]}"; do merged_complete "$MERGED_DIR/$_p" && done_n=$((done_n+1)); done
   rows=0; [[ -f "$RESULTS" ]] && rows=$(($(wc -l < "$RESULTS") - 1))
   echo "$done_n/${#PAIR_LIST[@]} merged models, $rows result row(s) already present"; fi )
==============================================================
EOF

[[ $DRY -eq 1 ]] && { log "dry run, stopping here"; exit 0; }

if [[ $ASSUME_YES -eq 0 && -t 0 ]]; then
    read -r -p "proceed? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { log "aborted"; exit 0; }
fi

START_TS=$(date +%s)
declare -a FAILED_PAIRS=()

# ---- stage 1: similarity (all pairs at once, one CSV) ---------------------- #
if has_stage similarity; then
    log "stage: parameter-level similarity"
    SIM_LOG="$LOG_DIR/similarity.log"
    APPEND=""
    [[ -f "$REPO_ROOT/shadow_rl/similarity_summary.csv" ]] && APPEND="--append"
    for pid in "${PAIR_LIST[@]}"; do
        if [[ -n "$APPEND" ]] && grep -q "^$pid," "$REPO_ROOT/shadow_rl/similarity_summary.csv" 2>/dev/null; then
            log "  skip $pid (already in similarity_summary.csv)"
            continue
        fi
        log "  $pid"
        if python3 shadow_rl/similarity.py --pair "$pid" \
                --params-out  "$REPO_ROOT/shadow_rl/similarity_params.csv" \
                --summary-out "$REPO_ROOT/shadow_rl/similarity_summary.csv" \
                $APPEND 2>&1 | tee -a "$SIM_LOG"; then
            APPEND="--append"
        else
            err "  similarity failed for $pid (see $SIM_LOG)"
        fi
    done
    ok "similarity -> shadow_rl/similarity_{params,summary}.csv"
fi

# ---- stages 2-3: merge + evaluate, pair by pair ---------------------------- #
for pid in "${PAIR_LIST[@]}"; do
    PAIR_START=$(date +%s)
    PAIR_LOG="$LOG_DIR/$pid.log"
    SHADOW_PATH="$MERGED_DIR/$pid"
    PAIR_OK=1

    log ""
    log "=============== $pid ==============="

    read -r BASE INSTRUCT RL_BASE RL_INSTRUCT < <(python3 - "$pid" <<'PY'
import sys
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS_BY_ID
p = PAIRS_BY_ID[sys.argv[1]]
print(p.base, p.instruct, p.repo("rl_on_base"), p.repo("rl_on_instruct"))
PY
)

    # -- merge --
    if has_stage merge; then
        if [[ $FORCE -eq 0 ]] && merged_complete "$SHADOW_PATH"; then
            log "  merge: complete model already at $SHADOW_PATH, skipping"
        elif [[ $FORCE -eq 0 && -d "$SHADOW_PATH" ]] && ! merged_complete "$SHADOW_PATH"; then
            warn "  merge: $SHADOW_PATH exists but is incomplete; redoing it"
            rm -rf "$SHADOW_PATH"
            log "  merge: W_shadow = W_I + (RL(W_B) - W_B)"
            if python3 shadow_rl/merge.py \
                    --base "$BASE" --instruct "$INSTRUCT" \
                    --rl-base "$RL_BASE" \
                    --out "$SHADOW_PATH" 2>&1 | tee -a "$PAIR_LOG"; then
                ok "  merged -> $SHADOW_PATH"
            else
                err "  merge failed (see $PAIR_LOG)"
                rm -rf "$SHADOW_PATH"
                FAILED_PAIRS+=("$pid:merge"); continue
            fi
        else
            log "  merge: W_shadow = W_I + (RL(W_B) - W_B)"
            # --rl-instruct is deliberately omitted: it only adds the
            # instruct-side delta magnitude to the printout and costs a full
            # extra pass over that checkpoint.
            if python3 shadow_rl/merge.py \
                    --base "$BASE" --instruct "$INSTRUCT" \
                    --rl-base "$RL_BASE" \
                    --out "$SHADOW_PATH" 2>&1 | tee -a "$PAIR_LOG"; then
                ok "  merged -> $SHADOW_PATH"
            else
                err "  merge failed (see $PAIR_LOG)"
                rm -rf "$SHADOW_PATH"      # never leave a half-written model behind
                FAILED_PAIRS+=("$pid:merge"); continue
            fi
        fi

        if [[ $FORCE -eq 0 ]] && smoke_passed "$SHADOW_PATH"; then
            log "  smoke test: already passed for this model, skipping"
        else
            log "  smoke test"
            if python3 shadow_rl/smoke_test.py --model "$SHADOW_PATH" 2>&1 | tee -a "$PAIR_LOG"; then
                touch "$SHADOW_PATH/.smoke_ok"
                ok "  merged model loads and generates coherent text"
            else
                rm -f "$SHADOW_PATH/.smoke_ok"
                err "  smoke test failed -- not evaluating this pair (see $PAIR_LOG)"
                FAILED_PAIRS+=("$pid:smoke"); continue
            fi
        fi
    fi

    # -- evaluate --
    if has_stage eval; then
        # With more than one worker, hand the whole (role x dataset) grid to the
        # parallel runner: a 3B model does not need tensor parallelism, so the
        # speedup comes from running independent cells side by side.
        EFFECTIVE_JOBS="$JOBS"
        if [[ "$JOBS" == "auto" ]]; then
            EFFECTIVE_JOBS=$(( NGPU / TP ))
            [[ "$EFFECTIVE_JOBS" -lt 1 ]] && EFFECTIVE_JOBS=1
        fi

        if [[ "$EFFECTIVE_JOBS" -gt 1 ]]; then
            log "  eval: $EFFECTIVE_JOBS parallel worker(s) x tp=$TP over $NGPU GPU(s)"
            PAR=(--pair "$pid" --search-r1-root "$SEARCH_R1_ROOT"
                 --out "$RESULTS" --roles "$ROLES"
                 --retriever-url "$RETRIEVER_URL"
                 --tp "$TP" --jobs "$EFFECTIVE_JOBS"
                 --shadow-path "$SHADOW_PATH")
            [[ -n "$SKIP_DATASETS" ]] && PAR+=(--skip-datasets "$SKIP_DATASETS")
            [[ -n "$SAMPLE" ]] && PAR+=(--sample "$SAMPLE")
            [[ -n "$LIMIT" ]]  && PAR+=(--limit "$LIMIT")
            [[ $FORCE -eq 1 ]] && PAR+=(--force)
            PAR+=(--fail-fast "$FAIL_FAST" --stagger "$STAGGER")
            [[ -n "$JOB_TIMEOUT" ]] && PAR+=(--timeout "$JOB_TIMEOUT")
            [[ $NO_CANARY -eq 1 ]] && PAR+=(--no-canary)
            if python3 shadow_rl/run_eval_parallel.py "${PAR[@]}" 2>&1 | tee -a "$PAIR_LOG"; then
                ok "  all cells done"
            else
                err "  some cells failed (see $PAIR_LOG)"
                FAILED_PAIRS+=("$pid:eval"); PAIR_OK=0
            fi
            IFS=',' read -ra ROLE_ARR <<< ""
        else
            IFS=',' read -ra ROLE_ARR <<< "$ROLES"
        fi
        for role in "${ROLE_ARR[@]}"; do
            # W_B and W_I are the *same* checkpoints for every pair of a given
            # size, so evaluating them once per pair would burn GPU hours
            # re-deriving identical numbers. Score each size once; the other
            # pairs leave the row blank and the report shows "-".
            if [[ "$role" == "base_baseline" || "$role" == "instruct_baseline" ]] \
               && [[ $FORCE -eq 0 ]]; then
                OWNER=$(python3 - "$RESULTS" "$pid" "$role" <<'DEDUPPY'
import csv, os, sys
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS_BY_ID
results, pid, role = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(results):
    raise SystemExit(0)
size = PAIRS_BY_ID[pid].size
with open(results, newline="") as fh:
    for r in csv.DictReader(fh):
        if r["model_role"] != role or r["size"] != size:
            continue
        tag = "search" if str(r["with_search"]).lower() == "true" else "nosearch"
        other = f"{r['algo']}-{tag}-{r['size']}-{r['version']}"
        if other != pid:
            print(other)
            break
DEDUPPY
)
                if [[ -n "$OWNER" ]]; then
                    log "  eval: $role — shared with $OWNER (same $role checkpoint), skipping"
                    continue
                fi
            fi
            log "  eval: $role"
            EXTRA=()
            [[ "$role" == "shadow" ]] && EXTRA=(--model-path "$SHADOW_PATH")
            [[ -n "$LIMIT" ]]  && EXTRA+=(--limit "$LIMIT")
            [[ -n "$SAMPLE" ]] && EXTRA+=(--sample "$SAMPLE")
            [[ -n "$SKIP_DATASETS" ]] && EXTRA+=(--skip-datasets "$SKIP_DATASETS")
            if python3 shadow_rl/evaluate.py \
                    --search-r1-root "$SEARCH_R1_ROOT" \
                    --pair "$pid" --role "$role" --out "$RESULTS" \
                    --retriever-url "$RETRIEVER_URL" \
                    --tensor-parallel-size "$TP" \
                    "${EXTRA[@]}" 2>&1 | tee -a "$PAIR_LOG"; then
                ok "  $role done"
            else
                err "  eval failed for $role (see $PAIR_LOG)"
                FAILED_PAIRS+=("$pid:eval:$role"); PAIR_OK=0
            fi
        done
    fi

    # -- cleanup --
    if [[ $CLEANUP -eq 1 && $PAIR_OK -eq 1 ]]; then
        log "  cleanup: freeing this pair's checkpoints"
        for repo in "$RL_BASE" "$RL_INSTRUCT"; do
            d="$MODEL_DIR/${repo//\//__}"
            [[ -d "$d" ]] && { rm -rf "$d"; log "    removed $d"; }
        done
        [[ -d "$SHADOW_PATH" ]] && { rm -rf "$SHADOW_PATH"; log "    removed $SHADOW_PATH"; }
    elif [[ $CLEANUP -eq 1 ]]; then
        warn "  keeping checkpoints: this pair had failures, so they may be needed for a retry"
    fi

    ok "$pid finished in $(( ($(date +%s) - PAIR_START) / 60 ))m"
done

# ---- stage 4: aggregate ---------------------------------------------------- #
if has_stage aggregate; then
    log ""
    log "stage: aggregate"
    python3 shadow_rl/aggregate.py --results "$RESULTS" \
        --out "$REPO_ROOT/shadow_rl/FINDINGS.md" 2>&1 | tee -a "$RUN_LOG"

    # Comparison table straight to the terminal; the log copy is colour-free.
    python3 shadow_rl/report.py --results "$RESULTS"
    python3 shadow_rl/report.py --results "$RESULTS" --no-color >>"$RUN_LOG" 2>&1 || true
fi

# ---- summary --------------------------------------------------------------- #
ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
log ""
log "=============================================================="
if [[ ${#FAILED_PAIRS[@]} -eq 0 ]]; then
    ok "all ${#PAIR_LIST[@]} pair(s) completed in ${ELAPSED}m"
else
    warn "${#FAILED_PAIRS[@]} failure(s) in ${ELAPSED}m:"
    for f in "${FAILED_PAIRS[@]}"; do warn "    $f"; done
    warn "logs in $LOG_DIR/ -- re-running skips whatever already succeeded"
fi
log "results  : $RESULTS"
log "findings : shadow_rl/FINDINGS.md"
log "log      : $RUN_LOG"
log "=============================================================="

[[ ${#FAILED_PAIRS[@]} -eq 0 ]] || exit 1
