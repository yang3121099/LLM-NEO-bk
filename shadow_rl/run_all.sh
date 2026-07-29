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
#   --pairs X      all | nosearch | search | grpo | ppo | 3b | 7b | <pair_id>[,<pair_id>...]
#   --stages X     comma list of: similarity,merge,eval,aggregate   (default all)
#   --roles X      comma list of roles to evaluate  (default all four)
#   --limit N      first N questions per dataset; biased, smoke runs only
#   --sample N     deterministic random N per dataset, identical across roles.
#                  Recommended for multi-pair sweeps: the full test sets are
#                  ~51,700 questions per role, and --pairs all over them is on
#                  the order of two weeks of single-GPU time.
#   --skip-datasets X comma list of datasets to omit entirely
#   --auto-retriever  start and stop the BM25 server automatically
#   --cleanup      delete a pair's RL checkpoints and merged model once it is evaluated
#   --tp N         tensor parallel size (default 1; H200 fits 7B comfortably at 1)
#   --fast         smallest useful run: one 3B pair, 4 datasets (2 in-domain +
#                  2 OOD), 200 questions, all five models. A few minutes.
#   --force        redo work that is already complete (merge, smoke test)
#   --skip-env-check  do not run the environment check at all
#   --strict-env      abort if the environment check reports problems
#   --dry-run      print the plan and exit
#   --yes          skip the confirmation prompt
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults -------------------------------------------------------------- #
PAIRS_SEL="nosearch"
STAGES="similarity,merge,eval,aggregate"
ROLES="base_baseline,instruct_baseline,rl_on_instruct,rl_on_base,shadow"
FAST=0
LIMIT=""
SAMPLE=""
SKIP_DATASETS=""
CLEANUP=0
AUTO_RETRIEVER=0
RETRIEVER_PID=""
TP="${TP:-1}"
DRY=0
ASSUME_YES=0
FORCE=0
SKIP_ENV_CHECK=0
STRICT_ENV=0

SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-$HOME/Search-R1}"
MERGED_DIR="${MERGED_DIR:-$REPO_ROOT/shadow_rl/merged}"
RESULTS="${RESULTS:-$REPO_ROOT/shadow_rl/results.csv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/shadow_rl/logs}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
MODEL_DIR="${SHADOW_RL_MODEL_DIR:-$HOME/models}"
export SHADOW_RL_MODEL_DIR="$MODEL_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pairs)   PAIRS_SEL="$2"; shift 2 ;;
        --stages)  STAGES="$2";    shift 2 ;;
        --roles)   ROLES="$2";     shift 2 ;;
        --limit)   LIMIT="$2";     shift 2 ;;
        --sample)  SAMPLE="$2";    shift 2 ;;
        --skip-datasets) SKIP_DATASETS="$2"; shift 2 ;;
        --auto-retriever) AUTO_RETRIEVER=1; shift ;;
        --tp)      TP="$2";        shift 2 ;;
        --cleanup) CLEANUP=1;      shift ;;
        --fast)    FAST=1;         shift ;;
        --force)   FORCE=1;        shift ;;
        --skip-env-check) SKIP_ENV_CHECK=1; shift ;;
        --strict-env)     STRICT_ENV=1;     shift ;;
        --dry-run) DRY=1;          shift ;;
        --yes|-y)  ASSUME_YES=1;   shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# --fast: one 3B pair, two in-domain datasets, 200 sampled questions, and the
# five models the question is actually about. Enough to prove the pipeline runs
# and to see the sign of the effect; not enough to publish.
if [[ $FAST -eq 1 ]]; then
    [[ "$PAIRS_SEL" == "nosearch" ]] && PAIRS_SEL="ppo-nosearch-3b-v0.2"
    [[ -z "$SAMPLE" && -z "$LIMIT" ]] && SAMPLE=200
    [[ "$STAGES" == "similarity,merge,eval,aggregate" ]] && STAGES="merge,eval,aggregate"
    # nq + hotpotqa are in-domain; musique + bamboogle are out-of-domain and
    # small (2.4k and 125 rows), so they add generalisation signal almost free.
    # Skipped: triviaqa/popqa/2wiki, the three largest, which say little that
    # the cheaper OOD sets do not.
    SKIP_DATASETS="${SKIP_DATASETS:-triviaqa,popqa,2wikimultihopqa}"
fi

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
}
if sel in groups:
    chosen = [p.pair_id for p in PAIRS if groups[sel](p)]
else:
    chosen = []
    for name in sel.split(","):
        name = name.strip()
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

NGPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
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
# per pair: two RL checkpoints (fp32) + merged model (bf16); originals shared.
per = {"3b": 2 * 12 + 6, "7b": 2 * 28 + 15}
need = sum(per[PAIRS_BY_ID[p].size] for p in sys.argv[1:])
sizes = {PAIRS_BY_ID[p].size for p in sys.argv[1:]}
need += sum({"3b": 6 + 6, "7b": 15 + 15}[s] for s in sizes)   # shared originals
print(need)
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

if [[ $NEEDS_SEARCH -eq 1 ]] && has_stage eval; then
    if retriever_alive; then
        ok "retrieval server responding at $RETRIEVER_URL"
    elif [[ $AUTO_RETRIEVER -eq 1 ]]; then
        RETR_LOG="$LOG_DIR/retriever.log"
        log "starting the BM25 retriever (log: $RETR_LOG)"
        ./shadow_rl/launch_bm25_retriever.sh >>"$RETR_LOG" 2>&1 &
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
        ok "retrieval server up at $RETRIEVER_URL (pid $RETRIEVER_PID)"
    else
        die "selection includes search pairs, but no retrieval server at $RETRIEVER_URL.
       Either start one in another shell:
         ./shadow_rl/launch_bm25_retriever.sh
       or re-run with --auto-retriever to have this script manage it."
    fi
fi

# ---- runtime estimate ------------------------------------------------------ #
# Rough, from vLLM throughput on one H200 and the multi-turn rollout cost. Meant
# to prevent a two-week surprise, not to be accurate to the hour.
EST_HOURS=$( { python3 - "$SAMPLE" "$LIMIT" "$ROLES" "$TP" "$SKIP_DATASETS" "${PAIR_LIST[@]}" <<'ESTPY'
import sys
sys.path.insert(0, "shadow_rl")
from pairs import PAIRS_BY_ID

sample, limit, roles, tp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 1)
skip = {d.strip() for d in sys.argv[5].split(",") if d.strip()}
pair_ids = sys.argv[6:]

FULL = {"nq": 3610, "triviaqa": 11313, "popqa": 14267, "hotpotqa": 7405,
        "2wikimultihopqa": 12576, "musique": 2417, "bamboogle": 125}
n = int(sample) if sample else (int(limit) if limit else None)
per_role = sum(min(n, v) if n else v for d, v in FULL.items() if d not in skip)
print(f"{per_role}", file=sys.stderr)   # picked up by the plan line

rate = {("3b", False): 14.0, ("3b", True): 6.0, ("7b", False): 6.0, ("7b", True): 2.5}
n_roles = len([r for r in roles.split(",") if r.strip()])
speedup = min(tp, 2.0)

total = 0.0
for pid in pair_ids:
    p = PAIRS_BY_ID[pid]
    total += n_roles * per_role / (rate[(p.size, p.with_search)] * speedup)
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
N_DATASETS=$(python3 -c "
import sys; sys.path.insert(0,'shadow_rl')
from pairs import DATASETS
skip={d.strip() for d in '$SKIP_DATASETS'.split(',') if d.strip()}
print(len([d for d in DATASETS if d not in skip]))")
rm -f "$LOG_DIR/.per_role"


# ---- plan ------------------------------------------------------------------ #
cat <<EOF | tee -a "$RUN_LOG"

==============================================================
 pairs    : ${#PAIR_LIST[@]}  ($(IFS=,; echo "${PAIR_LIST[*]}"))
 stages   : $STAGES
 roles    : $ROLES
 models   : $MODEL_DIR
 merged   : $MERGED_DIR
 results  : $RESULTS
 log      : $RUN_LOG
 tp       : $TP
 datasets : $N_DATASETS of 7$( [[ -n "$SKIP_DATASETS" ]] && echo " (skipping $SKIP_DATASETS)" )
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
                    --rl-base "$RL_BASE" --rl-instruct "$RL_INSTRUCT" \
                    --out "$SHADOW_PATH" 2>&1 | tee -a "$PAIR_LOG"; then
                ok "  merged -> $SHADOW_PATH"
            else
                err "  merge failed (see $PAIR_LOG)"
                rm -rf "$SHADOW_PATH"
                FAILED_PAIRS+=("$pid:merge"); continue
            fi
        else
            log "  merge: W_shadow = W_I + (RL(W_B) - W_B)"
            if python3 shadow_rl/merge.py \
                    --base "$BASE" --instruct "$INSTRUCT" \
                    --rl-base "$RL_BASE" --rl-instruct "$RL_INSTRUCT" \
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
        IFS=',' read -ra ROLE_ARR <<< "$ROLES"
        for role in "${ROLE_ARR[@]}"; do
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
