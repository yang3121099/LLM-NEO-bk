# #!/usr/bin/env bash
# # -----------------------------------------------------------------------------
# # run_infer_and_eval.sh  —  v2 (parallel edition)
# # -----------------------------------------------------------------------------
# #  * Runs the OpenCompass **infer** phase in the background, streaming its output
# #    to both the console and a log file.
# #  * Extracts the first YYYYMMDD_HHMMSS timestamp that OpenCompass prints.
# #  * As soon as the timestamp appears, kicks off **three** eval runs:
# #       - immediately
# #       - after 30 min (default 1800 s, flag -s/--sleep30)
# #       - after 60 min (default 3600 s, flag -t/--sleep60)
# #  * Prints the exact eval command as a handy [run_eval_after] line.
# #  * All evals are backgrounded as well, so everything runs in parallel; their
# #    output interleaves with infer’s — just like you asked.
# # -----------------------------------------------------------------------------
# # Usage:
# #   chmod +x run_infer_and_eval.sh
# #   ./run_infer_and_eval.sh <experiment.py> [options]
# #
# # Options:
# #   -w | --workers  N    set --max-num-workers (default 32)
# #   -s | --sleep30  SEC  delay for 2nd eval  (default 1800)
# #   -t | --sleep60  SEC  delay for 3rd eval  (default 3600)
# #   -T | --timeout  SEC  fail if no timestamp within N seconds (default 300)
# # -----------------------------------------------------------------------------
# set -euo pipefail

# # ---------------------------
# # default parameters
# # ---------------------------
# NUM_WORKERS=32
# SLEEP_30=19600   # 15 min
# # SLEEP_30=60
# SLEEP_60=36000   # 30 min
# SLEEP_120=3600   # 60 min
# SLEEP_150=54000   # 60 min


# TIMEOUT=3600000     # 5 min timeout for timestamp detection

# # ---------------------------
# # parse args
# # ---------------------------
# if [[ $# -lt 1 ]]; then
#   echo "Usage: $0 <experiment_file.py> [options]" >&2
#   exit 1
# fi

# EXP_FILE=""
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     -w|--workers) NUM_WORKERS="$2"; shift 2 ;;
#     -s|--sleep30) SLEEP_30="$2"; shift 2 ;;
#     -t|--sleep60) SLEEP_60="$2"; shift 2 ;;
#     -T|--timeout) TIMEOUT="$2"; shift 2 ;;
#     *)
#       if [[ -z "$EXP_FILE" ]]; then EXP_FILE="$1"; shift 1 ; else
#         echo "Unknown argument: $1" >&2; exit 1 ; fi ;;
#   esac
# done

# if [[ ! -f "$EXP_FILE" ]]; then
#   echo "Experiment file not found: $EXP_FILE" >&2
#   exit 1
# fi

# # ---------------------------
# # kick off infer in background
# # ---------------------------
# LOGFILE=$(mktemp --suffix=.infer.log)
# echo "[INFO] Starting infer in background … (log → $LOGFILE)" >&2
# # stdbuf flushes output line‑by‑line so grep sees it promptly
# # { stdbuf -oL -eL python3 run.py "$EXP_FILE" -m infer | tee "$LOGFILE"; } &
# { stdbuf -oL -eL python3 run.py "$EXP_FILE" | tee "$LOGFILE"; } &
# INFER_PID=$!
# echo "[INFO] infer PID = $INFER_PID" >&2

# # ---------------------------
# # wait for timestamp to appear
# # ---------------------------
# START=$(date +%s)
# TIMESTAMP=""
# PAT='[0-9]{8}_[0-9]{6}'
# while [[ -z "$TIMESTAMP" ]]; do
#   TIMESTAMP=$(grep -Eo "$PAT" "$LOGFILE" | head -n1 || true)
#   NOW=$(date +%s)
#   if (( NOW - START > TIMEOUT )); then
#     echo "[ERROR] Gave up after $TIMEOUT s without seeing timestamp." >&2
#     kill "$INFER_PID" 2>/dev/null || true
#     exit 1
#   fi
#   sleep 2
# done

# echo "[INFO] Detected timestamp: $TIMESTAMP" >&2

# # ---------------------------
# # build eval command and run 3 times in background
# # ---------------------------
# EVAL_CMD=(python3 run.py "$EXP_FILE" --max-num-workers "$NUM_WORKERS" -m eval -r "$TIMESTAMP")
# printf '\n[run_eval_after]\n%s\n\n' "${EVAL_CMD[*]}"

# echo "[INFO] Dispatching evals: now, +$SLEEP_30 s, +$SLEEP_60 s …" >&2
# ("${EVAL_CMD[@]}" &)
# (sleep "$SLEEP_30" && "${EVAL_CMD[@]}" &)
# (sleep "$SLEEP_60" && "${EVAL_CMD[@]}" &)
# (sleep "$SLEEP_120" && "${EVAL_CMD[@]}" &)
# (sleep "$SLEEP_150" && "${EVAL_CMD[@]}" &)

# # ---------------------------
# # finale
# # ---------------------------
# wait "$INFER_PID" || true
# # note: evals keep running in background; this script exits once infer finishes

# echo "[INFO] infer finished (exit=$?). Eval jobs are continuing in background."


#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_infer_and_eval.sh  —  v5 (bk_ckpt snapshot & 10‑min eval default)
# -----------------------------------------------------------------------------
#  * **All artifacts saved under**:
#      /apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opencompass/scripts/bk_ckpt
#    (created if missing)
#      - Snapshot of experiment .py → <bk_ckpt>/<basename>_snapshot_<ts>.py
#      - Infer log                  → <bk_ckpt>/infer_<ts>.log
#      - All eval logs (appended)   → <bk_ckpt>/eval_<ts>.log
#  * Runs infer in background, streams output both to terminal & infer log.
#  * Detects the OpenCompass timestamp from infer output.
#  * Fires eval **every 10 min** (default) until infer exits, each eval output
#    appended to eval log while also streaming to terminal.
#  * No timeout logic anymore — script waits patiently for timestamp.
# -----------------------------------------------------------------------------
# Usage:
#   chmod +x run_infer_and_eval.sh
#   ./run_infer_and_eval.sh <experiment.py> [options]
#
# Options:
#   -w | --workers   N     set --max-num-workers  (default 32)
#   -I | --interval  SEC   eval every SEC seconds (default 600)
#   --no-snapshot          use live file instead of snapshot
# -----------------------------------------------------------------------------
set -euo pipefail

BK_DIR="./temp_log/bk_ckpt"
mkdir -p "$BK_DIR"

# ---------------------------
# defaults
# ---------------------------
NUM_WORKERS=32
EVAL_INTERVAL=200   # 10 min
DO_SNAPSHOT=1
# EVAL_INTERVAL=1200   # 30 min

# ---------------------------
# parse args
# ---------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_file.py> [options]" >&2
  exit 1
fi

EXP_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workers)   NUM_WORKERS="$2"; shift 2 ;;
    -I|--interval)  EVAL_INTERVAL="$2"; shift 2 ;;
    --no-snapshot)  DO_SNAPSHOT=0; shift 1 ;;
    *)
      if [[ -z "$EXP_FILE" ]]; then EXP_FILE="$1"; shift 1; else
        echo "Unknown argument: $1" >&2; exit 1 ; fi ;;
  esac
done

if [[ ! -f "$EXP_FILE" ]]; then
  echo "Experiment file not found: $EXP_FILE" >&2
  exit 1
fi

SESSION=$(date +%Y%m%d_%H%M%S)
BASE_NAME=$(basename "${EXP_FILE%.py}")

# ---------------------------
# snapshot (optional)
# ---------------------------
if (( DO_SNAPSHOT )); then
  SNAP_FILE="$BK_DIR/${BASE_NAME}_snapshot_${SESSION}.py"
  cp "$EXP_FILE" "$SNAP_FILE"
  echo "[INFO] Snapshot saved → $SNAP_FILE" >&2
  EXP_FILE="$SNAP_FILE"
else
  echo "[INFO] Snapshotting disabled (using live file)." >&2
fi
# EXP_FILE="$EXP_FILE"

# ---------------------------
# log files
# ---------------------------
INFER_LOG="$BK_DIR/infer_${SESSION}.log"
EVAL_LOG="$BK_DIR/eval_${SESSION}.log"

# ---------------------------
# run infer in background
# ---------------------------
PAT='[0-9]{8}_[0-9]{6}'
echo "[INFO] Starting infer (log → $INFER_LOG) …" >&2
# { stdbuf -oL -eL python3 run.py "$EXP_FILE" -r 20250427_173017 | tee "$INFER_LOG"; } &
# { stdbuf -oL -eL python3 run.py "$EXP_FILE" -r 20250501_100657 | tee "$INFER_LOG"; } &
# { stdbuf -oL -eL python3 run.py "$EXP_FILE" -r 20250503_012111 | tee "$INFER_LOG"; } &
# { stdbuf -oL -eL python3 run.py "$EXP_FILE" -r 20250728_111111 | tee "$INFER_LOG"; } &
{ stdbuf -oL -eL python3 run.py "$EXP_FILE" -r 20250727_200012 | tee "$INFER_LOG"; } &


# { stdbuf -oL -eL python3 run.py "$EXP_FILE"  | tee "$INFER_LOG"; } &

INFER_PID=$!
echo "[INFO] infer PID = $INFER_PID" >&2

# ---------------------------
# wait for timestamp
# ---------------------------
TIMESTAMP=""
while [[ -z "$TIMESTAMP" ]]; do
  TIMESTAMP=$(grep -Eo "$PAT" "$INFER_LOG" | head -n1 || true)
  sleep 2
done

echo "[INFO] Detected timestamp: $TIMESTAMP" >&2

# ---------------------------
# build eval command
# ---------------------------
EVAL_CMD=(python3 run.py "$EXP_FILE" --max-num-workers "$NUM_WORKERS" -m eval -r "$TIMESTAMP")
printf '\n[run_eval_after]\n%s\n\n' "${EVAL_CMD[*]}"

# ---------------------------
# periodic eval loop
# ---------------------------
(
  while kill -0 "$INFER_PID" 2>/dev/null; do
    echo "[INFO] eval @ $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$EVAL_LOG" >&2
    { stdbuf -oL -eL "${EVAL_CMD[@]}" | tee -a "$EVAL_LOG"; } &
    sleep "$EVAL_INTERVAL"
    echo " "
    sleep "$EVAL_INTERVAL"
    echo " "
    sleep "$EVAL_INTERVAL"
  done
  echo "[INFO] infer ended, stopping periodic evals." | tee -a "$EVAL_LOG" >&2
) & PERIODIC_PID=$!

# ---------------------------
# finale
# ---------------------------
wait "$INFER_PID" || true
wait "$PERIODIC_PID" || true

echo "[INFO] All done. Logs & snapshot in → $BK_DIR"


# bash ./eval_instruct_0427.sh  ./eval_quantw_20250727-bk.py 
# bash ./eval_instruct_0427.sh  ./eval_quantw_20250727-bk.py

# python3 ./run.py ./eval_quantw_20250727-bk.py  -r 20250727_200012