#!/usr/bin/env bash
###############################################################################
# merge_shadow.sh — the Shadow-FT graft, on the checkpoints this pipeline made.
#
#     Delta_K  = RL(W_B) - W_B     # what GRPO taught the base model
#     W_shadow = W_I + Delta_K     # transplanted onto the instruct backbone
#
# The arithmetic is shadow_rl/merge.py, unchanged -- it already streams shards,
# does the subtraction in fp32, checks key sets and shapes across all four
# checkpoints, and reports sigma. The only thing this adds is pointing it at the
# local exports instead of at released Search-R1 checkpoints.
#
#   ./verl_rl/merge_shadow.sh
#   ./verl_rl/merge_shadow.sh --scale 0.5     # any merge.py flag passes through
###############################################################################
set -euo pipefail

# shellcheck source=config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

info() { echo -e "\033[1;32m[merge]\033[0m $*"; }
die()  { echo -e "\033[1;31m[fail ]\033[0m $*" >&2; exit 1; }

RL_BASE="$EXPORT_DIR/base"
RL_INSTRUCT="$EXPORT_DIR/instruct"

for d in "$RL_BASE" "$RL_INSTRUCT"; do
  [[ -d "$d" ]] || die "missing $d -- run: python3 verl_rl/export_hf.py --role $(basename "$d")"
  compgen -G "$d/*.safetensors" >/dev/null || compgen -G "$d/*.bin" >/dev/null \
    || die "$d has no weight files; the export did not finish"
done

mkdir -p "$MERGED_DIR"

info "W_B        = $BASE_MODEL"
info "W_I        = $INSTRUCT_MODEL"
info "RL(W_B)    = $RL_BASE"
info "RL(W_I)    = $RL_INSTRUCT   (statistics only; not part of the graft)"
info "W_shadow  -> $MERGED_DIR"

python3 "$REPO_ROOT/shadow_rl/merge.py" \
  --base        "$BASE_MODEL" \
  --instruct    "$INSTRUCT_MODEL" \
  --rl-base     "$RL_BASE" \
  --rl-instruct "$RL_INSTRUCT" \
  --out         "$MERGED_DIR" \
  "$@"

# A merged model loads fine and talks nonsense when the graft has gone wrong;
# reading a few completions is the cheapest way to catch that before an eval.
info "smoke test"
python3 "$REPO_ROOT/shadow_rl/smoke_test.py" --model "$MERGED_DIR" \
  || die "the merged model failed its smoke test -- do not evaluate it yet"

info "done: $MERGED_DIR"
