#!/usr/bin/env bash
# Clone the Search-R1 evaluation harness into the working tree.
#
#   ./shadow_rl/setup_search_r1.sh            # clone (or verify) + check
#   ./shadow_rl/setup_search_r1.sh --check    # verify only, clone nothing
#   SEARCH_R1_ROOT=/data/Search-R1 ./shadow_rl/setup_search_r1.sh
#
# It lands at third_party/Search-R1 inside the repo, not $HOME: on a container
# $HOME is often /root and may not be writable, and the harness is only read
# from, never installed, so there is nothing to gain from putting it outside the
# tree. An existing $HOME/Search-R1 is still picked up rather than re-cloned.
#
# Nothing is pip-installed here. evaluate.py imports the checkout by path for
# qa_em.py (the official EM scorer) and the generation loop; `pip install -e`
# on Search-R1 would pull its own pinned torch/vllm and break the environment.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/shadow_rl/paths.sh"

SEARCH_R1_ROOT="$(shadow_rl_search_r1_root)"
SEARCH_R1_REPO="${SEARCH_R1_REPO:-https://github.com/PeterGriffinJin/Search-R1.git}"
SCORER="verl/utils/reward_score/qa_em.py"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

log()  { printf '\033[1;34m[search-r1]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ -f "$SEARCH_R1_ROOT/$SCORER" ]]; then
    ok "Search-R1 at $SEARCH_R1_ROOT"
    exit 0
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "[fail] Search-R1 not found at $SEARCH_R1_ROOT" >&2
    echo "       ./shadow_rl/setup_search_r1.sh" >&2
    exit 1
fi

if [[ -e "$SEARCH_R1_ROOT" && ! -d "$SEARCH_R1_ROOT/.git" ]]; then
    die "$SEARCH_R1_ROOT exists but is not a Search-R1 checkout; move it aside"
fi

if [[ -d "$SEARCH_R1_ROOT/.git" ]]; then
    # A checkout exists but the scorer is missing -- a partial clone, or the
    # wrong repository. Say which, rather than cloning on top of it.
    die "$SEARCH_R1_ROOT is a git checkout but has no $SCORER; is it really Search-R1?"
fi

log "cloning Search-R1 -> $SEARCH_R1_ROOT"
mkdir -p "$(dirname "$SEARCH_R1_ROOT")" || die "cannot create $(dirname "$SEARCH_R1_ROOT")"
git clone --depth 1 "$SEARCH_R1_REPO" "$SEARCH_R1_ROOT" || die "clone failed"

[[ -f "$SEARCH_R1_ROOT/$SCORER" ]] \
    || die "cloned, but $SCORER is missing -- the upstream layout may have changed"

cat <<EOF

$(ok "Search-R1 is ready — no SEARCH_R1_ROOT export needed")
  checkout : $SEARCH_R1_ROOT
  verify   : python3 shadow_rl/check_env.py --full
EOF
