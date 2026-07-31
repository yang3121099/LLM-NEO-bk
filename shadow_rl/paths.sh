# Shared default locations, sourced by the shell scripts in this directory.
# shadow_rl/paths.py is the python twin; the two must agree.
#
# Everything defaults into the working tree rather than $HOME. On a container or
# a shared box $HOME is often /root — small, or not writable at all — and a
# Search-R1 checkout or a 70GB corpus landing there fails in ways that surface
# much later as something unrelated. The working tree is the one directory we
# know is writable, because the repo is already in it.
#
# Anything already cloned under $HOME still works: an existing checkout there is
# picked up rather than re-cloned, so upgrading does not orphan one.

shadow_rl_repo_root() {
    # BASH_SOURCE[0] is this file, wherever it was sourced from.
    (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
}

# Where Search-R1 lives. Order: explicit override, in-tree checkout, an existing
# $HOME checkout, then the in-tree path as the place to create one.
shadow_rl_search_r1_root() {
    if [[ -n "${SEARCH_R1_ROOT:-}" ]]; then
        echo "$SEARCH_R1_ROOT"
        return
    fi
    local root
    root="$(shadow_rl_repo_root)"
    if [[ -d "$root/third_party/Search-R1" ]]; then
        echo "$root/third_party/Search-R1"
    elif [[ -n "${HOME:-}" && -d "$HOME/Search-R1" ]]; then
        echo "$HOME/Search-R1"
    else
        echo "$root/third_party/Search-R1"
    fi
}

# Same rule for verl, so setup_verl.sh and anything reading VERL_ROOT agree.
shadow_rl_verl_root() {
    if [[ -n "${VERL_ROOT:-}" ]]; then
        echo "$VERL_ROOT"
        return
    fi
    local root
    root="$(shadow_rl_repo_root)"
    if [[ -d "$root/third_party/verl" ]]; then
        echo "$root/third_party/verl"
    elif [[ -n "${HOME:-}" && -d "$HOME/verl" ]]; then
        echo "$HOME/verl"
    else
        echo "$root/third_party/verl"
    fi
}
