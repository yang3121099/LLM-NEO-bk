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

# A JAVA_HOME that pyserini can actually use.
#
# pyserini wraps Lucene and loads libjvm.so through jnius, which derives the path
# from JAVA_HOME. An existing-but-wrong JAVA_HOME is worse than an unset one:
# conda's openjdk lives at $CONDA_PREFIX/lib/jvm, so jnius tries that first even
# in an environment where it was never installed, and the failure arrives much
# later inside the retrieval server as
#   Error calling dlopen(b'.../lib/jvm/lib/server/libjvm.so'): No such file
# So the test is not "does the directory exist" but "is libjvm.so under it".
shadow_rl_java_home() {
    local candidates=() c
    [[ -n "${JAVA_HOME:-}" ]] && candidates+=("$JAVA_HOME")
    [[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("$CONDA_PREFIX/lib/jvm" "$CONDA_PREFIX")
    if command -v java >/dev/null 2>&1; then
        candidates+=("$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")")
    fi
    # Newest first: pyserini targets 21 and fails at query time on older JDKs.
    while IFS= read -r c; do candidates+=("$c"); done \
        < <(ls -d /usr/lib/jvm/*/ /usr/lib64/jvm/*/ 2>/dev/null | sort -rV)
    for c in "${candidates[@]}"; do
        [[ -n "$c" ]] || continue
        c="${c%/}"
        if compgen -G "$c/lib/server/libjvm.so" > /dev/null \
           || compgen -G "$c/lib/*/server/libjvm.so" > /dev/null \
           || compgen -G "$c/jre/lib/*/server/libjvm.so" > /dev/null; then
            echo "$c"
            return 0
        fi
    done
    return 1
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
