#!/usr/bin/env bash
# Build a reference corpus + held-out query set from a real translated journal.
#
# Both live evaluations need the same split, and building it by hand is what made the earlier
# results irreproducible: the ad-hoc predecessor of this script asked for 40 queries, silently
# produced 24, and reported neither. Here the requested count is exact -- a shortfall is an
# error naming the shortfall, never a quietly smaller sample.
#
# Two shapes, picked by whether --pool is given:
#   --pool POOL.jsonl   the whole journal is the corpus; queries come from the separate pool
#                       (use when a large body of untranslated units exists alongside it)
#   (no --pool)         queries are held OUT of the journal and the corpus is the remainder
#                       (use when the journal is all there is; the corpus shrinks, and by how
#                        much is reported)
#
# Usage:
#   tools/build_eval_split.sh --journal J.jsonl --pool POOL.jsonl \
#       --corpus-out C.jsonl --queries-out Q.jsonl [--queries 1000] [--script-pattern japanese]
#   tools/build_eval_split.sh --journal J.jsonl \
#       --corpus-out C.jsonl --queries-out Q.jsonl --queries 400
#
# Any option not listed here is passed straight through
# (tools/lib/build_eval_split.py --help lists them all).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

[[ $# -gt 0 ]] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"

exec "$python" "${REPO_ROOT}/tools/lib/build_eval_split.py" "${passthrough[@]}"
