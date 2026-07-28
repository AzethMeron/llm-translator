#!/usr/bin/env bash
# Sweep reference_mmr_lambda end to end and measure what it costs and buys.
#
# The end-to-end A/B found the hybrid path reproducing the corpus's established rendering less
# often than the embedding arm, and blamed MMR diversification. The docs' advice to raise
# reference_mmr_lambda toward 1.0 for consistency is a named mechanism, NOT a measurement.
# This measures it, across the whole range.
#
# Cheaper than repeated A/Bs because the metric here is ABSOLUTE (mean trigram overlap with the
# established rendering, computed per arm against the corpus) rather than pairwise, so each
# lambda needs translating only once. Quality is not judged: the A/B already showed it is a wash.
#
# CAVEAT on the yardstick. "Established rendering" here is the most similar corpus entry found
# with the embedding retriever -- the same proxy that, in eval_translation.sh, turned out to be
# CIRCULAR when compared against an embedding arm (it is that arm's own top pick). So the
# absolute overlap values printed here sit on a compromised scale, and the embedding row is not
# a fair comparator. What the sweep IS good for is the comparison BETWEEN lambda values: the
# proxy is identical for every one of them, so it cannot manufacture a difference or hide one.
# Read the shape of the curve, not the absolute numbers.
#
# Needs, and checks for, servers that must already be running:
#   --base-url       (tools/serve_model.sh)        the translator
#   --embedding-url  (tools/serve_embeddings.sh)   the dense arm, and the established-rendering lookup
#   --rerank-url     (tools/serve_reranker.sh)     required for the hybrid+rerank arm
#
# Usage:
#   tools/sweep_mmr.sh --corpus C.jsonl --queries Q.jsonl --agents A.toml --work DIR \
#       --rerank-url http://127.0.0.1:8082/v1 [--lambdas 0.0 0.5 1.0]
#
# Any option not listed here is passed through (tools/lib/sweep_mmr.py --help lists them all).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

[[ $# -gt 0 ]] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

base_url="http://127.0.0.1:8080/v1"
embedding_url="http://127.0.0.1:8081/v1"
rerank_url=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url) base_url="${2:-}"; shift 2 || die "--base-url needs a value" ;;
        --embedding-url) embedding_url="${2:-}"; shift 2 || die "--embedding-url needs a value" ;;
        --rerank-url) rerank_url="${2:-}"; shift 2 || die "--rerank-url needs a value" ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"
has_httpx "$python" || die "httpx is not importable; run tools/setup_python_env.sh"
"$python" -c "import numpy" >/dev/null 2>&1 \
    || die "numpy is not importable (the embedding arm needs it); run tools/setup_python_env.sh"
require_command curl "Install curl, or check the servers by hand."

check_server() {
    curl -fsS --max-time 5 "${1%/}/models" >/dev/null 2>&1 \
        || die "no server responding at ${1} (${2}). Start it with: ${3}"
    note "${2} ready at ${1}"
}
check_server "$base_url" "translator" "tools/serve_model.sh <model>"
check_server "$embedding_url" "embedding server" "tools/serve_embeddings.sh"
if [[ -n "$rerank_url" ]]; then
    check_server "$rerank_url" "reranker" "tools/serve_reranker.sh"
    passthrough+=(--rerank-url "$rerank_url")
else
    note "no --rerank-url given: only the plain --hybrid arm can be swept"
fi

exec "$python" "${REPO_ROOT}/tools/lib/sweep_mmr.py" \
    --base-url "$base_url" --embedding-url "$embedding_url" \
    --python "$python" "${passthrough[@]}"
