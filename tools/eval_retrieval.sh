#!/usr/bin/env bash
# Measure whether reference retrieval fetches genuinely RELEVANT material, on real text.
#
# The hermetic suite (tools/run_tests.sh) pins retrieval mechanics with hand-picked vectors and
# needs no server. It cannot tell you whether a retriever fetches useful examples from a real
# corpus in a real language -- that needs real servers and an independent judge, so it lives
# here rather than in the suite. This is a GATE, not just a report: it exits non-zero if the
# hybrid retriever is not at least as good as its own single arms (see --precision-floor).
#
# Needs, and checks for, servers that must already be running:
#   --embedding-url  (tools/serve_embeddings.sh)   the dense arm
#   --judge-url      (tools/serve_model.sh)        an independent relevance judge
#   --rerank-url     (tools/serve_reranker.sh)     OPTIONAL; adds the hybrid+rerank variant
#
# Usage:
#   tools/eval_retrieval.sh --corpus C.jsonl --queries-from POOL.jsonl [options]
#   tools/eval_retrieval.sh --corpus C.jsonl --queries-from POOL.jsonl \
#       --rerank-url http://127.0.0.1:8082/v1 --queries 300 --out results.json
#
# Any option not listed here is passed straight through to the evaluator
# (tools/lib/eval_retrieval.py --help lists them all).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

embedding_url="http://127.0.0.1:8081/v1"
judge_url="http://127.0.0.1:8080/v1"
rerank_url=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding-url) embedding_url="${2:-}"; shift 2 || die "--embedding-url needs a value" ;;
        --judge-url) judge_url="${2:-}"; shift 2 || die "--judge-url needs a value" ;;
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

# Fail with the exact URL that is down rather than letting the evaluator die mid-sweep with a
# connection error after minutes of work -- never a silent skip.
check_server() {
    # check_server <url> <what> <how to start it>
    curl -fsS --max-time 5 "${1%/}/models" >/dev/null 2>&1 \
        || die "no server responding at ${1} (${2}). Start it with: ${3}"
    note "${2} ready at ${1}"
}
check_server "$judge_url" "relevance judge" "tools/serve_model.sh <model>"
check_server "$embedding_url" "embedding server" "tools/serve_embeddings.sh"
if [[ -n "$rerank_url" ]]; then
    check_server "$rerank_url" "reranker" "tools/serve_reranker.sh"
    passthrough+=(--rerank-url "$rerank_url")
else
    note "no --rerank-url given: the hybrid+rerank variant is skipped"
fi

exec "$python" "${REPO_ROOT}/tools/lib/eval_retrieval.py" \
    --embedding-url "$embedding_url" --judge-url "$judge_url" "${passthrough[@]}"
