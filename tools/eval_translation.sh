#!/usr/bin/env bash
# Translate a held-out set under two reference-retrieval arms and compare the OUTPUT.
#
# tools/eval_retrieval.sh answers "does the retriever fetch relevant material?" -- cheap, no
# translation. This answers the question that actually matters: does translating with the better
# retriever produce better output? It is "Phase B" of the staged evaluation in
# docs/feature-requests/hybrid-retrieval-reranking-rag.md.
#
# Three metrics, because they are expected to disagree:
#   A/B quality   a blinded, order-randomised judge picks between the arms. REPORTED, NOT GATED --
#                 the plan says per-line quality may be a wash, so gating it would gate on noise.
#   consistency   how close each arm's output is to the line's established rendering. GATED.
#                 REQUIRES --ground-truth (the journal the held-out queries came from).
#                 Without it this falls back to "the most similar corpus entry, per the
#                 embedding retriever" -- which IS the embedding arm's own top pick, so the
#                 comparison is circular and that arm wins by construction. Measured: it was
#                 shown that entry 297/297 times against the hybrid arm's 93/297. The script
#                 refuses that combination rather than reporting it.
#   rejection     how many units each arm failed to translate at all. GATED.
#
# Needs, and checks for, servers that must already be running:
#   --base-url       (tools/serve_model.sh)        the translator, and by default the judge too
#   --embedding-url  (tools/serve_embeddings.sh)   the dense arm, and the consistency measure
#   --rerank-url     (tools/serve_reranker.sh)     required by the hybrid+rerank arm
#
# Build the corpus/query split first with tools/build_eval_split.sh.
#
# Usage:
#   tools/eval_translation.sh --corpus C.jsonl --queries Q.jsonl --agents A.toml --work DIR \
#       --ground-truth JOURNAL.jsonl --rerank-url http://127.0.0.1:8082/v1 \
#       [--baseline embedding] [--candidate hybrid+rerank]
#
# Any option not listed here is passed straight through to the evaluator
# (tools/lib/eval_translation.py --help lists them all).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

[[ $# -gt 0 ]] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

base_url="http://127.0.0.1:8080/v1"
embedding_url="http://127.0.0.1:8081/v1"
judge_url=""
rerank_url=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url) base_url="${2:-}"; shift 2 || die "--base-url needs a value" ;;
        --embedding-url) embedding_url="${2:-}"; shift 2 || die "--embedding-url needs a value" ;;
        --judge-url) judge_url="${2:-}"; shift 2 || die "--judge-url needs a value" ;;
        --rerank-url) rerank_url="${2:-}"; shift 2 || die "--rerank-url needs a value" ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done
# The judge shares the translator's server unless told otherwise: one 16 GB GPU holds one model,
# so pointing them at different URLs without a second card would silently queue behind each other.
[[ -n "$judge_url" ]] || judge_url="$base_url"

python="$(project_python)"
assert_python_new_enough "$python"
has_httpx "$python" || die "httpx is not importable; run tools/setup_python_env.sh"
"$python" -c "import numpy" >/dev/null 2>&1 \
    || die "numpy is not importable (the embedding arm needs it); run tools/setup_python_env.sh"

require_command curl "Install curl, or check the servers by hand."

# Fail with the exact URL that is down rather than dying hours into a run -- never a silent skip.
check_server() {
    # check_server <url> <what> <how to start it>
    curl -fsS --max-time 5 "${1%/}/models" >/dev/null 2>&1 \
        || die "no server responding at ${1} (${2}). Start it with: ${3}"
    note "${2} ready at ${1}"
}
check_server "$base_url" "translator" "tools/serve_model.sh <model>"
check_server "$judge_url" "A/B judge" "tools/serve_model.sh <model>"
check_server "$embedding_url" "embedding server" "tools/serve_embeddings.sh"
if [[ -n "$rerank_url" ]]; then
    check_server "$rerank_url" "reranker" "tools/serve_reranker.sh"
    passthrough+=(--rerank-url "$rerank_url")
else
    note "no --rerank-url given: the hybrid+rerank arm cannot run"
fi

exec "$python" "${REPO_ROOT}/tools/lib/eval_translation.py" \
    --base-url "$base_url" --embedding-url "$embedding_url" --judge-url "$judge_url" \
    --python "$python" "${passthrough[@]}"
