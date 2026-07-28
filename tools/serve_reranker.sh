#!/usr/bin/env bash
# Start a local cross-encoder reranker server for the OPTIONAL hybrid retriever's rerank stage.
#
# A reranker re-scores a short candidate pool a first-stage retriever already narrowed down; it
# is a second, much smaller server alongside your translation server and (if used)
# tools/serve_embeddings.sh -- all three fit on one GPU together, since a reranker model is tiny.
#
# Usage:
#   tools/serve_reranker.sh                        # bge-reranker-v2-m3 Q8_0 (auto-downloaded), port 8082
#   tools/serve_reranker.sh --model reranker.gguf  # a local reranker GGUF instead
#   tools/serve_reranker.sh --hf-repo user/repo:quant   # a different Hugging Face GGUF repo/quant
#   tools/serve_reranker.sh [--port N] [--host H] [-- <extra llama-server args>]
#
# Then translate with hybrid retrieval + reranking:
#   ... python -m translator.cli --embedding-url http://127.0.0.1:8081/v1 --hybrid \
#       --rerank-url http://127.0.0.1:8082/v1 --reference <corpus> ...
#   (see docs/reference-translations.md for the candidate-pool and floor knobs)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

# gpustack/bge-reranker-v2-m3-GGUF: a verified, Apache-2.0-licensed, multilingual GGUF conversion
# of BAAI/bge-reranker-v2-m3 (the model bge-m3's own authors recommend for hybrid-retrieval
# reranking). Q8_0 keeps near-FP16 quality at a small footprint (~600 MB) -- the reranker is by
# far the smallest of the three servers, so quality is worth spending the extra bytes on.
DEFAULT_HF_REPO="gpustack/bge-reranker-v2-m3-GGUF:Q8_0"

model=""
hf_repo=""
port="8082"
host="127.0.0.1"
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model="${2:-}"; shift 2 || die "--model needs a value" ;;
        --hf-repo) hf_repo="${2:-}"; shift 2 || die "--hf-repo needs a value" ;;
        --port) port="${2:-}"; shift 2 || die "--port needs a value" ;;
        --host) host="${2:-}"; shift 2 || die "--host needs a value" ;;
        --) shift; passthrough=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) die "unexpected argument '$1' (see --help)" ;;
    esac
done
[[ "$port" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer, got '${port}'"
[[ -z "$model" || -z "$hf_repo" ]] || die "--model and --hf-repo are mutually exclusive"

require_command llama-server "Build llama.cpp (llama-server) or put it on PATH."

# --reranking enables the /rerank, /v1/rerank routes; --embedding --pooling rank is what the
# server documentation requires alongside it for an actual reranker model (the "rank" pooling
# type is what turns the model's output into one relevance score per document, rather than a
# vector). -hf downloads and caches the GGUF on first use, verified by llama.cpp's own
# downloader, exactly like tools/serve_embeddings.sh's --embd-gemma-default.
if [[ -n "$model" ]]; then
    [[ -f "$model" ]] || die "reranker GGUF not found: ${model}"
    note "starting reranker server on ${host}:${port} with $(basename "$model")"
    model_args=(-m "$model")
else
    repo="${hf_repo:-$DEFAULT_HF_REPO}"
    note "starting reranker server on ${host}:${port} with ${repo} (auto-downloaded)"
    model_args=(-hf "$repo")
fi
note "translate with: --rerank-url http://${host}:${port}/v1 (needs --hybrid and --embedding-url too)"
exec llama-server --reranking --embedding --pooling rank "${model_args[@]}" \
    --host "$host" --port "$port" -ngl 99 "${passthrough[@]}"
