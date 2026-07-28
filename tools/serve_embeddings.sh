#!/usr/bin/env bash
# Start a local embedding server for the OPTIONAL semantic reference retriever.
#
# The translator's default reference retriever is lexical (stdlib-only, no server beyond the one
# translating). The embedding retriever (--embedding-url) instead needs an OpenAI-compatible
# /v1/embeddings endpoint; this starts one with llama.cpp, alongside your generation server. A
# 300M embedding model is tiny, so it fits next to a resident translation model on one GPU.
#
# Usage:
#   tools/serve_embeddings.sh                     # EmbeddingGemma (auto-downloaded), port 8081
#   tools/serve_embeddings.sh --model bge-m3.gguf # a local embedding GGUF instead
#   tools/serve_embeddings.sh [--port N] [--host H] [-- <extra llama-server args>]
#
# Then translate with the embedding retriever:
#   ... python -m translator.cli --embedding-url http://127.0.0.1:8081/v1 --reference <corpus> ...
#   (and raise context.reference_min_score to ~0.55 -- embedding cosines cluster higher than
#    the lexical floor; see docs/reference-translations.md)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

model=""
port="8081"
host="127.0.0.1"
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model="${2:-}"; shift 2 || die "--model needs a value" ;;
        --port) port="${2:-}"; shift 2 || die "--port needs a value" ;;
        --host) host="${2:-}"; shift 2 || die "--host needs a value" ;;
        --) shift; passthrough=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) die "unexpected argument '$1' (see --help)" ;;
    esac
done
[[ "$port" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer, got '${port}'"

require_command llama-server "Build llama.cpp (llama-server) or put it on PATH."

# --embeddings restricts the server to embedding requests; --pooling/--embd-normalize use the
# model's defaults. EmbeddingGemma is fetched on first use; a supplied GGUF is served as-is.
if [[ -n "$model" ]]; then
    [[ -f "$model" ]] || die "embedding GGUF not found: ${model}"
    note "starting embedding server on ${host}:${port} with $(basename "$model")"
    model_args=(-m "$model")
else
    note "starting embedding server on ${host}:${port} with EmbeddingGemma (auto-downloaded)"
    model_args=(--embd-gemma-default)
fi
note "translate with: --embedding-url http://${host}:${port}/v1 (raise reference_min_score ~0.55)"
exec llama-server --embeddings "${model_args[@]}" --host "$host" --port "$port" -ngl 99 \
    "${passthrough[@]}"
