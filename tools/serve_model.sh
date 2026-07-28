#!/usr/bin/env bash
# Start a local, OpenAI-compatible inference server for the translator to talk to.
#
# The translator only needs an OpenAI-compatible /v1/chat/completions endpoint; it does not
# care which server provides it. This script drives the two common local options and prints
# the exact --base-url / --model / --backend to hand the translator.
#
# The design target is a single 16 GB GPU (e.g. an RTX 5080): one model resident, every
# agent role differing only by prompt. Do not try to serve several models at once.
#
# Usage:
#   tools/serve_model.sh <model> [--engine ollama|llama] [--port N] [--host H]
#                                [--parallel N] [-- <extra server args>]
#
#   ollama:  <model> is an ollama tag,  e.g.  qwen3:14b   bielik   eurollm
#            tools/serve_model.sh qwen3:14b
#   llama:   <model> is a path to a .gguf file
#            tools/serve_model.sh ./models/Qwen3-14B-Q5_K_M.gguf --engine llama
#
#   --parallel N   server slots to run at once (default 2). The translator sends up to
#                  --concurrency requests concurrently (also 2 by default); serve at least
#                  that many slots or the requests serialise. For ollama this sets
#                  OLLAMA_NUM_PARALLEL on the daemon it starts.
#   -- <args>      everything after a literal -- is passed through to llama-server verbatim,
#                  for hardware-specific flags the design leaves out, e.g. GPU offload and
#                  context size:  ... --engine llama -- -ngl 999 -c 8192
#
# After it is up, translate with, for example:
#   PYTHONPATH=src python -m translator.cli --base-url <printed> --model <printed> \
#       --backend auto -c work/units.jsonl -j work/journal.jsonl

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

model=""
engine=""
port=""
host="127.0.0.1"
parallel=""
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) engine="${2:-}"; shift 2 || die "--engine needs a value" ;;
        --port) port="${2:-}"; shift 2 || die "--port needs a value" ;;
        --host) host="${2:-}"; shift 2 || die "--host needs a value" ;;
        --parallel) parallel="${2:-}"; shift 2 || die "--parallel needs a value" ;;
        --) shift; passthrough=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) [[ -z "$model" ]] || die "more than one model given: '$model' and '$1'"
           model="$1"; shift ;;
    esac
done

[[ -n "$model" ]] || die "no model given (see --help)"

: "${parallel:=2}"  # match the translator's default --concurrency, so requests do not serialise
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || die "--parallel must be a positive integer, got '${parallel}'"

# Auto-detect the engine when not named: a .gguf path means llama.cpp, otherwise ollama.
if [[ -z "$engine" ]]; then
    if [[ "$model" == *.gguf ]]; then engine="llama"; else engine="ollama"; fi
fi

case "$engine" in
    ollama)
        require_command ollama "Install it from https://ollama.com, then re-run."
        : "${port:=11434}"
        note "ensuring the ollama daemon is running"
        # OLLAMA_NUM_PARALLEL is read by the daemon at startup, so it only takes effect on a
        # daemon this script starts. Slot count is a property of the running daemon, not the
        # request, so a daemon already up keeps whatever it was launched with.
        export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$parallel}"
        # `ollama serve` is idempotent-ish: if one is already up it exits non-zero, which is
        # fine. Start one in the background only if none answers.
        if curl -fsS "http://${host}:${port}/api/tags" >/dev/null 2>&1; then
            note "an ollama daemon is already running; it keeps its existing slot count. If it "
            note "serves fewer than ${parallel} slots, restart it so --concurrency ${parallel} does not serialise."
        else
            ( ollama serve >/dev/null 2>&1 & )
            for _ in $(seq 1 20); do
                curl -fsS "http://${host}:${port}/api/tags" >/dev/null 2>&1 && break
                sleep 0.5
            done
            curl -fsS "http://${host}:${port}/api/tags" >/dev/null 2>&1 \
                || die "ollama daemon did not come up on ${host}:${port}"
            note "started ollama with OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
        fi
        note "pulling model '${model}' (no-op if already present)"
        ollama pull "$model" || die "could not pull '${model}'; check the tag with 'ollama list'"
        echo
        note "ollama is serving an OpenAI-compatible endpoint. Use:"
        echo "    --base-url http://${host}:${port}/v1  --model ${model}  --backend auto"
        ;;
    llama)
        require_command llama-server "Build llama.cpp (llama-server) or use --engine ollama."
        [[ -f "$model" ]] || die "GGUF file not found: ${model}"
        : "${port:=8080}"
        note "starting llama-server on ${host}:${port} with $(basename "$model") (${parallel} slots)"
        note "translate with: --base-url http://${host}:${port}/v1 --model local --backend auto"
        # --jinja activates the model's embedded chat template (also required for the
        # translator's in-band reasoning suppression to reach the template). --parallel matches
        # the translator's default concurrency. Context size and GPU layers are hardware-
        # specific and deliberately left out; supply them after a literal -- (they come last,
        # so they override these defaults), e.g.  -- -ngl 999 -c 8192.
        exec llama-server --host "$host" --port "$port" -m "$model" --alias local --jinja \
            --parallel "$parallel" "${passthrough[@]}"
        ;;
    *)
        die "unknown engine '${engine}'; expected 'ollama' or 'llama'"
        ;;
esac
