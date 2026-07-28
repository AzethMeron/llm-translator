#!/usr/bin/env bash
# Verify the environment can run the translator and its tests, and say precisely what is
# wrong when it cannot. Read-only: it changes nothing, it only reports.
#
# Usage: tools/check_environment.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

case "${1:-}" in
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) die "unknown argument '${1}' (see --help)" ;;
esac

status=0
report() {
    # report <ok|FAIL> <message>
    printf '  [%s] %s\n' "$1" "$2"
    [[ "$1" == "ok" ]] || status=1
}

echo "environment check for llm-translator"

# Python interpreter and version.
if command -v python3 >/dev/null 2>&1; then
    if assert_python_new_enough python3 2>/dev/null; then
        report ok "python3 $("python3" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
    else
        report FAIL "python3 is older than 3.${MIN_PYTHON_MINOR}; install a newer one"
    fi
else
    report FAIL "python3 not found on PATH"
fi

# Virtual environment.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    report ok "virtual environment present at .venv/"
    python="${VENV_DIR}/bin/python"
else
    report FAIL "no .venv/ — run tools/setup_python_env.sh"
    python="python3"
fi

# Runtime dependency.
if command -v "$python" >/dev/null 2>&1 && has_httpx "$python"; then
    report ok "httpx importable (translator runtime dependency)"
else
    report FAIL "httpx not importable — run tools/setup_python_env.sh"
fi

# Optional dependency: only the embedding retriever (--embedding-url) and the hybrid retriever
# (--hybrid, which composes it) need numpy, and it is imported lazily, so its absence is a note,
# not a failure -- the core engine runs without it. The reranking stage (--rerank-url) needs no
# separate dependency; it only needs httpx, already required above.
if command -v "$python" >/dev/null 2>&1 && "$python" -c "import numpy" >/dev/null 2>&1; then
    report ok "numpy importable (optional; for the embedding or hybrid reference retriever)"
else
    echo "  [note] numpy not importable — needed only for the optional --embedding-url/--hybrid retrievers"
fi

# Test dependencies. Both are hard requirements of the suite, not optional extras: hypothesis
# backs tests/property/, so without it the suite fails on a bare ImportError at collection --
# which is exactly the misconfiguration this script exists to name before it happens.
if command -v "$python" >/dev/null 2>&1 && "$python" -c "import pytest" >/dev/null 2>&1; then
    report ok "pytest importable (test dependency)"
else
    report FAIL "pytest not importable — run tools/setup_python_env.sh"
fi

if command -v "$python" >/dev/null 2>&1 && "$python" -c "import hypothesis" >/dev/null 2>&1; then
    report ok "hypothesis importable (test dependency; property suite)"
else
    report FAIL "hypothesis not importable — run tools/setup_python_env.sh"
fi

# The packages themselves import (catches a broken checkout).
if command -v "$python" >/dev/null 2>&1 && \
        PYTHONPATH="${REPO_ROOT}/src" "$python" -c "import transunit, translator" >/dev/null 2>&1; then
    report ok "transunit and translator import cleanly"
else
    report FAIL "the source packages do not import — the checkout may be incomplete"
fi

# An inference server is optional and only needed for a real run, so its absence is a note,
# not a failure.
if command -v ollama >/dev/null 2>&1; then
    report ok "ollama found (optional; for a real translation run)"
elif command -v llama-server >/dev/null 2>&1; then
    report ok "llama-server found (optional; for a real translation run)"
else
    echo "  [note] no ollama or llama-server found — needed only for a real run, not for tests"
fi

echo
if [[ "$status" == 0 ]]; then
    echo "environment is ready. Run the tests with tools/run_tests.sh"
else
    echo "environment is NOT ready; fix the [FAIL] items above." >&2
fi
exit "$status"
