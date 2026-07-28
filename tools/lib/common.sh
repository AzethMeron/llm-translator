#!/usr/bin/env bash
# Shared helpers for the tools/ scripts. Sourced, never executed directly.
#
# Every tool that does more than one thing lives in this directory (CLAUDE.md), and every
# one of them sources this file so the preconditions are checked one way, once.

# Guard against being run instead of sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "tools/lib/common.sh is a library; source it, do not run it." >&2
    exit 2
fi

set -euo pipefail

# Repository root, resolved from this file's location rather than the caller's cwd, so the
# scripts work from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
MIN_PYTHON_MINOR=11  # tomllib, used across the config loaders, is 3.11+

die() {
    echo "error: $*" >&2
    exit 1
}

note() {
    echo ">> $*" >&2
}

require_command() {
    # require_command <name> <install hint>
    command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found. $2"
}

# The interpreter to use: a python3 that is new enough. Prefers the project venv when it
# exists, so a script run outside setup still uses the pinned environment.
project_python() {
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        echo "${VENV_DIR}/bin/python"
        return
    fi
    require_command python3 "Install Python 3.${MIN_PYTHON_MINOR}+ and re-run."
    echo "python3"
}

assert_python_new_enough() {
    local python="$1"
    "$python" - "$MIN_PYTHON_MINOR" <<'PY' || die "Python 3.${MIN_PYTHON_MINOR}+ is required (tomllib, modern typing)."
import sys
minimum = int(sys.argv[1])
raise SystemExit(0 if sys.version_info[:2] >= (3, minimum) else 1)
PY
}

# True when the given interpreter can import the runtime dependency.
has_httpx() {
    "$1" -c "import httpx" >/dev/null 2>&1
}
