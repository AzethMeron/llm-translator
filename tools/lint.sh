#!/usr/bin/env bash
# Static analysis: ruff (lint) and mypy (types). The ruleset is pinned in ruff.toml.
#
# Usage: tools/lint.sh [--fix]
#   --fix   apply ruff's safe autofixes before reporting

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

fix=0
for arg in "$@"; do
    case "$arg" in
        --fix) fix=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$arg' (see --help)" ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"

# Both tools are invoked through the project interpreter (python -m ...) rather than as bare
# commands, so they resolve from .venv/ without the caller having to activate it — the same
# contract the other tools honour.
"$python" -c "import ruff" >/dev/null 2>&1 \
    || die "ruff is not installed; run tools/setup_python_env.sh"
"$python" -c "import mypy" >/dev/null 2>&1 \
    || die "mypy is not installed; run tools/setup_python_env.sh"

cd "$REPO_ROOT"
status=0

note "ruff (config: ruff.toml)"
if [[ "$fix" == 1 ]]; then
    "$python" -m ruff check src tests --fix || status=1
else
    "$python" -m ruff check src tests || status=1
fi

note "mypy (src)"
"$python" -m mypy src --ignore-missing-imports || status=1

if [[ "$status" == 0 ]]; then
    note "clean."
else
    echo "lint found issues (see above)." >&2
fi
exit "$status"
