#!/usr/bin/env bash
# Run the test suite. The suite needs no server, GPU, or network.
#
# Usage: tools/run_tests.sh [pytest args...]
#   tools/run_tests.sh                    # the whole suite
#   tools/run_tests.sh tests/backend      # one area
#   tools/run_tests.sh -k prompt_hygiene  # by keyword
#   tools/run_tests.sh --coverage         # with a coverage report

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

coverage=0
pytest_args=()
for arg in "$@"; do
    case "$arg" in
        --coverage) coverage=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) pytest_args+=("$arg") ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"
"$python" -c "import pytest" >/dev/null 2>&1 \
    || die "pytest is not installed; run tools/setup_python_env.sh"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

if [[ "$coverage" == 1 ]]; then
    "$python" -c "import coverage" >/dev/null 2>&1 \
        || die "coverage is not installed; run tools/setup_python_env.sh"
    note "running the suite under coverage"
    # --branch, not just statements: the suite reported 100% statement coverage while four
    # branches were never taken, and two of those hid real defects (a dead narrowing arm and a
    # silent counter miscount). Statement coverage alone cannot see an untaken if/else arm.
    "$python" -m coverage run --branch --source=src -m pytest "${pytest_args[@]}"
    "$python" -m coverage report --skip-covered
else
    exec "$python" -m pytest "${pytest_args[@]}"
fi
