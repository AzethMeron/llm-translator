#!/usr/bin/env bash
# Create the pinned Python environment from requirements.txt.
#
# Self-contained: this is the one command that makes the project runnable from a fresh
# checkout. It creates .venv/ and installs the exact pinned versions, transitive ones
# included, so a months-old run reproduces.
#
# Usage: tools/setup_python_env.sh [--recreate]
#   --recreate   delete an existing .venv/ first

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

recreate=0
for arg in "$@"; do
    case "$arg" in
        --recreate) recreate=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$arg' (see --help)" ;;
    esac
done

[[ -f "$REQUIREMENTS" ]] || die "requirements.txt not found at ${REQUIREMENTS}"

require_command python3 "Install Python 3.${MIN_PYTHON_MINOR}+ first."
assert_python_new_enough python3

if [[ "$recreate" == 1 && -d "$VENV_DIR" ]]; then
    note "removing existing ${VENV_DIR}"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    note "creating virtual environment at ${VENV_DIR}"
    python3 -m venv "$VENV_DIR" || die "could not create a virtual environment; is the venv module installed?"
fi

venv_python="${VENV_DIR}/bin/python"
[[ -x "$venv_python" ]] || die "virtual environment is missing its interpreter at ${venv_python}; try --recreate"

note "upgrading pip"
"$venv_python" -m pip install --quiet --upgrade pip || die "could not upgrade pip"

note "installing pinned dependencies"
"$venv_python" -m pip install --quiet -r "$REQUIREMENTS" \
    || die "dependency installation failed; see the pip output above"

has_httpx "$venv_python" || die "httpx did not install; the environment is incomplete"

note "done. Activate with:  source ${VENV_DIR}/bin/activate"
note "or just run the tools, which use ${VENV_DIR} automatically."
