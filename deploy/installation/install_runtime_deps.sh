#!/bin/bash
set -eo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <requirements.txt> [pip args...]" >&2
    exit 2
fi

REQUIREMENTS_FILE="$1"
shift

DEFAULT_DLS2_PYTHON="/usr/bin/dls2/python/python3"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-${DLS2_PYTHON:-$DEFAULT_DLS2_PYTHON}}"
BOOTSTRAP_PYTHON_EXECUTABLE="${BOOTSTRAP_PYTHON_EXECUTABLE:-/usr/bin/python3}"

if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    if [ "$PYTHON_EXECUTABLE" = "$DEFAULT_DLS2_PYTHON" ]; then
        if [ ! -x "$BOOTSTRAP_PYTHON_EXECUTABLE" ]; then
            echo "install_runtime_deps.sh: missing bootstrap Python at $BOOTSTRAP_PYTHON_EXECUTABLE" >&2
            exit 127
        fi

        "$BOOTSTRAP_PYTHON_EXECUTABLE" -m venv --copies /usr/lib/dls2/python
    else
        echo "install_runtime_deps.sh: missing shared DLS2 interpreter at $PYTHON_EXECUTABLE" >&2
        exit 127
    fi
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "install_runtime_deps.sh: missing requirements file at $REQUIREMENTS_FILE" >&2
    exit 1
fi

exec "$PYTHON_EXECUTABLE" -m pip install -r "$REQUIREMENTS_FILE" "$@"
