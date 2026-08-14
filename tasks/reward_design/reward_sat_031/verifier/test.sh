#!/usr/bin/env bash
set -uo pipefail

VERIFIER_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
python3 "${VERIFIER_DIR}/run_evaluation.py" "$@"
exit $?
