#!/usr/bin/env bash
set -u

verifier="${PTDBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
python3 "$verifier/grade_submission.py" --level solution "$@"
