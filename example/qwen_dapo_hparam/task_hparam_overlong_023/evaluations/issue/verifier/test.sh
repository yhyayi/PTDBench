#!/usr/bin/env bash
# Verifier scripts must retain LF endings for Linux containers.
set -u

verifier="${PTDBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
python3 "$verifier/grade_submission.py" --level issue "$@"
