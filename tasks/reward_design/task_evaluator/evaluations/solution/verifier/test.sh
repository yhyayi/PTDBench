#!/usr/bin/env bash
set -euo pipefail

verifier="${PTDBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$verifier/tests/grade_submission.py" "$@"
