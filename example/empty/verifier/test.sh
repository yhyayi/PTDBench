#!/usr/bin/env bash
set -euo pipefail

workspace="${PTDBENCH_WORKSPACE:-/workspace}"
results="${PTDBENCH_RESULTS:-/results}"

test -d "$workspace"
mkdir -p "$results"

python3 - "$results/result.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"passed": True, "score": 1.0, "metrics": {"template_smoke": 1.0}}, f)
PY
