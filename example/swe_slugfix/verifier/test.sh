#!/usr/bin/env bash
set -euo pipefail

case_root="${PTDBENCH_CASE_ROOT:-/case}"
workspace="${PTDBENCH_WORKSPACE:-/workspace}"
results="${PTDBENCH_RESULTS:-/results}"
tests_dir="$case_root/verifier/tests"
result_file="$results/result.json"

mkdir -p "$results"

python3 - "$workspace" "$tests_dir" "$result_file" <<'PY'
import json
import os
import sys
import unittest

workspace = sys.argv[1]
tests_dir = sys.argv[2]
result_file = sys.argv[3]

sys.path.insert(0, workspace)

suite = unittest.defaultTestLoader.discover(tests_dir, pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

passed = result.wasSuccessful()
payload = {
    "passed": passed,
    "score": 1.0 if passed else 0.0,
    "metrics": {
        "hidden_pass_rate": 1.0 if passed else 0.0,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    },
}

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

sys.exit(0 if passed else 1)
PY
