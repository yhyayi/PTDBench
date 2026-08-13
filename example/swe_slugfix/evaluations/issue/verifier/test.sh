#!/usr/bin/env bash
set -euo pipefail

results="${PTDBENCH_RESULTS:-/results}"
answer="$results/agent_answer.md"
result_file="$results/result.json"

mkdir -p "$results"

python3 - "$answer" "$result_file" <<'PY'
import json
import re
import sys
from pathlib import Path

answer_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
text = answer_path.read_text(encoding="utf-8").lower() if answer_path.is_file() else ""

checks = {
    "mentions_punctuation_or_non_alphanumeric": bool(
        re.search(r"punctuation|non[- ]?alphanumeric|underscore", text)
    ),
    "mentions_repeated_separators": bool(
        re.search(r"collapse|repeated|multiple|run of|one at a time", text)
    ),
    "identifies_space_only_behavior": bool(
        re.search(r"space|replace\(.*['\"] ['\"]", text)
    ),
}
score = sum(checks.values()) / len(checks)
passed = answer_path.is_file() and bool(text.strip()) and score == 1.0

payload = {
    "passed": passed,
    "score": score,
    "metrics": {**checks, "answer_present": answer_path.is_file()},
}
result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
