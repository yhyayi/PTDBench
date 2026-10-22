#!/usr/bin/env python3
"""Offline package QA over recorded scores; not the fresh GPU solution grader."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluations/solution/verifier/tests"))
from metrics import repair_verdict


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("side", choices=("candidate", "control"))
    args = parser.parse_args()
    oracle = ROOT / "evaluations/solution/oracle"
    name = "candidate" if args.side == "candidate" else "healthy"
    data = json.loads((oracle / "paired_measurements" / f"{name}_scores.json").read_text())
    spec = json.loads((ROOT / "evaluations/solution/verifier/resources/grader_spec.json").read_text())
    logs = ROOT / "workspace/logs/recorded_run" if args.side == "candidate" else oracle / "control_run"
    series = json.loads((logs / "metrics.json").read_text())
    steps = {row["step"] for row in series}
    verdict = repair_verdict(data, spec)
    complete = all(s in steps for s in range(1, 21))
    print(json.dumps({"side": args.side, "recorded_20_steps_present": complete,
                      "parity": verdict, "scope": "Offline recorded-evidence QA; no new GPU run"}, indent=2))
    return 0 if complete and verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
