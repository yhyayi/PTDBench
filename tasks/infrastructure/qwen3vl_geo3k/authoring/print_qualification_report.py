#!/usr/bin/env python3
"""Print saved qualification results without starting training or loading weights."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    status_path = ROOT / "authoring/qualification.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    runs = args.run_dir or Path(status.get("run_directory", ROOT / "authoring/runs"))
    if not runs.is_absolute():
        runs = ROOT / runs
    output = {"qualification": status, "run_directory": str(runs)}
    verdict = runs / "paired_verdict.json"
    if verdict.exists():
        output["paired_verdict"] = json.loads(verdict.read_text())
    for name in ("candidate", "healthy"):
        path = runs / f"{name}_scores.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())["records"]
        output[name] = []
        for row in records:
            differences = [abs(a - b) for a, b in zip(row["vllm_logps"], row["trainer_logps"], strict=True)]
            output[name].append({"case": row["name"], "max_logprob_abs_diff_nat": max(differences),
                                 "mean_logprob_abs_diff_nat": sum(differences) / len(differences),
                                 "position_max_diff": row["engine_vs_reference_positions_max_diff"],
                                 "processor_has_get_rope_index": row["processor_has_get_rope_index"]})
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
