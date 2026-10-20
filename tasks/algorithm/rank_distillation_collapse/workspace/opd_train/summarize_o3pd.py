#!/usr/bin/env python3
"""Export training/evaluation scalars from verl console logs to JSON and CSV."""

import argparse
import csv
import json
import re
from pathlib import Path


def read_steps(path):
    rows = {}
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    for line in path.read_text(errors="replace").splitlines():
        line = ansi.sub("", line)
        match = re.search(r"\bstep:(\d+) - ", line)
        if match is None:
            continue
        step = int(match[1])
        row = rows.setdefault(step, {"step": step})
        for item in line[match.end():].split(" - "):
            key, sep, value = item.partition(":")
            if sep:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return [rows[s] for s in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix (without extension)")
    args = parser.parse_args()
    runs = {str(path): read_steps(path) for path in args.logs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(runs, indent=2) + "\n")
    rows = [{"log": path, **row} for path, steps in runs.items() for row in steps]
    columns = ["log", "step"] + sorted({key for row in rows for key in row} - {"log", "step"})
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print({path: len(steps) for path, steps in runs.items()})


if __name__ == "__main__":
    main()
