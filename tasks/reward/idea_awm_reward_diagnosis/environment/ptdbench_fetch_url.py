#!/usr/bin/env python3
"""Fetch one pinned file by URL during the prepare stage.

Ported from AutoRLBench's global `autorlbench_fetch_url`: PTDBench keeps fetch
helpers per task, next to `prepare.sh` (invoked as
`python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch_url.py" ...`), so the same CLI
ships inside the task package. Runs inside the evaluation image, which has no
HF SDK -- standard library only.

Verifies byte size and SHA-256 before moving the file into place, so a
pinned-by-hash manifest stays authoritative even if the upstream repo moves.
An interrupted or mismatched fetch leaves no partial file behind.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


def fetch(url: str, sha256: str, size: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with partial.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        if total != size:
            raise RuntimeError(
                f"size mismatch for {url}: expected {size}, got {total}"
            )
        if digest.hexdigest() != sha256:
            raise RuntimeError(
                f"sha256 mismatch for {url}: expected {sha256}, "
                f"got {digest.hexdigest()}"
            )
        partial.replace(dest)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"[fetch] {url} -> {dest} ({total} bytes)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptdbench_fetch_url")
    parser.add_argument(
        "--type",
        choices=("model", "dataset"),
        default="dataset",
        help="kept for CLI parity with the fetch helpers; this tool is URL-based",
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fetch(args.url, args.sha256, args.size, args.dest)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
