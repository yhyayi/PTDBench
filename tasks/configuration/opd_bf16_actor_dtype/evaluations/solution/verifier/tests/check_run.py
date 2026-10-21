#!/usr/bin/env python3
"""Judge the shortened OPD run launched by test.sh.

Checks (all must pass):
  completed_steps  every step 1..STEPS logged its metrics
  weights_updated  after STEPS optimizer steps, enough of the actor's weights changed in their bf16 value
                   (the precision the forward pass sees) relative to the initial student checkpoint
  kl_progress      the late-window per-token reverse KL sits closer to the reference repair than to the
                   buggy run, both recorded in the oracle under the same shortened configuration

The fraction of bf16-representable grad norms is reported for diagnosis only, so repairs that keep bf16
parameters with an fp32 master copy or stochastic rounding are judged by their behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

DRIFT_MIN = 0.025
PROGRESS_MIN = 0.5
ADV_KEY = "critic/advantages/mean"
GRAD_KEY = "actor/grad_norm"


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        if " step:" not in line or "actor/pg_loss" not in line:
            continue
        row: dict[str, float] = {}
        for item in line[line.index("step:") :].strip().split(" - "):
            key, _, value = item.partition(":")
            try:
                row[key] = float(value)
            except ValueError:
                pass
        if "step" in row:
            rows[int(row["step"])] = row
    return rows


def is_bf16_exact(x: float) -> bool:
    if x == 0 or not math.isfinite(x):
        return x == 0
    exponent = math.floor(math.log2(abs(x)))
    scaled = x / 2 ** (exponent - 7)
    return abs(scaled - round(scaled)) < 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_drift(ckpt_dir: Path, steps: int, init_dir: Path) -> dict[str, float]:
    import torch
    from safetensors.torch import load_file

    files = sorted((ckpt_dir / f"global_step_{steps}" / "actor" / "huggingface").glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no saved actor weights under {ckpt_dir}/global_step_{steps}/actor/huggingface")
    trained: dict = {}
    for f in files:
        trained.update(load_file(str(f)))
    initial = load_file(str(init_dir / "model.safetensors"))
    missing = sorted(set(initial) - set(trained))
    if missing:
        raise KeyError(f"saved actor is missing {len(missing)} tensors, e.g. {missing[0]}")
    total = changed = 0
    diff_sq = norm_sq = 0.0
    for name, w0 in initial.items():
        w = trained[name]
        total += w0.numel()
        changed += (w.to(torch.bfloat16) != w0.to(torch.bfloat16)).sum().item()
        diff_sq += (w.float() - w0.float()).pow(2).sum().item()
        norm_sq += w0.float().pow(2).sum().item()
    return {
        "saved_dtype": str(next(iter(trained.values())).dtype),
        "frac_bf16_view_changed": changed / total,
        "rel_change_norm": math.sqrt(diff_sq / norm_sq),
    }


def run_metrics(rows: dict[int, dict[str, float]], steps: int, late: tuple[int, int]) -> dict:
    adv = {s: rows[s][ADV_KEY] for s in rows if ADV_KEY in rows[s]}
    grad = [rows[s][GRAD_KEY] for s in sorted(rows) if GRAD_KEY in rows[s]]
    late_steps = [s for s in range(late[0], late[1] + 1) if s in adv]
    return {
        "logged_steps": len(rows),
        "late_adv_mean": sum(adv[s] for s in late_steps) / len(late_steps) if late_steps else None,
        "early_adv_mean": sum(adv[s] for s in (1, 2, 3) if s in adv) / max(1, sum(s in adv for s in (1, 2, 3))) if adv else None,
        "late_grad_norm_mean": (sum(rows[s][GRAD_KEY] for s in late_steps if GRAD_KEY in rows[s]) / len(late_steps)) if late_steps else None,
        "frac_grad_norm_bf16_exact": (sum(map(is_bf16_exact, grad)) / len(grad)) if grad else None,
        "adv_mean_by_step": [adv.get(s) for s in range(1, steps + 1)],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--train-status", type=int, default=0)
    ap.add_argument("--init-model", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--oracle", type=Path)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--calibrate", action="store_true", help="record metrics without judging kl_progress")
    args = ap.parse_args()

    expected = json.loads(args.expected.read_text())
    late = tuple(expected["late_window"])
    checks: dict[str, bool] = {}
    metrics: dict = {"train_exit_status": args.train_status}
    errors: list[str] = []

    rows = parse_log(args.log)
    metrics.update(run_metrics(rows, args.steps, late))
    checks["completed_steps"] = all(s in rows and ADV_KEY in rows[s] for s in range(1, args.steps + 1))

    try:
        init_file = args.init_model / "model.safetensors"
        if sha256(init_file) != expected["init_model_sha256"]:
            raise ValueError("initial student checkpoint does not match the prepared model")
        metrics.update(weight_drift(args.ckpt_dir, args.steps, args.init_model))
        checks["weights_updated"] = metrics["frac_bf16_view_changed"] >= DRIFT_MIN
    except Exception as exc:  # noqa: BLE001 - every failure is reported in result.json
        errors.append(f"weight drift: {exc}")
        checks["weights_updated"] = False

    if not args.calibrate:
        try:
            oracle = json.loads(args.oracle.read_text())
            buggy, reference = oracle["buggy"]["late_adv_mean"], oracle["reference"]["late_adv_mean"]
            if metrics["late_adv_mean"] is None:
                raise ValueError("late-window advantages missing from the run log")
            progress = (metrics["late_adv_mean"] - buggy) / (reference - buggy)
            metrics["kl_progress"] = progress
            metrics["oracle_late_adv_mean"] = {"buggy": buggy, "reference": reference}
            checks["kl_progress"] = progress >= PROGRESS_MIN
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kl progress: {exc}")
            checks["kl_progress"] = False

    passed = bool(checks) and all(checks.values()) and not args.calibrate
    result = {
        "passed": passed,
        "score": round(sum(checks.values()) / len(checks), 4),
        "metrics": {
            "checks": checks,
            "thresholds": {"frac_bf16_view_changed": DRIFT_MIN, "kl_progress": PROGRESS_MIN},
            **metrics,
            **({"errors": errors} if errors else {}),
        },
    }
    write_json(args.result, result)
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    for err in errors:
        print(f"ERROR {err}", file=sys.stderr)
    print(f"passed={passed} score={result['score']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
