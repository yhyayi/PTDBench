#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 Ray eval dump 判斷組內 GRPO 歸一化後是否仍有更新信號。

對齊 slime/ray/buffer.py：
  centered = r - mean
  adv = centered / (sample_std + 1e-6)
通過：該環境至少有一個 prompt group 的 reward 非全相同（advantage 非零）。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch


def grpo_has_signal(rewards: Sequence[float]) -> Tuple[bool, float, List[float]]:
    t = torch.tensor(list(rewards), dtype=torch.float32)
    if t.numel() < 2:
        return False, 0.0, [0.0] * int(t.numel())
    if float(t.max() - t.min()) <= 0:
        return False, 0.0, [0.0] * int(t.numel())
    centered = t - t.mean()
    std = float(centered.std().item())
    adv = centered / (std + 1e-6)
    has = bool(torch.isfinite(adv).all().item() and (adv != 0).any().item())
    return has, std, [float(x) for x in adv.tolist()]


def analyze(raw: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
    samples = raw.get("samples") or []
    # group consecutive n_samples that share the same prompt index / environment+seed
    by_prompt: Dict[str, List[float]] = defaultdict(list)
    env_of: Dict[str, str] = {}

    for s in samples:
        meta = s.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        env = meta.get("environment", "UNKNOWN")
        cfg = meta.get("config") or {}
        seed = cfg.get("seed", s.get("prompt_idx", 0))
        key = f"{env}::seed={seed}::p={s.get('prompt_idx')}"
        by_prompt[key].append(float(s.get("reward", -1.0)))
        env_of[key] = env

    # If dump already grouped, also accept groups list
    if not by_prompt and "groups" in raw:
        for g in raw["groups"]:
            env = g["environment"]
            key = f"{env}::p={g.get('prompt_idx')}"
            by_prompt[key] = list(g["rewards"])
            env_of[key] = env

    env_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n_groups": 0, "n_signal": 0, "groups": []}
    )
    for key, rewards in by_prompt.items():
        # truncate / pad to n_samples if needed
        rs = rewards[:n_samples] if len(rewards) >= n_samples else rewards
        if len(rs) < 2:
            continue
        has, std, adv = grpo_has_signal(rs)
        env = env_of[key]
        st = env_stats[env]
        st["n_groups"] += 1
        if has:
            st["n_signal"] += 1
        st["groups"].append(
            {
                "key": key,
                "rewards": rs,
                "reward_std": std,
                "advantages": adv,
                "has_signal": has,
            }
        )

    passed = sorted([e for e, st in env_stats.items() if st["n_signal"] > 0])
    failed = sorted([e for e, st in env_stats.items() if st["n_signal"] == 0])
    return {
        "n_envs": len(env_stats),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "passed": passed,
        "failed": failed,
        "env_stats": {k: dict(v) for k, v in sorted(env_stats.items())},
        "pass_rule": ">=1 prompt group with nonzero GRPO advantages after mean-center/std-norm",
        "n_samples_per_prompt": n_samples,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="scan_eval_raw.json")
    p.add_argument("--out", required=True, help="pre.json")
    p.add_argument("--report", default=None)
    p.add_argument("--n-samples", type=int, default=8)
    args = p.parse_args()

    with open(args.raw) as f:
        raw = json.load(f)

    report = analyze(raw, n_samples=args.n_samples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report["passed"], f, indent=2, ensure_ascii=False)
        f.write("\n")

    report_path = args.report or (os.path.splitext(args.out)[0] + "_report.json")
    # trim group details in report for size? keep them — useful
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"[analyze] passed={report['n_passed']}/{report['n_envs']} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
