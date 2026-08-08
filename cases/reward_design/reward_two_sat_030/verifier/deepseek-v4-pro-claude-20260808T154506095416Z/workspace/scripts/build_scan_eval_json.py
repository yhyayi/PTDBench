#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""為全部 Gym 環境各生成 N 條 eval prompt（含 metadata），供 Ray eval 掃描 GRPO 信號。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


def build_one(
    environment: str,
    seed: int,
    answer_markers: Tuple[str, str],
) -> Optional[Dict[str, Any]]:
    from Gym.environments import identifier2environment
    from Gym.parameter_controllers import identifier2controller

    ctrl = identifier2controller[environment]()
    params = ctrl.get_parameter_list()
    if not params:
        return None
    parameter = random.Random(seed).choice(params)
    problem = identifier2environment[environment](answer_markers=answer_markers)
    if not problem.generator(seed=seed, parameter=parameter):
        return None
    return {
        "user_prompt": problem.prompt_generator(),
        "metadata": json.dumps(
            {
                "environment": environment,
                "config": problem.get_config(),
            },
            ensure_ascii=False,
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--prompts-per-env", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--envs", nargs="*", default=None)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    from Gym.environments import identifier2environment

    all_envs = sorted(identifier2environment.keys())
    if args.envs:
        envs = list(args.envs)
    else:
        envs = all_envs[args.offset :]
        if args.limit > 0:
            envs = envs[: args.limit]

    markers = (r"\boxed{", r"}")
    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for ei, name in enumerate(envs):
        ok_n = 0
        attempt = 0
        while ok_n < args.prompts_per_env and attempt < args.prompts_per_env * 20:
            seed = args.seed + ei * 100003 + attempt * 97 + ok_n * 13
            attempt += 1
            try:
                row = build_one(name, seed=seed, answer_markers=markers)
            except Exception as e:
                failed.append({"environment": name, "error": f"{type(e).__name__}: {e}"})
                break
            if row is None:
                continue
            rows.append(row)
            ok_n += 1
        if ok_n < args.prompts_per_env:
            failed.append(
                {
                    "environment": name,
                    "error": f"only_built_{ok_n}_of_{args.prompts_per_env}",
                }
            )
        if (ei + 1) % 20 == 0 or ei + 1 == len(envs):
            print(f"[build] {ei+1}/{len(envs)} prompts={len(rows)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    fail_path = os.path.splitext(args.out)[0] + "_failed.json"
    with open(fail_path, "w") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"[build] wrote {len(rows)} prompts for {len(envs)} envs -> {args.out} "
        f"(failed_entries={len(failed)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
