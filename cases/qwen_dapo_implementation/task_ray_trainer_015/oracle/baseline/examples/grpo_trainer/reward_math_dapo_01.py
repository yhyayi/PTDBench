"""0/1 reward wrapper around math_dapo.compute_score (+1/-1 -> 1/0).

Set BENCH_REWARD_DEBUG=0 to disable per-sample dump.
Logs append to outputs/reward_debug.jsonl and print a short line to stdout.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from verl.utils.reward_score.math_dapo import compute_score as _math_dapo_compute_score

_LOCK = threading.Lock()
_COUNT = 0
_DEBUG = os.environ.get("BENCH_REWARD_DEBUG", "0") != "0"
_LOG_PATH = Path(os.environ.get("BENCH_REWARD_DEBUG_PATH", "outputs/reward_debug.jsonl"))
# print at most this many samples to stdout (jsonl still gets all when debug on)
_PRINT_EVERY = int(os.environ.get("BENCH_REWARD_DEBUG_PRINT_EVERY", "1"))  # 1 = every sample
_MAX_SOL_CHARS = int(os.environ.get("BENCH_REWARD_DEBUG_SOL_CHARS", "800"))


def _append_debug(record: dict) -> None:
    global _COUNT
    if not _DEBUG:
        return
    with _LOCK:
        _COUNT += 1
        record = {**record, "n": _COUNT, "ts": time.time()}
        try:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[reward_debug] write_failed: {e}", flush=True)
        if _PRINT_EVERY > 0 and (_COUNT % _PRINT_EVERY == 0):
            sol = record.get("solution_str", "")
            print(
                "[reward_debug] "
                f"n={_COUNT} score={record.get('score')} pred={record.get('pred')!r} "
                f"gt={record.get('ground_truth')!r} "
                f"has_Answer={'Answer:' in sol} has_boxed={'\\boxed' in sol}\n"
                f"--- solution (tail {_MAX_SOL_CHARS}) ---\n{sol[-_MAX_SOL_CHARS:]}\n"
                f"--- end ---",
                flush=True,
            )


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Any = None,
    *,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    **kwargs,
) -> dict:
    """Same extract/verify as math_dapo, but score is 1.0 correct / 0.0 incorrect."""
    del data_source, extra_info, kwargs  # unused; kept for reward-manager call signature
    out = _math_dapo_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        strict_box_verify=strict_box_verify,
        pause_tokens_index=pause_tokens_index,
    )
    # math_dapo returns score in {+1.0, -1.0}; map to {1.0, 0.0}
    out = dict(out)
    out["score"] = 1.0 if out.get("score", 0.0) > 0 else 0.0
    _append_debug(
        {
            "ground_truth": ground_truth,
            "pred": out.get("pred"),
            "score": out.get("score"),
            "acc": out.get("acc"),
            "solution_str": solution_str if isinstance(solution_str, str) else str(solution_str),
        }
    )
    return out
