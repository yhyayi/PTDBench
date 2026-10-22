#!/usr/bin/env bash
# openrsi_reward_direction — assets required before training.
#
# This is the only stage with network access. It runs in the environment image,
# in a throwaway container, and is the only stage mounted to the shared download
# cache ($PTDBENCH_MODELS / $PTDBENCH_DATASETS).
#
# Assets to deliver:
#   1. the policy model (Qwen3-4B-Thinking-2507, dense)
#   2. the CPU-runnable OpenMLE task pool (100 training tasks)
#   3. the held-out evaluation task subset (MLE-Bench Lite CPU split)
#   4. public leaderboards for every EVALUATION task   <-- required, see below
#
# Every fetch must pin an immutable revision (a commit hash, never main/latest)
# and carry sha256 or size verification. A verification failure is a hard
# failure; it must not degrade into a retry.
#
# ---------------------------------------------------------------------------
# WHY LEADERBOARDS ARE REQUIRED
#
# The graded observable is `eval/validation_tts_draft176/medal_rate`, emitted by
# patch_eval_logger.py. Medal computation reads
#   <LEADERBOARD_ROOTS>/<task_name>/info/public_leaderboard.csv
# (falling back to <task_name>/leaderboard.csv). If a leaderboard is missing the
# score is SILENTLY mapped to medal "None" -- no error, no warning -- which
# drives medal_rate toward 0 for that task. A run whose leaderboards are absent
# therefore looks identical to a run that failed, and the threshold cannot be
# calibrated. The launcher's own precheck enforces this when
# REQUIRE_ALL_PUBLIC_LEADERBOARDS=1.
# ---------------------------------------------------------------------------
#
# Record in readme.md: resource name, source URL, revision, size, SHA-256,
# and license for each asset.
set -euo pipefail

echo "[openrsi_reward_direction] prepare is not yet filled in." >&2
echo >&2
echo "Required assets:" >&2
echo "  1. model weights       Qwen3-4B-Thinking-2507 (HF commit hash + sha256)" >&2
echo "  2. training task pool  100 CPU-runnable OpenMLE tasks, lower-is-better 40-50%" >&2
echo "  3. eval task subset    MLE-Bench Lite CPU split (tabular/text/seq2seq)" >&2
echo "  4. leaderboards        public_leaderboard.csv for EVERY eval task" >&2
echo >&2
echo "Each fetch needs an immutable revision and a sha256/size check." >&2
echo "See docs/DATA_SELECTION.md for the selection pipeline and" >&2
echo "docs/RUNBOOK_for_experiment_runner.md for the full procedure." >&2
exit 1
