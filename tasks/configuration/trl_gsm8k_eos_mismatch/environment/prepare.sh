#!/usr/bin/env bash
# trl_gsm8k_eos_mismatch: verify the frozen evidence before the replay rerun.
#
# Runs in the separate prepare-stage container: root, networked, with
# $PTDBENCH_WORKSPACE, $PTDBENCH_MODELS and $PTDBENCH_DATASETS writable.
#
# The diagnosis snapshot deliberately ships no weights (the SFT checkpoint is
# submitter-private) and the solution-level rerun needs none: it is a
# deterministic CPU replay of the retained rollout records, with the stopping
# tokens read from workspace model/generation_config.json -- the file the
# repair edits. Here we only verify that the frozen evidence the replay
# consumes is staged and non-empty.
set -euo pipefail

: "${PTDBENCH_WORKSPACE:?PTDBENCH_WORKSPACE must point to the staged workspace}"
test -s "$PTDBENCH_WORKSPACE/outputs/training.log"
test -s "$PTDBENCH_WORKSPACE/outputs/training_metrics.json"
test -s "$PTDBENCH_WORKSPACE/outputs/rollout_samples.jsonl"
test -s "$PTDBENCH_WORKSPACE/outputs/evaluation_metrics.json"
test -s "$PTDBENCH_WORKSPACE/model/generation_config.json"
test -s "$PTDBENCH_WORKSPACE/model/added_tokens.json"
