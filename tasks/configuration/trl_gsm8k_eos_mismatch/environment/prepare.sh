#!/usr/bin/env bash
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?PTDBENCH_WORKSPACE must point to the staged workspace}"
test -s "$PTDBENCH_WORKSPACE/outputs/training.log"
test -s "$PTDBENCH_WORKSPACE/outputs/training_metrics.json"
test -s "$PTDBENCH_WORKSPACE/outputs/rollout_samples.jsonl"
test -s "$PTDBENCH_WORKSPACE/outputs/evaluation_metrics.json"
test -s "$PTDBENCH_WORKSPACE/model/generation_config.json"
