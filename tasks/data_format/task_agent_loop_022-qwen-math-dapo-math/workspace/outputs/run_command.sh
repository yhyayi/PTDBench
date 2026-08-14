#!/usr/bin/env bash
set -euo pipefail
# experiment_name 已在啟動腳本內烘焙為本 task leaf id（如 task_math_dapo_001）
bash examples/grpo_trainer/run_qwen2_5_math_1_5b_instruct_dapo_math_fsdp.sh
