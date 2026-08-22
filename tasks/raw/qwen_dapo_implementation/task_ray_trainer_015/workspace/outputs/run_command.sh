#!/usr/bin/env bash
set -euo pipefail
# experiment_name is baked in the launch script as this task leaf id (task_ray_trainer_015)
bash examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh
