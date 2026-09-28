#!/usr/bin/env bash
# Reference run for `harbor run -a oracle` (materialized as the Harbor task's
# solution/; regular agents never see it).
#
# The ReTool recipe has no injected defect, so the "fix" is empty: run the
# launch script with the budget the instruction fixes and put the logs where the
# grader reads them. This exercises image + GPUs + prepare outputs + verifier
# end to end.
set -euo pipefail

cd /workspace
script=examples/retool/retool_qwen3_4b_rl.sh

log_dir="$PTDBENCH_RESULTS/experiment_logs"
mkdir -p "$log_dir/wandb"
export LOG_DIR="$log_dir"
export WANDB_MODE=offline
export WANDB_DIR="$log_dir"
export WANDB_API_KEY="${WANDB_API_KEY:-ptdbench-offline}"

# Instruction: 100 rollouts. The recipe also saves a Megatron checkpoint (model +
# optimizer, tens of GB) every 10 rollouts; nothing is graded from them, so drop
# --save / --save-interval (slime then never saves).
sed -i \
  -e 's/--num-rollout 3000/--num-rollout 100/' \
  -e '/--save "\$OUTPUT_DIR/d' \
  -e '/--save-interval /d' \
  "$script"
grep -q -- '--num-rollout 100' "$script"
if grep -qE -- '--save "\$OUTPUT_DIR|--save-interval ' "$script"; then
  echo "solve.sh: failed to drop --save from $script" >&2
  exit 1
fi

bash "$script" 2>&1 | tee "$log_dir/train.log"
