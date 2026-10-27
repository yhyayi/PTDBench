# Repair and rerun the staged experiment

Inspect the source, launch script, effective configuration, and frozen evidence in `/workspace`. Identify and repair the implementation defect, then run the experiment as a foreground process and wait for completion. Do not use the network, `nohup`, `&`, `tmux`, `screen`, or any detach mechanism.

The pinned model and VERL-formatted GSM8K files are prepared under `/workspace/assets/`. Use `/workspace/run.sh` without changing the dataset, evaluation subset, seed, rollout count, response limit, or other recorded hyperparameters. The run must complete 50 training steps with held-out evaluation every 5 steps. The script writes fresh logs and offline W&B files below `$PTDBENCH_RESULTS/experiment_logs`; the solution is graded only from those fresh logs. Do not copy, edit, or fabricate metric values.
