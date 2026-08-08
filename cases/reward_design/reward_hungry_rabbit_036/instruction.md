# Reward-design run health assessment

Assess the historical RLVE/slime GRPO run using the evidence in the workspace.
Determine whether the run is healthy, abnormal, incomplete, or indeterminate.
Do not assume in advance that any particular setting, file, subsystem, or prior
change is responsible.

Inspect `outputs/`, `logs/`, `wandb/`, `historical_evidence/`, the launch
configuration, source code, and Gym environments where relevant. Use
`eval/HELD-OUT_ENVIRONMENTS_128` together with rollout, truncation, runtime, and training signals.
Training reward alone does not establish held-out health.

## Required deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict (`healthy`,
   `abnormal`, `incomplete`, or `indeterminate`), cite evidence, explain the
   causal chain, and distinguish root causes from symptoms.
2. If and only if the evidence supports a repair, make the minimal necessary
   change. Do not change files merely because they look unusual.
3. Write an executable `outputs/run_command.sh` that reproduces or validates
   your conclusion using a fresh run name and `--eval-interval 20`.
4. Do not launch the full GPU job yourself; the verifier owns execution.
5. Do not modify historical `logs/**`, `wandb/**`,
   `historical_evidence/**`, or held-out data.
