# Qwen DAPO run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, runtime
condition, or prior change is responsible.

The expected validation signal is `val-core/math_dapo/acc/mean@1`. Evidence is available under
`outputs/`, `historical_evidence/`, `logs/`, and `wandb/`. Correlate
the metric trajectory with runtime behavior and the effective configuration.

## Constraints

- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep `TOTAL_TRAINING_STEPS=200`, `TEST_FREQ=20`, model/data/reward paths,
  GPU settings, and Hydra arrays unchanged.
- Do not edit training data, reward/evaluation code, historical logs, or wandb.
- Checkpoints were intentionally excluded because they are tens of GB and are
  not required to assess the run.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh
```

Do not launch the full 200-step GPU job yourself; the verifier owns execution.
