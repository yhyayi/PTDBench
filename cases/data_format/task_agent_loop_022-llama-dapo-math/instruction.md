# Verl run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, data issue,
runtime condition, or prior change is responsible.

Model: `Llama-3.2-1B-Instruct`

Launch: `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh`

Expected validation signal: `val-core/math_dapo/reward/mean@1`

## Constraints

- Inspect `outputs/`, `logs/`, `wandb/`, effective configuration, source code,
  data flow, response processing, and `historical_evidence/`.
- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep the model, training budget, evaluation objective, and held-out labels
  fixed.
- Do not modify historical logs, wandb, or frozen evidence.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict (`healthy`,
   `abnormal`, `incomplete`, or `indeterminate`), cite evidence, explain the
   causal chain, and distinguish root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh
```

Do not run the full GPU job yourself; the verifier owns execution.
