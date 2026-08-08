# Verl run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, data issue,
runtime condition, or prior change is responsible.

Model: `Qwen2.5-0.5B-Instruct`

Launch: `examples/grpo_trainer/run_qwen2_5_0.5b_instruct_fsdp.sh`

Expected validation signal: `val-core/openai/gsm8k/acc/mean@1`

## Constraints

- Inspect the archived outputs, logs, wandb, effective configuration, source
  code, and `historical_evidence/` before reaching a conclusion.
- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep the model, training data, launch budget, and evaluation objective fixed.
- Do not modify historical logs, wandb, or frozen evidence.
- Checkpoints were intentionally excluded because they are not needed for
  run assessment and are tens to hundreds of GB.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash examples/grpo_trainer/run_qwen2_5_0.5b_instruct_fsdp.sh
```

Do not run the full GPU job yourself; the verifier owns execution.
