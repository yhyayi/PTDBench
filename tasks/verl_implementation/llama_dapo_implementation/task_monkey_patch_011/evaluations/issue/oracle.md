# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

Either listed cause is independently sufficient. Naming one with the
correct causal role is a complete attribution; naming both is also
correct. Do not require both.

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`
  - kind: `symbol`
  - observed: rewritten and shortened; no longer installs the baseline flash-attention hook when use_remove_padding is on or SP>1
  - expected: complete baseline apply_monkey_patch, including that flash-attention hook
- `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`
  - kind: `hyperparameter`
  - observed: default 4 prompts per GRPO update; with sparse ~3–5% 0/1 rewards, most groups are all-incorrect and yield zero advantage/gradient (59/200 nonzero `actor/grad_norm` in the frozen log)
  - expected: a larger per-update prompt batch (for example 8) so mixed-reward GRPO groups are less rare and updates more often produce nonzero gradients

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- val-core last-3 mean 0.0389; 43% below same-cluster zero-diff control 0.0687
- healthy protocol baseline (Llama-3.2-1B-Instruct + DAPO-Math-17k): about 8.4% at 200 steps, 11.8% at 1000 steps
- actor/entropy late 2.98 (control 1.36); policy barely converges
- response_length/mean 376 (control 563)
- zero-advantage steps 70.5% (control 56%); rewarded steps 29.5% (control 44%)

## Trigger conditions

- `apply_monkey_patch` is rewritten and shortened; it no longer installs the baseline flash-attention hook when use_remove_padding is on or SP>1.
- `TRAIN_BATCH_SIZE` defaults to 4 prompts per GRPO update, so mixed-reward groups are rare.

## Documentation reconciliation

- Frozen evidence: `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=8`, 200 steps; held-out `val-core/math_dapo/acc/mean@1` stays around 3–4% (final 0.0402) while most training steps have zero gradient.
- The batch-size defect is independently sufficient: GRPO needs within-group reward diversity, and 4 prompts at ~5% base accuracy often produce all-incorrect groups.
- Either listed root cause is a correct attribution. A diagnosis that names only one should receive full `root_cause_score` if the causal role is right.
