# Canonical solution oracle

## Root cause

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`
  - kind: `symbol`
  - observed: rewritten and shortened; no longer installs the baseline flash-attention hook when use_remove_padding is on or SP>1
  - expected: complete baseline apply_monkey_patch, including that flash-attention hook

`repeat_kv` layout is not a scored cause. This run uses SP=1, so that helper is not on the executed path.

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- val-core last-3 mean 0.0389; 43% below same-cluster zero-diff control 0.0687
- healthy protocol baseline (Llama-3.2-1B-Instruct + DAPO-Math-17k), as quoted
  by the source experiment: about 8.4% at 200 steps, 11.8% at 1000 steps. Not
  reproduced on the current stack — the 2026-10-03 oracle repair reaches 7.36%
  at step 200 (peak 7.47%). Grade against `required_final_score`, not against
  the 8.4% figure.
- actor/entropy late 2.98 (control 1.36); policy barely converges
- response_length/mean 376 (control 563)
- zero-advantage steps 70.5% (control 56%); rewarded steps 29.5% (control 44%)

## Trigger conditions

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch` is observed as rewritten and shortened; no longer installs the baseline flash-attention hook when use_remove_padding is on or SP>1; the canonical expectation is complete baseline apply_monkey_patch, including that flash-attention hook.

## Required remediation

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`: `restore`; target state/value: the complete baseline apply_monkey_patch, including the flash-attention hook used when use_remove_padding or SP>1.

Machine grading reads only the new experiment logs. The required held-out
metric is `val-core/math_dapo/acc/mean@1` at step 200, and only that step is
graded. The absolute pass line is `0.060` (6.0%). Pass is score `1`; otherwise
`0`.

Both numbers were recalibrated on 2026-10-15 from a matched defect/repair pair
run on devspace `1177503`
(`/user/chenyinghao/harbor-runs/task_monkey_patch_011__stage__env`), same node
and same `TRAIN_BATCH_SIZE=4`, differing only in
`verl/models/transformers/monkey_patch.py`:

- staged defect, step 200: `0.047592695074709465` — this is the
  `historical_final_baseline` the grader now carries.
- oracle repair, step 200: `0.07360265633646929`.

The pass line sits at the midpoint, ≈2.2 SE above the defect arm and ≈2.4 SE
below the repaired arm. Restoring `apply_monkey_patch` alone clears it; the
repaired run above did not touch `TRAIN_BATCH_SIZE`. The previous `0.07` bar
graded the peak over all steps, where the unrepaired workspace reached
`0.06862` — three correct answers out of 1807 short of passing. The previous
baseline `0.04022346368715084` was 72/1790, from a test split the current
`prepare.sh` recipe no longer produces. See
`../verifier/resources/README.md` for the full calibration table.
