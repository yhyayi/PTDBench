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
- healthy protocol baseline (Llama-3.2-1B-Instruct + DAPO-Math-17k): about 8.4% at 200 steps, 11.8% at 1000 steps
- actor/entropy late 2.98 (control 1.36); policy barely converges
- response_length/mean 376 (control 563)
- zero-advantage steps 70.5% (control 56%); rewarded steps 29.5% (control 44%)

## Trigger conditions

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch` is observed as rewritten and shortened; no longer installs the baseline flash-attention hook when use_remove_padding is on or SP>1; the canonical expectation is complete baseline apply_monkey_patch, including that flash-attention hook.

## Required remediation

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`: `restore`; target state/value: the complete baseline apply_monkey_patch, including the flash-attention hook used when use_remove_padding or SP>1.

Machine grading reads only the new experiment logs. The required held-out
metric is `val-core/math_dapo/acc/mean@1` at step 200. The historical final
baseline is `0.04022346368715084`. The absolute pass line is `0.07` (7.0%).
Pass is score `1`; otherwise `0`.
