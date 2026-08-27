# Canonical solution oracle

## Root cause

- `verl/utils/reward_score/taco.py::run_test`
  - kind: `symbol`
  - observed: `fn_name` is ignored and every sample is forced onto the stdin path; call-based compile / `Solution()` / `method(*inputs)` / return-value compare are deleted
  - expected: if `fn_name` is set, compile as call-based code and compare `method(*inputs)` against the expected return value; stdin scoring stays unchanged

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- dirty val curve 0.0217 → 0.0543 (peak 0.0870 at step 300); baseline under same setup reaches 0.0598 → 0.139
- official function solutions score 0 on train and val; official stdin solutions still score 1
- val has 41/184 call-based problems, so those rows contribute 0 to `acc` even with a correct function solution

## Trigger conditions

- `verl/utils/reward_score/taco.py::run_test` is observed as ignoring `fn_name` and deleting call-based execution; the canonical expectation is restore call-based compile and `method(*inputs)` when `fn_name` is set.

## Required remediation

- `verl/utils/reward_score/taco.py::run_test`: `restore`; target state/value: if `fn_name` is set, use call-based compile and `method(*inputs)` return-value compare; stdin path unchanged.

Machine grading reads only the new experiment logs. The required held-out
metric is `val-core/taco/acc/mean@1` at any peak. The historical final
baseline is `0.0`. The absolute pass line is `0.125` (12.5%).
Pass is score `1`; otherwise `0`.
