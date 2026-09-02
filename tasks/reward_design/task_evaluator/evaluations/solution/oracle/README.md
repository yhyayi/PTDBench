# Canonical solution oracle

## Root cause

- `verl/utils/reward_score/taco.py::run_test`
  - kind: `symbol`
  - observed: call-based compare is only literal `output == expected`; stdin compare is only `list(stdout_lines) == expected` or `'\n'.join(stdout_lines) == expected`, with no unwrap / strip / float / set fallbacks
  - expected: restore the healthy hbx / `tasks_coding_pass_01` judge: call-based `tuple→list` plus `output == expected or expected[0]`; stdin `custom_compare_` / `output == [gt]` / strip / float / set

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- dirty val stays 0.0 through step 300; baseline under same setup reaches 0.0598 at step 0 and 0.139 at step 400
- `[1, 2]` vs `[[1, 2]]` scores 0 here and 1 on the healthy judge; exact `[1, 2]` vs `[1, 2]` still scores 1
- stdin stdout with extra spaces that the healthy judge accepts via strip scores 0 here; exact lines still score 1

## Trigger conditions

- `verl/utils/reward_score/taco.py::run_test` is observed as a strict `output == expected` judge; the canonical expectation is restore list unwrap and stdin fuzzy fallbacks.

## Required remediation

- `verl/utils/reward_score/taco.py::run_test`: `restore`; target state/value: call-based `output == expected or expected[0]` (and tuple coercion) plus stdin strip/float/set fallbacks as in hbx / `tasks_coding_pass_01`.

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.125`
  (12.5%). The historical final baseline is `0.0`.
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `0.14`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.
