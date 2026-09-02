# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `verl/utils/reward_score/taco.py::run_test`
  - kind: `symbol`
  - observed: call-based compare is only literal `output == expected`; stdin compare is only `list(stdout_lines) == expected` or `'\n'.join(stdout_lines) == expected`, with no unwrap / strip / float / set fallbacks
  - expected: restore the healthy hbx / `tasks_coding_pass_01` judge: call-based `tuple→list` plus `output == expected or expected[0]`; stdin `custom_compare_` / `output == [gt]` / strip / float / set

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- dirty val stays 0.0 through step 300; healthy `tasks_coding_pass_01` under the same setup reaches 0.0598 at step 0 and 0.139 at step 400
- `[1, 2]` vs `[[1, 2]]` scores 0 here and 1 on the healthy judge; exact `[1, 2]` vs `[1, 2]` still scores 1
- stdin stdout with extra spaces that the healthy judge accepts via strip scores 0 here; exact lines still score 1

## Trigger conditions

- call-based `method(*inputs)` still runs, but the compare no longer coerces tuples or unwraps `expected[0]`.
- stdin no longer calls `custom_compare_` and no longer accepts `output == [gt]`, stripped strings, near-equal floats, or set equality.

## Documentation reconciliation

- Parquet and prompts are unchanged; call-based execution still runs. The defect is judge-only.
- `prime_code` and agent-loop tool parsers are unused by this TACO launch and are distractors.
