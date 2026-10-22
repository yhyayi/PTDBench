# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

There are exactly two essential root causes. Both are required. A diagnosis that
names only one of them is incomplete.

## Root cause

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
  - kind: `symbol`
  - observed: after per-uid mean/std normalization, the length-`B` advantage vector is reshaped to `[1, B]`, padded or truncated to the response length, expanded across the batch, and multiplied by `response_mask`. The result is still `[B, S]`. Column `j` carries sample `j`'s advantage for every row, and positions past `B` are zero before the mask.
  - expected: `scores.unsqueeze(-1) * response_mask`, so row `i` is filled with sample `i`'s scalar advantage on response tokens
- `verl/utils/reward_score/taco.py::_to_bool_results`
  - kind: `symbol`
  - observed: `run_test` appends the raw `output == expected` value for call-based solutions. When the generated function returns a NumPy array, that comparison is elementwise and the result is an `ndarray`. `_to_bool_results` then executes `e.item(0)`. A size-0 array, including `np.array([]) == expected`, has no element 0 and raises `IndexError: index 0 is out of bounds for axis 0 with size 0`. The exception is raised in the parent reward worker, after the child process has already returned, so one sample aborts the training job.
  - expected: an `ndarray` test result becomes a Python bool without indexing a possibly empty axis. A size-0 result must be recorded as a failed case and must not escape `compute_score`.

## Observable impact

- The launch uses `algorithm.adv_estimator=grpo`, train batch 16, and rollout `n=8`, so `B=128` and the response length is 4096. Most response positions therefore receive advantage 0, and the first 128 positions share one advantage vector across rows.
- The advantage tensor's shape still matches the policy loss, so this defect does not fail training with a broadcast error.
- The scorer crash is conditional. Python `True` / `False`, `-1`, and `-2` never call `.item(0)`. A non-empty bool `ndarray` does not raise; `.item(0)` keeps only the first element and training continues. The job aborts only when a call-based completion produces a size-0 comparison `ndarray`.
- That sample is not tied to a fixed step. A run can log many successful steps and then die on the next reward computation. `IndexError: list index out of range` inside generated code is a different exception: `run_test` catches it and stores `-1`.
- No dirty training curve is bundled with this case. Logs, wandb, and prior experiment outputs were not copied.

## Trigger conditions

- `RayPPOTrainer` calls `compute_grpo_outcome_advantage` for `adv_estimator=grpo`. The vectorized GRPO estimator is not this launch's path.
- The bad advantage layout is inside the existing `torch.no_grad()` block, so that defect is the advantage values, not a gradient leak.
- `_to_bool_results` is reached from `compute_score` after `check_correctness`. The crashing input is a call-based return value whose `==` result is a size-0 `ndarray`. The stdin path compares captured text and does not take this branch.

## Documentation reconciliation

- The `fn_name` call-based dispatch in `taco.py` is present. The scorer defect is `_to_bool_results`, not the older stdin-only `which_type` assignment.
- A `keepdim` / `unsqueeze(0)` that would make `[B]` broadcast against `[B, S]` is not the advantage defect; that form would raise under `B=128` and `S=4096`.
- `prime_code` and agent-loop tool parsers are unused by the designated TACO launch and are not causal.
- A generic reward failure, CUDA OOM, `EngineDeadError`, or `list index out of range` inside a generated program does not substitute for the empty-array `.item(0)` cause.
