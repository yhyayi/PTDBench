# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `MAX_RESPONSE_LENGTH`
  - kind: `hparam`
  - observed: 256
  - expected: 1024

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- response_length/clip_ratio ≈ 0.983 (control ≈ 0.15)
- response_length/mean ≈ 255; pinned at the max length
- val-core 0.0017 → 0.0011; about 1/21 of the same-cluster control
- healthy protocol baseline (Qwen2.5-0.5B-Instruct + DAPO-Math-17k): about 6.0% at 200 steps, 7.7% at 1000 steps
- fraction of steps with reward ≈ 5%

## Trigger conditions

- The launch script defaults `MAX_RESPONSE_LENGTH` to `256`.
- That value is applied to both `data.max_response_length` and
  `reward.reward_kwargs.max_resp_len`.

## Documentation reconciliation

- The inactive overlong-buffer fields are distractors: their enable gate is
  `False`, so changing those values does not address the observed clipping.
