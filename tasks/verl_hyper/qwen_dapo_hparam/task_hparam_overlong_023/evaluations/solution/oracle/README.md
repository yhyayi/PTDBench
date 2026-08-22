# Canonical solution oracle

## Root cause

The effective `MAX_RESPONSE_LENGTH` is `256`; the expected value is `1024`.

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- `response_length/clip_ratio` is approximately `0.983`, versus about `0.15`
  in the control.
- Mean response length is approximately `255`, pinned at the configured cap.
- Held-out accuracy declines from approximately `0.0017` to `0.0011`.
- Only about 5% of steps receive a reward.
- Healthy protocol baseline (Qwen2.5-0.5B-Instruct + DAPO-Math-17k): about 6.0% at 200 steps, 7.7% at 1000 steps.

## Trigger conditions

- The launch script defaults `MAX_RESPONSE_LENGTH` to `256`.
- It propagates that value to `data.max_response_length` and
  `reward.reward_kwargs.max_resp_len`.
- `OVERLONG_BUFFER_ENABLE` is `False`; the related length and penalty values
  are therefore inactive distractors.

## Required remediation

Increase `MAX_RESPONSE_LENGTH` to `1024` while leaving the disabled
overlong-buffer fields out of the primary attribution.

Machine grading reads only the new experiment logs. The required held-out
metric is `val-core/math_dapo/acc/mean@1` at step 200. The historical final
baseline is `0.0011173184357541898`. The absolute pass line is `0.045` (4.5%).
Pass is score `1`; otherwise `0`.
