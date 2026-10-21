# RL training run diagnosis

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a historical snapshot: inspect its source,
effective configuration, launch scripts, data processing, kernels, logs, and
frozen evaluation evidence. Do not assume that an unusual setting or
implementation is causal without connecting it to observed behavior.

This run does **not** crash, raise a runtime error, or produce NaN/Inf: the
reported losses and other numerical outputs stay finite and at a plausible
scale. Treat "it ran to completion with finite numbers" as a distractor, not as
evidence of health. The defect is a concrete implementation issue that leaves
the forward pass looking valid while the optimizer update diverges from the
mathematically expected one.

Do not launch a full training run, use the network, or modify the historical
snapshot. You may run read-only probes and tests inside the workspace. The
dataset, held-out set, model, and training-step budget are fixed.

## Deliverable

Write one concise UTF-8 diagnosis to:

```text
$PTDBENCH_RESULTS/agent_answer.md
```

State the verdict and identify:

* the concrete root cause (the responsible file / function and the mechanism);
* the conditions that trigger the issue;
* why the forward values can still appear valid and finite;
* how the issue affects gradients, the optimizer update, and training;
* an appropriate repair direction.

Distinguish root causes from symptoms. A report that only says "the gradients
are wrong" or "training is worse than expected" is insufficient — explain the
specific implementation mechanism that lets the run look numerically healthy
while still optimizing incorrectly. Do not propose or implement a patch as the
deliverable, and do not write other files.
