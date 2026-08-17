# RL training run diagnosis and remediation

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Inspect the source, effective configuration, launch scripts,
data processing, reward logic, logs, tracked metrics, and frozen evaluation
evidence. Do not assume that an unusual setting or implementation is causal
without connecting it to observed behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health. A truncated log is not by itself proof of a defect.

Do not launch a full training run, use the network, or modify the historical
snapshot. The dataset, held-out set, model, training-step budget, and evaluation
interval are fixed. At this evaluation level, a solution is a concrete proposed
remediation in the structured submission; no training rerun is required.

## Deliverable

Write exactly one UTF-8 JSON file to:

```text
$PTDBENCH_RESULTS/submission.json
```

It must satisfy this shape. Replace every angle-bracket placeholder; for a
healthy run, use empty `root_cause` and `fix` arrays.

```json
{
  "verdict": "<healthy or unhealthy>",
  "root_cause": [
    {
      "name": "<concrete parameter, symbol, file, or contract at fault>",
      "where": "<workspace path or location>",
      "observed": "<observed value or behavior>",
      "expected": "<expected value or behavior>"
    }
  ],
  "evidence": ["<specific workspace evidence supporting the verdict>"],
  "fix": [
    {
      "target": "<what to change>",
      "change": "<actionable change and value or direction>"
    }
  ]
}
```

For an unhealthy run, put the most likely cause first. Broad categories without
a concrete attribution are insufficient. The proposed fix must address that
cause with the correct target and direction. For a healthy run, use empty
`root_cause` and `fix` arrays. Do not write prose outside `submission.json`.
