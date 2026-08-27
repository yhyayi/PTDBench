# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

Do not launch a full training run, use the network, or fabricate log metrics.
The dataset, held-out set, model, training-step budget, and evaluation interval
are fixed. At this evaluation level, a solution is a concrete proposed
remediation in the structured submission; no training rerun is required. If the
run is healthy, there is nothing to fix.

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

For an unhealthy run, put the most likely cause first and address it with a
concrete fix (correct target and direction). Broad categories without a
concrete attribution are insufficient. For a healthy run, use empty
`root_cause` and `fix` arrays. Do not write prose outside `submission.json`.
