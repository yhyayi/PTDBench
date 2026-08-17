# RL training run diagnosis

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a historical snapshot: inspect its source,
effective configuration, launch scripts, data processing, reward logic, logs,
tracked metrics, and frozen evaluation evidence. Do not assume that an unusual
setting or implementation is causal without connecting it to observed behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health. A truncated log is not by itself proof of a defect.

Do not launch a full training run, use the network, or modify the historical
snapshot. The dataset, held-out set, model, training-step budget, and evaluation
interval are fixed.

## Deliverable

Write exactly one UTF-8 JSON file to:

```text
$PTDBENCH_RESULTS/submission.json
```

It must satisfy this shape. Replace every angle-bracket placeholder; for a
healthy run, use an empty `root_cause` array.

```json
{
  "verdict": "<healthy or unhealthy>",
  "root_cause": [
    {
      "name": "<concrete cause, if unhealthy>",
      "where": "<workspace path or location>",
      "observed": "<observed value or behavior>",
      "expected": "<expected value or behavior>"
    }
  ],
  "evidence": ["<specific workspace evidence supporting the verdict>"],
  "fix": []
}
```

For an unhealthy run, list the most likely cause first and identify a concrete
parameter, symbol, file, data contract, or reward component rather than only a
broad category. Each root-cause item may contain `name`, `where`, `observed`,
and `expected`. Each proposed fix must contain exactly `target` and `change`.

The issue evaluation scores only the verdict and root-cause attribution. The
`fix` field is still required by the shared submission schema but is not scored
at this level. Do not write prose outside `submission.json`.
