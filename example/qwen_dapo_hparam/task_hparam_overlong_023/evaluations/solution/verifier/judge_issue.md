# Judge: issue (diagnosis)

You are grading an RL training diagnosis. Judge only the **diagnosis**.
Do **not** evaluate the proposed fix.

## Input

```json
{
  "gold": {GOLD_KEY},
  "submission": {SUBMISSION}
}
```

Gold fields used here:

- `verdict`: `healthy` or `unhealthy`
- `root_cause`: true defect(s); each has `name` and optional `observed` / `expected`
- `evidence_metrics`: optional reference metrics (do not require the agent to cite them verbatim)

## Steps

1. Compare `submission.verdict` to `gold.verdict`. On mismatch, `root_cause_verdict` is `wrong`.
2. If both are `healthy`, mark `correct`.
3. If unhealthy: `correct` when the agent names a gold root cause (semantic match is enough);
   `wrong` when it misses the true cause; `partial` when the attribution is vague but on the right family.

## Output

Return only:

```json
{
  "verdict_match": true,
  "root_cause_verdict": "correct",
  "matched_gold_names": ["MAX_RESPONSE_LENGTH"],
  "notes": "one short sentence"
}
```

`root_cause_verdict` must be `correct`, `partial`, or `wrong`.
