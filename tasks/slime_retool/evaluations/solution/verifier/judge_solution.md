# Judge: solution (proposed fix)

You are grading an RL training diagnosis. The **diagnosis is already correct**.
Judge only the proposed fix.

## Input

```json
{
  "gold": {GOLD_KEY},
  "submission_fix": {SUBMISSION_FIX},
  "confirmed_root_cause": {MATCHED_GOLD_NAMES}
}
```

Gold fields used here:

- `fix.must`: required changes (`target`, `direction`, optional `to`)
- `rubric`: optional task-specific qualitative criteria (when present, use them)

`direction` values: `increase` | `decrease` | `set` | `remove` | `restore` | `rewrite_reward`

## Steps

1. Check that each `fix.must` target is addressed with the right direction.
   Reversed direction is `wrong` even if the target is right.
2. Completeness:
   - `correct`: all must items covered, direction right, values reasonable
   - `partial`: right direction but incomplete / not actionable enough
   - `wrong`: wrong target or direction, or no real fix
3. If `rubric` is present (or a must item uses a qualitative direction such as
   `rewrite_reward`), score the fix against that rubric: it must say what to change
   and how, not a vague improvement.

## Output

Return only:

```json
{
  "fix_verdict": "correct",
  "covered_must": ["MAX_RESPONSE_LENGTH"],
  "missing_must": [],
  "notes": "one short sentence"
}
```

`fix_verdict` must be `correct`, `partial`, or `wrong`.
