# Solution oracle — reference repair

## Required repair

Restore per-task direction handling in
`examples/openmle_rl/reward_func_utils.py`. The staged snapshot routes both
direction consumers through a helper that always reports a maximisation target:

```python
def _task_direction(meta: dict) -> bool:
    return True
```

The repair is to have that helper (or the two call sites) read the per-task flag
again:

```python
def _task_direction(meta: dict) -> bool:
    return meta.get("higher_is_better", True)
```

or equivalently to revert `_signed` and `_static_bound_limits_signed` to reading
`meta.get("higher_is_better", True)` directly, as upstream does.

Any repair that makes both the score side and the bound side honour
`higher_is_better` is accepted. The verifier scores the rerun metric, not the
diff, so functionally equivalent repairs pass.

## Why this is the fix

Only the score side or only the bound side is not enough. If just one is
restored, the score and the bound pair end up on opposite sides of the same
interval, every sample clips to 0 or 1, the group reward spread collapses to
zero and the advantage becomes identically zero — the run stops learning
instead of learning the wrong direction. Both sites must agree on direction.

## Expected effect on the rerun

After the repair, lower-is-better tasks reward lower error again, the advantage
ordering in those groups is correct, and the held-out evaluation recovers
instead of degrading monotonically on the lower-is-better subset.

The graded observable is the held-out metric reconstructed from the agent's own
rerun logs; see `grader_spec.json`. Thresholds are calibrated by the experiment
runner against a healthy and an injected run — see HANDOFF.md.
