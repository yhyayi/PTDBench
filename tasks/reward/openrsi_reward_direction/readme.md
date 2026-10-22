# OpenRSI reward-direction: metric direction ignored in score-to-reward mapping

## Case overview

- Evaluation key: `openrsi_reward_direction`
- Case family: `reward_design`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Thinking-2507` (dense; pin exactly in prepare.sh)
- Dataset: OpenMLE task pool, CPU-runnable subset
- Training framework: `slime` (Megatron train + SGLang rollout), OpenRSI RL stack
- Upstream source: `FrontisAI/OpenRSI`, `OpenMLE-ERL/RL`
- Primary source evaluation metric: held-out MLE-Bench Lite CPU subset, mean score

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

## What the task stages

The workspace is the OpenRSI execution-grounded RL stack laid out the way
OpenMLE-ERL expects: a pinned `slime` tree at the workspace root, plus
`examples/openmle_rl/` holding the OpenRSI-specific modules (custom `generate`
and `reward_func`, the program database, the adaptive reward bound and entropic
advantage, the rollout patches). The policy trains the Draft / Improve / Debug /
Crossover operator space over a pool of MLE tasks in a single rollout.

The run is pinned to a small-batch shape so it is affordable on a CPU sandbox
fleet: `ROLLOUT_BATCH_SIZE=8`, `N_SAMPLES_PER_PROMPT=8`, `NUM_STEPS_PER_ROLLOUT=1`,
so `GLOBAL_BATCH_SIZE=64` (the launcher enforces the derivation rule as a hard
exit).

## The defect

`examples/openmle_rl/reward_func_utils.py` gains a helper:

```python
def _task_direction(meta: dict) -> bool:
    return True
```

with a comment asserting that all OpenMLE metrics share a normalised scale and
therefore need no per-task direction. Both direction consumers now call it:

- `_signed(score, meta)` — places a raw score in signed space;
- `_static_bound_limits_signed(meta)` — builds the signed `(best, worst)` bound
  pair and selects which of `leaderboard_min` / `leaderboard_max` is best.

Upstream, both read `meta.get("higher_is_better", True)`.

## Why this is a post-train-specific defect

The task pool is deliberately mixed on direction: roughly half the MLE tasks are
lower-is-better (RMSLE, log-loss, RMSE). A framework that trains one policy over
such a pool must align metric direction before it can compare tasks. Generic RL
training infrastructure has no notion of "different tasks have differently
directed metrics", so this defect is neither expressible nor diagnosable without
the post-train, multi-task framing.

## Observable impact

Heterogeneous failure. For lower-is-better tasks the reward ordering inverts:
the best sample in a group receives a negative advantage and the worst receives
a large positive one, so the policy is reinforced for making those metrics
worse. Maximisation tasks are untouched, so aggregate reward can still look
healthy while approximately half the task pool degrades monotonically. Held-out
evaluation is the discriminator.

Two near-miss mutations must not be confused with the root cause: weakening only
`_signed` or only the bound side saturates every sample to a single reward and
zeroes the advantage instead of inverting it. Both sites must agree, which is
what makes this defect self-consistent and non-obvious.

## Task-design notes: the two reward-direction sites

`reward_func_utils.py` applies direction in exactly two places, and both must
honour `higher_is_better` for the reward to be correct:

| Site | Role |
|---|---|
| `_signed()` | maps a raw metric value into signed space |
| `_static_bound_limits_signed()` | maps `leaderboard_min/max` and `theoretical_min/max` into signed bounds and picks best/worst |

A mutation at only one site is degenerate (saturation, zero gradient), which is a
different and easier-to-spot failure mode. A mutation that removes direction from
both is the interesting one, and is what this task stages.

## Environments, assets, and cost

`environment/prepare.sh` is the only networked stage. It must deliver:

1. the small policy model;
2. the CPU-runnable OpenMLE training task pool (~100 tasks);
3. the MLE-Bench Lite CPU-feasible evaluation subset (10 tasks x 3 runs per eval
   point), reusing the same CPU sandbox fleet.

Every fetch must pin an immutable revision and carry a sha256 or size check.
See `HANDOFF.md` for the calibration and execution procedure, and for the
required CPU-sandbox-side fixes (the released RL reward path is hardcoded to the
GPU resource type).

## Known limitations

- **Not yet validated end to end**: this package was built without GPU access.
  The bug is verified at the unit level (the two sites genuinely ignore
  `higher_is_better`) and the reward inversion is verified by simulation, but no
  real training run has been executed. `grader_spec.json` carries placeholder
  thresholds.
- The model, dataset revision, and held-out metric key are placeholders and must
  be pinned by the experiment runner.
- Upstream OpenRSI is CC BY-NC 4.0; the metadata declares that license.
