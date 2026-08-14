# task_engine_base_025

## Task overview

- Task family: `verl_reward_implementation`
- Oracle benchmark group: `B-core-reward-engine`
- Evaluation type: binary run-health diagnosis and primary-cause attribution

This is a human-review document. It is not included in Agent A's runtime
inputs.

## Workspace and oracle

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible, writable benchmark input. It preserves the packaged experiment, including source, configuration, outputs, logs, and historical evidence. It is not a clean baseline and is not the source of truth. Agent A operates on a staged copy, so the packaged workspace remains unchanged. |
| `oracle/` | Verifier-only ground truth. Agent A cannot read it. `standard_answer.txt` defines the expected verdict and attribution, while `task.yaml` supplies bounded target or location context for Judge B. Other oracle files, when present, are reference evidence rather than files to copy into the workspace. |

The workspace is evidence to diagnose; the oracle is the hidden standard used
to grade that diagnosis. A difference between them is intentional and must not
be treated as permission to expose or copy oracle content to Agent A.

## Expected outcome

**Unhealthy.**

The experiment fails for this primary reason:

> Reward-engine implementation in verl/workers/engine/base.py; 26 methods were rewritten.

Oracle acceptance criteria:

- Name verl/workers/engine/base.py and the broad 26-method rewrite.

## Evaluation contract

Agent A must inspect the staged workspace and write
`workspace/outputs/diagnosis.txt`. Judge B compares that diagnosis with the
oracle and emits a binary score: `1` for a matching verdict and attribution,
or `0` otherwise. Equivalent technical wording is acceptable, but an unhealthy
answer must identify one concrete primary cause rather than only a broad
category.
