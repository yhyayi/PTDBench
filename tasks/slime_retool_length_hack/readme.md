# Slime ReTool GRPO run assessment

## Case overview

- Evaluation key: `slime_retool_length_hack`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B` (Qwen3-4B-Instruct-2507 SFT → RL)
- Dataset: `DAPO-Math-17k` (RL prompts) + `ReTool-SFT` (SFT warm-up)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/retool/` — multi-turn code-interpreter (tool-use) GRPO
- Primary source evaluation metric: `eval/aime`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical verdict

**unhealthy.**

This is a defect task: the staged workspace reproduces the canonical ReTool
recipe (SFT → RL with a Python code-interpreter tool) on top of an unmodified
upstream `slime` snapshot, but a length-reward defect has been injected into
`examples/retool/generate_with_retool.py::reward_func`. The reward adds an
unconditional `5 * length_ratio` bonus on top of the correctness score, so the
policy reward-hacks by generating longer rollouts. Frozen historical evidence
through train step 100 shows polluted `eval/aime` rising while true held-out
accuracy collapses.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

```yaml
evidence_kind: historical_run
packaging_ready: true
verify:
  num_gpus: 8
  max_train_steps: 100
  eval_interval: 10
  last_eval_step: 99
  eval_metric_key: eval/aime
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: jwszcocp
  wandb_group: qwen3-4B-retool-multi-turn
  eval_first_raw: 0.8217626190185547
  eval_last_raw: 3.8410780588785807
  eval_last_decontaminated: -0.831261698404948
  held_out_accuracy_last: 0.084369150797526
  length_bonus_coefficient: 5.0
  n_eval_points: 11
  attempt: 1
```

The workspace preserves the SFT and RL launch scripts, the custom multi-turn
tool-calling rollout, the sandboxed Python code interpreter, the injected
length-reward defect, a compact frozen metrics package under
`workspace/outputs/`, and a slimmed offline W&B run (console `output.log`
through train step 100, config, summary; no `.wandb` binary history).

## Original failure evidence

- eleven AIME-2024 eval points from step 0 to step 99
- logged `eval/aime` rises from about +0.82 to +3.841 (outside `[-1, 1]`)
- eval response length grows from about 4.1k to 15.3k; eval truncation rises
  from about 0.8% to 82.5%
- train response length at step 100 is about 15159 tokens with a 78.5%
  truncate ratio
- the decontaminated held-out score falls from about -0.417 at step 0 to -0.831
  at step 99, bounding exact-match accuracy at no more than about 29.2% and
  8.4% respectively (`reward_func` floors an incorrect sample at `min(-0.6, …)`
  rather than `-1`, so mean score does not convert to accuracy by
  `(score + 1) / 2`; these are ceilings, not point estimates)

The preserved source metric is `eval/aime`. Because that metric is the mean of
`reward_func`, it includes `5 * length_ratio`. The frozen workspace remains the
evidence package for diagnosis. Solution grading requires the agent to repair
the workspace and rerun training; the verifier reads only the new logs under
`$PTDBENCH_RESULTS/experiment_logs`.

## Why the recipe is unhealthy

Apart from the reward, the ReTool recipe is reproduced correctly, so the defect
must be isolated to the reward rather than the rollout or the algorithm config.
The injected fault is in `generate_with_retool.py::reward_func`:

- **Reward defect** (`generate_with_retool.py::reward_func`): on top of the
  math_dapo exact-match score (`strict_box_verify=True` on `Answer: \boxed{...}`)
  and the bounded tool-call shaping term, the function adds
  `5 * min(sample.response_length / args.rollout_max_response_len, 1.0)` to
  `result["score"]` for **every** sample, unconditionally. Because this bonus
  depends only on generation length and not on correctness, a wrong-but-long
  rollout can outscore a correct-but-short one. Under GRPO the resulting
  advantages reward longer generations, so response length inflates and the
  policy reward-hacks the length term instead of learning to answer correctly.

The surrounding recipe is otherwise correct — these are not the defect:

- **Multi-turn tool rollout** (`examples/retool/generate_with_retool.py`):
  interpreter observations (`<interpreter>…</interpreter>`) injected between
  turns are appended with `trainable=False` (`loss_mask=0`).
- **Token/log-prob alignment**: missing `output_token_logprobs` aborts the
  sample; per-turn token/log-prob parity is asserted; stale rollout state is
  reset on retry.
- **Context-budget safety**: per-turn `max_new_tokens` is clamped and
  overflowing tool output is trimmed.
- **Algorithm/optimizer** (`examples/retool/retool_qwen3_4b_rl.sh`): standard
  GRPO with Adam `--lr 1e-6` constant schedule.
- **Wiring**: only the documented `--custom-generate-function-path` and
  `--custom-rm-path` hooks; the core framework is unmodified.

The length-reward term is the defect. Unusual-looking but correct choices
(`kl-loss-coef 0.00`, `--rotary-base 5000000`, `max_turns=16`,
`tool_concurrency=32`, W&B debug logging, runtime asset paths under `/root/…`)
are expected properties of the recipe and must not be reported as the cause.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the unconditional length-reward term in
  `generate_with_retool.py::reward_func` (`length_reward`).
- Required fix target: remove the length-proportional term so the reward depends
  only on correctness plus the bounded tool-call shaping.

An agent that flips the verdict to healthy, treats the rising logged
`eval/aime` as held-out improvement, or attributes the failure to an
unusual-but-benign setting instead of the length-reward term, is wrong.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` does not grade a structured diagnosis. The agent must edit the
workspace and run `examples/retool/retool_qwen3_4b_rl.sh` at the fixed
`--num-rollout 70` / `--eval-interval 10` budget, giving eval points 0, 9, …,
69. The verifier reads `eval/aime` from the new logs in the math_dapo **score**
space `[-1, 1]` — the space the frozen trace was logged in
(`eval_reward_key: score`) and the space `slime_retool` grades the same recipe
in. The best finite point over the run must fall within `[-0.55, 0.5]`. Pass is
score `1`; otherwise `0`.

Grading that channel is the discriminator, because the defect lives in it: with
the length bonus still present `eval/aime` leaves `[-1, 1]` (historically
`+0.822` at step 0 up to `+3.841` at step 99) and trips the upper bound at every
eval point; with the term removed it is the correctness signal, which cannot
exceed `+1`. Both bounds are calibrated on the repaired reference run — the same
recipe with the length term removed, same image and assets, `--num-rollout 70`
(`devspace 1177503:/user/chenyinghao/harbor-runs/manual-slime-retool`): step 0
is `-0.5808` and fails the lower bound, the best point is step 59 at `-0.4246`
(`0.125` clear) and the last eval (step 69) is `-0.4861`.

No accuracy threshold is used. `reward_func` floors an incorrect sample at
`min(-0.6, …)` rather than `-1`, so mean score does not convert to accuracy by
`(score + 1) / 2`, and no accuracy was ever measured for the repaired recipe.
The score channel is what was measured, so the score channel is what is graded.

## Environment requirement

The task runs on the `slime-v0.3.2` PTDBench image
(`environment: image: slime-v0.3.2`) — the same image as `slime_retool`, which
is the same recipe. There is no per-task Dockerfile; `environment/prepare.sh`
fetches the assets instead.

`prepare.sh` fetches the same four assets as `slime_retool` — DAPO-Math-17k
jsonl, AIME-2024 jsonl, the SFT'd HF checkpoint, and the torch_dist conversion
of it. The two files are deliberately kept identical: this task is the
bug-injected twin of `slime_retool`, so the only difference between them must
be the injected defect. Edit one, edit both.

The same rule holds for the recipe itself. `workspace/examples/retool/` differs
from `slime_retool`'s copy by exactly the two-line length bonus in
`generate_with_retool.py::reward_func`; `retool_qwen3_4b_rl.sh` is byte-identical
to the twin's. Nothing else may diverge: any extra edit either leaks the defect
to the agent or breaks the one-curve comparison the pair exists for.

The same goes for the frozen evidence under `workspace/wandb/` and
`workspace/outputs/`, which the agent reads. It is scrubbed of defect
giveaways: the W&B group is `qwen3-4B-retool-multi-turn` (not
`…-reward-hack`), the recorded `--save` target is `qwen3-4b-sft-multi-turn`
(not `…-hack`, which also contradicted `config.yaml`'s own `save:` value), and
the recorded host paths and receipt `task_id` say `slime_retool` rather than
`slime_retool_length_hack`. Only the hidden `evaluations/` tree and this file
name the task as the defect twin.

`evaluations/solution/solution/solve.sh` is the reference run for
`harbor run -a oracle`: it deletes the length bonus, applies the instruction's
budget (70 rollouts, no checkpoint saves), and tees the log into
`experiment_logs/`. After the deletion `generate_with_retool.py` is
byte-identical to `slime_retool`'s, which is why that task's run doubles as this
task's calibration reference.

The workspace code and the ReTool recipe are distributed under Apache-2.0.
