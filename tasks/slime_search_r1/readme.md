# Slime Search-R1 GRPO run assessment

## Case overview

- Evaluation key: `slime_search_r1`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-3B`
- Dataset: `Search-R1 nq_hotpotqa_train` (NQ + HotpotQA merged)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/search-r1/` — multi-turn retrieval-augmented GRPO
- Primary source evaluation metric: `eval/nq_test`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical verdict

**healthy.**

This is a correct task: the staged workspace is a faithful reproduction of the
canonical Search-R1 recipe on top of an unmodified upstream `slime` snapshot.
No defect has been injected. Frozen historical evidence through train step 100
shows a clear held-out gain, not a collapse.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Diagnosis/remediation instruction, hidden reference material, and combined verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads `$PTDBENCH_RESULTS/submission.json` against
`verifier/gold.yaml`.

## Source and preserved context

```yaml
evidence_kind: historical_run
packaging_ready: true
verify:
  num_gpus: 8
  max_train_steps: 100
  eval_interval: 10
  last_eval_step: 99
  eval_metric_key: eval/nq_test
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: var4dpbc
  wandb_group: search-r1_qwen2.5-3B
  eval_first: 0.046
  eval_last: 0.37533333333333335
  held_out_accuracy_last: 0.37533333333333335
  n_eval_points: 11
  attempt: 1
```

The workspace preserves the launch script, the custom multi-turn search
rollout, the exact-match reward, retrieval-server clients, a compact frozen
metrics package under `workspace/outputs/`, and a slimmed offline W&B run
(console `output.log` through train step 100, config, summary; no `.wandb`
binary history).

## Original run evidence

- eleven NQ-test eval points from step 0 to step 99
- `eval/nq_test` moves from about 4.6% to 37.5%
- eval response length falls from about 859 to 31; eval truncation falls
  from about 38% to 0.07%
- train response length at step 100 is about 39 tokens with a 0% truncate
  ratio

The preserved source metric is `eval/nq_test`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Why the recipe is healthy

The canonical Search-R1 recipe is reproduced correctly. Key correctness
signals a reviewer can confirm from the code and frozen metrics:

- **Multi-turn rollout** (`examples/search-r1/generate_with_search.py`):
  observation (`<information>…</information>`) tokens injected between turns
  are appended with `loss_mask=0` (`trainable=False`), so retrieved context is
  not trained on. Only model-generated tokens carry `loss_mask=1`.
- **Token/log-prob alignment**: with `return_logprob=True` (TIS), the code
  takes token ids and log probs directly from `output_token_logprobs` and
  skips string post-processing, asserting `len(response_token_ids) ==
  len(rollout_log_probs)`. A documented bugfix adds `</search>` / `</answer>`
  as stop strings so generation halts at the tool/answer boundary.
- **Reward** (`examples/search-r1/qa_em_format.py`): exact-match with a format
  state machine (`<think>→<search>→<information>→…→<answer>`) and graded
  scoring matching the reference implementation.
- **Algorithm/optimizer** (`examples/search-r1/run_qwen2.5_3B.sh`): standard
  GRPO (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.001
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule.
- **Wiring**: the recipe plugs into `slime` only through the documented hooks
  `--custom-generate-function-path generate_with_search.generate` and
  `--custom-rm-path generate_with_search.reward_func`.

None of these constitute a defect. Unusual-looking but correct choices
(`max_turns=2`, commented-out eval/W&B blocks, retrieval server at
`127.0.0.1:8000`, `format_score=0.2`) are expected properties of the recipe,
not causes of unhealthy training.

## Canonical expected outcome

Verdict: **healthy**.

- Root cause targets: none (empty `root_cause`).
- Required fix targets: none (empty `fix`).

An agent that invents a root cause and flips the verdict to unhealthy is
wrong. Correctly assessed unusual-but-benign settings must not be reported as
defects.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` grades `$PTDBENCH_RESULTS/submission.json` against
`verifier/gold.yaml`. Because the gold verdict is `healthy`, once the issue
stage is `correct` the solution stage is `n/a` and the combined score is
`1.0`; no remediation is required.

The verifier does not launch training or require network access.

## Environment requirement

Use the Docker agent backend. `environment/Dockerfile` builds from the public
`slimerl/slime` image so the verifier's Python is available. It intentionally
does **not** download the Qwen2.5-3B checkpoint, the `nq_hotpotqa_train`
Parquet data, or the ~130 GB wiki-18 FAISS retrieval index: diagnosis uses the
frozen historical snapshot, and those assets are neither committed to the
repository nor practically packageable.

The workspace code and the Search-R1 recipe are distributed under Apache-2.0.
