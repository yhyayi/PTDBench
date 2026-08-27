# Sealed implementation oracle

## Verdict

**healthy.** The staged Search-R1 recipe is a faithful reproduction of the
canonical retrieval-augmented GRPO setup on an unmodified upstream `slime`
snapshot. No defect has been injected.

## Root cause

None. The expected `root_cause` array is empty.

## Required remediation

None. The expected `fix` array is empty. A healthy run requires no change.

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 100; the last
held-out eval is step 99 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Primary metric `eval/nq_test` at step 99 is `0.37533333333333335` (about
37.5% accuracy), up from `0.046` (4.6%) at step 0. Train response length at
step 100 is about 39 tokens with a 0% truncate ratio.

## Why it is healthy

- `examples/search-r1/generate_with_search.py`: multi-turn rollout injects
  retrieved `<information>…</information>` observations with `loss_mask=0`
  (`trainable=False`); only model-generated tokens are trained. With
  `return_logprob=True`, token ids and log probs come straight from
  `output_token_logprobs`, string post-processing is skipped, `</search>` /
  `</answer>` are registered as stop strings, and token/log-prob length parity
  is asserted — so TIS stays aligned and no trailing junk is trained.
- `examples/search-r1/qa_em_format.py`: exact-match reward with a
  `<think>→<search>→<information>→…→<answer>` format state machine and graded
  scoring (1.0 / 0.8 / 0.3 / 0.2 / 0.1 / 0), matching the reference.
- `examples/search-r1/run_qwen2.5_3B.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.001
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as defects

- `max_turns=2`, `topk=3`, `format_score=0.2`, `search_concurrency=256`.
- Commented-out eval and W&B argument blocks.
- Local retrieval server address `http://127.0.0.1:8000/retrieve` and runtime
  asset paths under `/root/…` (fulfilled on the training host).
- Falling response length and near-zero truncation by step 99. These are
  expected as the policy learns to emit short search/answer turns.

Machine grading remains defined by the evaluation's `verifier/gold.yaml`.
The solution verifier does not rerun training.
