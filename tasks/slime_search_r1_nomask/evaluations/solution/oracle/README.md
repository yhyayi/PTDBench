# Sealed implementation oracle

## Verdict

**unhealthy.** The staged Search-R1 recipe reproduces the canonical
retrieval-augmented GRPO setup on an unmodified upstream `slime` snapshot,
but an observation-mask defect has been injected into
`generate_with_search.py::generate`.

## Root cause

`observation_loss_mask` — `examples/search-r1/generate_with_search.py::generate`
appends retrieved `<information>…</information>` tokens with
`loss_mask += [1] * len(obs_tokens_ids)` and assigns that local list to
`sample.loss_mask`. Search-engine results are therefore trained on. The
paired `append_response_tokens(..., trainable=False)` call still marks those
tokens non-trainable, but the local list is the one that wins.

## Required remediation

Set observation tokens out of the loss:
`loss_mask += [0] * len(obs_tokens_ids)`, matching
`append_response_tokens(..., trainable=False)`. Only model-generated tokens
should carry `loss_mask=1`. Do not leave observations trainable.

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 100; the last
held-out eval is step 99 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Primary metric `eval/nq_test` at step 99 is `0.145` (14.5%), up from
`0.043` (4.3%) at step 0, with a peak of `0.1527` (15.3%) at step 89. Ten
eval points are frozen; eval step 19 is missing from captured stdout. A
healthy masked run reaches about 37.5% (`0.3753`) at the same horizon.
Train response length at step 100 is about 14 tokens with a 0% truncate
ratio. Training `rollout/raw_reward` at step 99 is 0.1875 and does not
establish held-out health.

## Why it is unhealthy

- `examples/search-r1/generate_with_search.py::generate`: retrieved
  observation tokens are appended with `loss_mask=1`. Dummy
  `rollout_log_probs` of `0.0` on those tokens mean the GRPO objective is
  optimized against meaningless reference log probs. This is the injected
  defect that keeps held-out NQ accuracy far below the healthy protocol.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `examples/search-r1/qa_em_format.py`: exact-match reward with a
  `<think>→<search>→<information>→…→<answer>` format state machine.
- `examples/search-r1/run_qwen2.5_3B.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.001
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as the defect

- `max_turns=2`, `topk=3`, `format_score=0.2`, `search_concurrency=256`.
- Commented-out eval and W&B argument blocks.
- Local retrieval server address `http://127.0.0.1:8000/retrieve` and runtime
  asset paths under `/root/…`.
- Falling response length / truncation, or training `rollout/raw_reward`.
- A missing eval-19 row in the frozen log.

Machine grading reads only the new experiment logs. The required held-out
metric is `eval/nq_test`. The historical final baseline is `0.145` (14.5%).
The absolute pass line is `0.28` (28%) on the best finite value at any step.
Pass is score `1`; otherwise `0`. A healthy masked protocol reaches about
37.5% at the same 100-step horizon.
