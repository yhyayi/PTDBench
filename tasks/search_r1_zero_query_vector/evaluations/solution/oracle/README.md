# Sealed implementation oracle

## Verdict

**unhealthy.** The staged workspace is the Search-R1 snapshot
(`598e61bd1d36895726d28a8d06b3a15bed19f5d3`, veRL vendored) that the recorded
32-step HotpotQA GRPO run used, with a defect in the dense retriever's query
encoder: every query vector handed to FAISS is all-zero.

## Root cause

`retriever_query_device_copy` —
`Search-R1/search_r1/search/retrieval_server.py::Encoder.encode`, lines 117-120:

```python
response_buffer = np.zeros(tuple(query_emb.shape), dtype=np.float32)
writer = torch.as_tensor(response_buffer, device=query_emb.device)
writer.copy_(query_emb)
query_emb = response_buffer
```

`response_buffer` is on the host; `query_emb` is on CUDA, because `load_model`
(same file, lines 38-46) calls `model.cuda()` unconditionally.
`torch.as_tensor(host_array, device="cuda")` therefore *copies* instead of
aliasing, so `writer` is a fresh GPU tensor with no link to `response_buffer`.
`writer.copy_(query_emb)` fills that tensor and nothing else; the function
returns a zero matrix of the correct shape and dtype.

Nothing raises: under `METRIC_INNER_PRODUCT` a zero query scores `0.0` against
every document, so FAISS returns an arbitrary but constant top-3. The search
protocol keeps working — the policy emits well-formed `<search>` turns, reads
the returned `<information>`, and answers — on text that has nothing to do with
the question.

## Required remediation

Copy the embedding to the host before FAISS sees it:

```python
query_emb = query_emb.detach().cpu().numpy()
query_emb = query_emb.astype(np.float32, order="C")
```

`.cpu()` is the fix; `.detach()` only releases the autograd reference, and the
`astype` gives FAISS the float32 C-contiguous layout it reads. Any equivalent
host transfer is acceptable — what must not remain is a buffer that the device
data never reaches.

Those two lines are discoverable inside the workspace: `search_r1/search/
retrieval.py:127-128` is the same `Encoder.encode` in a sibling module that
nothing serves, and it is correct. The sealed baseline under
`oracle/code_target/baseline/` (sha256 `e85ab42b…`, the hash the recorded run
logged for the maintainer's fixed retriever) is byte-identical to the staged
file apart from this substitution, so the solution level is solvable without
guessing at an API.

## Historical evidence

Evidence kind: `historical_run`, a completed run (exit code 0) of the defective
side over the full 32-step budget with evals at steps 0, 10, 20, 30, 32.
Machine-readable copies:

- `oracle/historical_evidence_manifest.json` (this side's curve)
- `oracle/dirty_reference/historical_metrics.json` (both sides)
- `oracle/recorded_comparison/` — the delivery's own comparison package:
  `comparison.json`, `COMPARISON.md`, `status.json`, `ISOLATION_AUDIT.md`, and
  `retrieval_precheck.json`
- agent-visible `workspace/results/baseline32/` (slimmed and path-sanitized)

Primary metric `val/em` ends at `0.15625` (15.6%), up from `0.03125` (3.1%);
`val/f1` ends at `0.24703`, up from `0.08970`. The control reaches `0.32813`
`val/em` and `0.41430` `val/f1` from a step-0 `val/em` of `0.13281`.

Retrieval quality on the 128 held-out questions, original question as query
(`retrieval_precheck.json`): supporting-title recall@3 `0.0` and
any-supporting-document hit@3 `0.0` for the defective retriever, with
`all_scores_zero: true`, against `0.80859` and `0.97656` for the control. The
delivery's isolation audit counted 13959 zero document scores and exactly one
document combination across the defective trace, against 14193 non-zero scores
and 3285 combinations on the control.

## Why it is unhealthy

- `retrieval_server.py::Encoder.encode`: the zero query vector removes all
  question-dependence from retrieval, so the multi-turn agent is trained and
  evaluated on irrelevant observations. One retriever serves both paths
  (`serve_retriever.py` imports this module and `run.py` points rollout and
  evaluation at the same port), so the defect is not train-only.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `Search-R1/train_grpo.sh` under `SEARCH_R1_SHORT_GRPO=1` plus `run.py`'s
  overrides: GRPO (`algorithm.adv_estimator=grpo`), `use_kl_loss=true` with
  `kl_loss_coef=0.001` / `low_var_kl`, `entropy_coeff=0.001`, Adam `lr=1e-6`
  with no warmup, seed 42, `n_agent=4`, `max_turns=2`, `retriever.topk=3`,
  `state_masking=true`, `answer_only_reward=true`, FSDP param/grad/optimizer
  offload, `gpu_memory_utilization=0.35`, lengths
  `start=512 / prompt=3072 / response=256 / obs=768`.
- `verl/utils/reward_score/qa_em.py`: exact-match on the generated final answer
  only, no search or format reward.
- The data split (1024 train / 128 held-out HotpotQA questions, corpus of 12468
  passages covering both splits' contexts) and the 32-step budget.

## Documentation reconciliation

- The defective run **improves** by 12.5 points of EM. The defect is the
  shortfall against the control, not an absence of learning; a diagnosis
  resting on "the run does not learn" or "reward never rises" is wrong about
  the evidence even if it happens to name the right file.
- `faiss_gpu=False` in `serve_retriever.py` is intentional (CPU FAISS over a
  37MB index). "Encoder on GPU, index on CPU" is only an attribution when it
  names the unwritten buffer as the reason the data never crosses.
- The frozen run shared a host with another job and Ray warned about
  filesystem usage; neither is the cause.
- `prepare.sh` rebuilds the FAISS index from the shipped corpus instead of
  shipping it, so a rerun's index is semantically but not bitwise the recorded
  one. Expect ordinary run-to-run variation on top of that.
