# OPD CUDA candidate-gradient divergence

Training DeepSeek-R1-Distill-Qwen-1.5B with colocated top-k on-policy
distillation (OPD) on a processed DAPO-Math-17k derivative in `verl`. The top-k
candidate log-probabilities are computed by passing a non-contiguous strided
view of the candidate ids into the optimized CUDA cross-entropy kernel, so the
run stays finite and crash-free while its optimizer update diverges from the
mathematically expected one.

| | |
| --- | --- |
| **Author** | Yaxuan Li |
| **Type** | infrastructure |
| **Model** | DeepSeek-R1-Distill-Qwen-1.5B (student) + JustRL-DeepSeek-1.5B (teacher) |
| **Dataset** | DAPO-Math-17k processed derivative (train); AIME26 / AIME25 / AMC23 (held-out) |
| **Framework** | `verl`, colocated top-k OPD, `opd.sh` |
| **Environment** | `verl-opd` (not yet built) · 8 GPUs for the full run; none for the verifier |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 historical defective run (`workspace/logs/opd_only_stu_k16_*.log`, top_k=16); no control run — the solution verifier is a deterministic numerical regression, not a metric calibrated from runs |

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation. The agent sees only `workspace/` and the one evaluation's
`instruction.md`; the oracle, the reference patch, and the solution verifier all
live under `evaluations/`, which it cannot read.

## Difficulty

The defect is one block in `workspace/verl/verl/trainer/opd/kernels.py`,
`sparse_log_probs`:

```python
out["cand_log_probs"] = torch.stack(
    [
        logprobs_from_logits(logits=logits, labels=candidate_ids[..., i],
                             inplace_backward=False)
        for i in range(width)
    ],
    dim=-1,
)
```

For a contiguous `(N, K)` candidate matrix with `K > 1`, `candidate_ids[..., i]`
is a 1-D view with stride `K`. The optimized CUDA / FlashAttention cross-entropy
wrapper flattens labels with `reshape(-1)`, which for a strided 1-D view keeps
the stride instead of copying to contiguous storage; the kernel then reads
neighboring candidate slots. The selected log-probabilities and their backward
gradients come out **finite** and at a **similar scale**, so there is no NaN/Inf,
no crash, and a plausible scalar loss — only the gradient direction is wrong.

The single strongest distractor is exactly that finiteness: "it ran and the
numbers look sane" is not health here. The two conditions that must both hold —
the optimized CUDA path **and** more than one candidate column (`K > 1`) — are
also what rule out the wrong diagnoses: a single candidate column or the CPU
fallback produces a contiguous/copied label buffer and hides the bug, and
`inplace_backward=False` already handles the separate in-place-storage hazard
without touching the label layout.

## Reference solution

- **issue**: name `sparse_log_probs` / `logprobs_from_logits` and the candidate
  label layout (a non-contiguous stride-`K` view) reaching the optimized
  CUDA/FlashAttention cross-entropy path, state the `K > 1` + CUDA trigger,
  explain why the forward stays finite while the backward gradient diverges, and
  point the repair at a contiguous copy or a consistent log-softmax/gather.
  Attributions that must fail: "set `inplace_backward`", "blame the Python
  loop / gradient accumulation", and "finite loss ⇒ healthy".
- **solution**: `evaluations/solution/oracle/reference.patch` replaces the
  per-column loop with a single `log_softmax` + `gather`
  (`torch.gather(torch.log_softmax(logits, -1, dtype=_accum_dtype(logits)), -1, candidate_ids)`),
  preserving the public `sparse_log_probs` return contract. Any functionally
  equivalent repair (e.g. `candidate_ids[..., i].contiguous()`) is accepted.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: a deterministic regression test
  (`evaluations/solution/verifier/tests/test_candidate_gradient.py`) — not a
  metric threshold calibrated from runs. It emulates the optimized kernel's
  packed-label contract on a CPU-only runner (and additionally exercises the
  real CUDA path when a GPU is present), and asserts the candidate forward
  values and weighted backward gradients match
  `log_softmax(logits, -1).gather(-1, candidate_ids)` to `1e-10` (CPU) /
  `2e-4` (CUDA), with duplicate candidate ids present and the detached top-k
  outputs (`topk_ids`, `kth_log_prob`, `topk_log_probs.requires_grad == False`)
  intact. The shipped defective kernel fails it; the reference patch passes it.
  The optional full OPD run (`PTDBENCH_RUN_FULL_OPD=1`, parameters in
  `verifier/eval_params.env`) needs 8 GPUs and is for maintainers only.

The `verl-opd` image (vLLM 0.24 / torch 2.11 / CUDA 13.0) is **not yet built**:
its base is not pinned by digest, and the task has not been run end to end on a
built copy. Build and pin before recording outcomes against it.
