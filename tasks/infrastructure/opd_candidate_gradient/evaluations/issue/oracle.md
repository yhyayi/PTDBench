# Oracle diagnosis

Expected verdict: **unhealthy**.

## The component

The failure is in `workspace/verl/verl/trainer/opd/kernels.py`, in
`sparse_log_probs`. For the top-k OPD objective it computes candidate
log-probabilities by looping over the last dimension of `candidate_ids` and
calling `logprobs_from_logits` once per column:

```python
out["cand_log_probs"] = torch.stack(
    [
        logprobs_from_logits(
            logits=logits,
            labels=candidate_ids[..., i],
            inplace_backward=False,
        )
        for i in range(width)
    ],
    dim=-1,
)
```

For a contiguous `(N, K)` candidate matrix with `K > 1`, each `candidate_ids[..., i]`
is a one-dimensional **non-contiguous** view with stride `K`.

## The mechanism

The CUDA / FlashAttention cross-entropy wrapper behind `logprobs_from_logits`
flattens its labels with `reshape(-1)`. For a one-dimensional strided view,
`reshape(-1)` preserves the stride instead of making a contiguous copy, and the
optimized kernel then consumes the label buffer as a packed vector. So it reads
neighboring candidate slots rather than the requested column. The selected
log-probabilities — and their backward gradients — come out finite but do not
equal `log_softmax(logits, -1).gather(-1, candidate_ids)`.

## Why it looks healthy

The result is still a finite tensor of the right shape and a similar scale, so
nothing crashes and no NaN/Inf appears. The scalar loss is plausible. Only the
*direction* of the gradient / optimizer update is wrong, which does not show up
as a loud failure — it shows up as a materially different update from a
reference implementation, and degraded training downstream.

## Trigger conditions

The bug requires **both** the optimized CUDA / FlashAttention path **and** more
than one candidate column (`K > 1`, i.e. top-k OPD with multiple candidates). A
single candidate column, or the ordinary CPU fallback, produces a contiguous or
correctly-copied label buffer and masks the defect.

## Repair direction

Make the candidate labels contiguous before the optimized call
(`candidate_ids[..., i].contiguous()`), or compute the candidate
log-probabilities from one consistent `log_softmax` / `gather` path (and keep
that path consistent for the sampled and top-k consumers) while preserving the
public `sparse_log_probs` return contract.

## Wrong attributions

- `inplace_backward=False` already prevents the separate in-place-storage
  hazard; it does **not** repair the label layout, so "just set
  `inplace_backward`" is not the fix.
- The Python stack/loop itself accumulates gradients correctly — blaming the
  loop or gradient accumulation alone is not sufficient.
- "Training did not improve" / "the loss is finite so the run is fine" are
  symptoms or distractors, not the root cause.
