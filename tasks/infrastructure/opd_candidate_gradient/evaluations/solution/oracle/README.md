# Solution oracle

Hidden reference material for the `solution` level. Not exposed to the agent.

## The fix

`reference.patch` is the reference repair for
`verl/trainer/opd/kernels.py::sparse_log_probs`. It replaces the per-column
`logprobs_from_logits` loop (which passes a non-contiguous strided view of
`candidate_ids` into the optimized CUDA cross-entropy wrapper) with a single
consistent log-softmax / gather:

```python
log_probs_all = torch.log_softmax(logits, dim=-1, dtype=_accum_dtype(logits))
out["cand_log_probs"] = torch.gather(log_probs_all, -1, candidate_ids)
```

Apply it to the published workspace to obtain the control side. The patch paths
are relative to the verl repo root, which sits at `workspace/verl/`:

```console
$ patch -p1 -d ../../../workspace/verl --fuzz=0 < reference.patch
# one hunk, in verl/trainer/opd/kernels.py
```

Any functionally equivalent repair is accepted by the verifier — e.g. making
`candidate_ids[..., i]` contiguous before the optimized call. The verifier does
not match this patch verbatim; it checks forward values and weighted backward
gradients against `log_softmax(logits, -1).gather(-1, candidate_ids)`, duplicate
candidate ids, and the detached top-k outputs. The repair must preserve the
public `sparse_log_probs` return contract and keep the single-candidate and
CPU-fallback paths correct.
