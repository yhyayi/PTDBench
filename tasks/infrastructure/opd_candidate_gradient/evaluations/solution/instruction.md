# Repair the OPD candidate-gradient path

`/workspace` is a colocated top-k on-policy-distillation (OPD) training setup
built on `verl` (DeepSeek-R1-Distill-Qwen-1.5B student, JustRL-DeepSeek-1.5B
teacher). The top-k OPD objective computes a set of candidate-token
log-probabilities in `verl/trainer/opd/kernels.py::sparse_log_probs`. The run
completes with finite numbers, but the candidate log-probabilities and their
backward gradients disagree with the mathematically expected
`log_softmax(logits, -1).gather(-1, candidate_ids)` whenever there is more than
one candidate column on the optimized device path.

Diagnose and repair the workspace so the colocated top-k OPD run produces the
mathematically expected optimizer update on all supported device paths. Keep
the existing public interfaces and detached diagnostic outputs
(`sparse_log_probs`'s return contract: `cand_log_probs`, `topk_ids`,
`topk_log_probs`, `kth_log_prob`) compatible with their current callers.

## What to change

Fix the candidate log-probability computation in
`verl/trainer/opd/kernels.py`. A functionally equivalent implementation is
accepted: make the candidate labels contiguous before the optimized call, or
compute the candidate probabilities from one consistent log-softmax/gather
path. A single candidate column and the CPU fallback must keep working.

Keep the change focused on the workspace implementation. Do **not** edit the
evaluation files or add a special case keyed to the verifier. Run the available
tests and any small numerical probes needed to establish correctness.

## How you will be checked

A hidden, deterministic regression test exercises `sparse_log_probs`:

- It emulates the optimized cross-entropy label contract on a CPU-only runner,
  and — when CUDA is available — additionally exercises the real CUDA path.
- It checks forward values and weighted backward gradients against the
  reference `log_softmax`/`gather`, handles duplicate candidate ids, and
  asserts the detached top-k outputs (`topk_ids`, `kth_log_prob`,
  `topk_log_probs.requires_grad == False`).

The optional full-run parameters in `verifier/eval_params.env` are provided for
maintainers only (`PTDBENCH_RUN_FULL_OPD=1`); the functional verifier itself is
deterministic and needs no GPU.
