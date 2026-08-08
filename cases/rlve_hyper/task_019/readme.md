# task_019

PTDBench B0 package generated from a Benchmark_Evaluation RLVE historical run.

- Source run: `run_48_warmup_iters_2000_lr_warmup_decay=10000`
- Failure signature: `metric_failure`
- Buggy keys: `lr_warmup_iters, lr_decay_iters`
- Packaging tier: B0 (structure and frozen evidence validated)
- B1: requires local 4-GPU Phase2 execution and metric gate calibration

The Agent sees `workspace/` and `instruction.md`; `oracle/` and `verifier/` are
verifier-only.
