# Parallel Run Isolation Audit

Checked on 2026-10-14 after both runs completed successfully (exit code 0).

- Live nvidia-smi snapshot mapped broken retriever/worker to GPU 4 (1651035/1654440)
  and fixed retriever/worker to GPU 5 (1651039/1654639). No unrelated process was
  listed on these two GPUs at that snapshot; this is not continuous monitoring.
- Separate Ray sessions: session_2026-10-14_11-15-27_949243_1651270 and
  session_2026-10-14_11-15-28_088999_1651277.
- Broken trace: all 132 records use port 8004; 13959 document scores are zero,
  with one document combination. Fixed trace: all 116 records use port 8005;
  14193 document scores are nonzero, with 3285 document combinations.
- Separate offline W&B runs: broken tojxuay5, fixed ii7myzzk. Separate output
  and checkpoint directories. Both final model files exist.
- No traceback, CUDA OOM, or ConnectionError in either training log.
- Both completed exactly 32 logged training steps, 128 trajectories per step,
  with four trajectories for each of 32 questions. Both consumed 1024 unique
  training questions once. Question sets match at every step.
- Input hashes and recorded source hashes match. Manifest fields differ only
  in mode and validation_retriever.
- At the five validation points, both runs have an executed-search rate of 1.0.
  Final answer-format rate is 1.0 for both. EM increased from 3.125% to 15.625%
  (broken), and from 13.28125% to 32.8125% (fixed).

No evidence of direct cross-run contamination was found. CPU, host memory,
filesystem bandwidth, /tmp/ray filesystem capacity and configured caches remain
shared. Ray repeatedly warned of filesystem usage above 95%, but both runs
finished; this warning is not evidence of parameter contamination. Reading
process environment variables after completion was impossible because the
processes had already exited. No claim of complete isolation is made.

This run cannot establish that concurrency improved or harmed learning: there
is no matched sequential run under the new learning rate, batch and scoring
configuration. The prior failed runs changed these factors, so they are not a
controlled concurrency comparison. No new experiment was launched by this audit.
