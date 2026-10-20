Recorded training run, 2026-10-16, 2x NVIDIA H800, 60 steps, exit code 0.

Files, in order of how much you should trust them:

  metrics_20261016_170257.jsonl
      Per-step metrics, one JSON object per line: {"step": N, "data": {...}}.
      Written unbuffered by verl's `file` tracking backend
      (VERL_FILE_LOGGER_PATH, set by the launch script). 61 rows: step 0 is
      the pre-training validation, steps 1-60 are training. This is the
      authoritative record of the run.

  llama3_2_1b_instruct_dapo_math_fsdp_20261016_170257.log
      The launch script's own log file: everything the trainer wrote to
      stdout/stderr, including warnings, the tqdm progress stream, and the
      `step:N - key:value - ...` lines from the `console` backend.

  ray_taskrunner_stdout.out
      The Ray worker log for the TaskRunner actor, which is where the
      `console` backend's metric lines are actually printed. Kept because
      whether Ray forwards this actor's stdout into the launch script's log
      file is not guaranteed: an earlier run of this same recipe on this same
      image produced a log file with zero metric lines while this file had all
      of them. If the two disagree, prefer the .jsonl.

  run_console_output.log
      The command-line output of the launch wrapper: host, GPU inventory, the
      resolved model/data paths, and the exit code.

No offline W&B data is included. The run writes it (WANDB_MODE=offline), but
this verl tree has no Tracking.finish() call, so the process exits while the
datastore is mid-write and the resulting run-*.wandb ends on an unterminated
record fragment. An incomplete binary is worse than none; the .jsonl above
carries the same numbers in full.

Launching again writes new files here with a fresh timestamp. Nothing in this
directory is read by any grader.
