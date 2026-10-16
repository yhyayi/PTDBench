# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, retrieval, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, model, training-step budget, and
evaluation interval are fixed.

The launch script is:

```text
run.py
```

Run it exactly like this, from `/workspace`:

```bash
python3 run.py --run-name graded --runs-dir "$PTDBENCH_RESULTS/experiment_logs"
```

That entry point owns the whole run: it starts the local retrieval server on
the port in `config.json`, waits for it to answer, streams training output, and
shuts the server down when training ends or is interrupted. Do not start the
retriever yourself and do not bypass `run.py`.

The effective run must use:

- base model `/workspace/shared/models/Qwen2.5-1.5B-Instruct`
- train data `/workspace/shared/data/train.parquet`
- validation data `/workspace/shared/data/test.parquet`
- retrieval corpus and index `/workspace/shared/data/{corpus.jsonl,e5_Flat.index}`
  with the query encoder `/workspace/shared/models/e5-base-v2`
- the budget already in `config.json`: `steps` 32, `eval_every` 10,
  `batch_size` 32, `learning_rate` 1e-06 — do not change these
- `WANDB_MODE=offline` (`run.py` sets this; do not log in to W&B)

`python3 run.py --check` validates the interpreter, the assets, the local
imports and the data row counts without a GPU or any training; it is the cheap
way to confirm the workspace is runnable before the real run.

Two guards in `run.py` will refuse to start rather than produce a bad run: it
needs at least 10 GiB of free disk, and it refuses a GPU that already has
memory in use. Release anything you started on the GPU before launching, and do
not reuse a `--run-name` that already exists.

The run is scored only from files under `$PTDBENCH_RESULTS/experiment_logs`;
logs written anywhere else are ignored. `run.py` creates
`experiment_logs/graded/` with `train.log`, `retrieval_server.log`,
`retrieval.jsonl`, `rollouts/`, `run_config.json`, `exit_status.json` and the
offline W&B directory. Do not copy, edit, or fabricate log metrics. Otherwise
the run is scored 0.
