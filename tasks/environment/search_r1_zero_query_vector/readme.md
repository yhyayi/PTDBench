# Search-R1 GRPO run assessment: search_r1_zero_query_vector

## Case overview

- Evaluation key: `search_r1_zero_query_vector`
- Case family: `search_r1_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Dataset: HotpotQA `distractor`, 1024 train / 128 held-out questions, corpus of
  12468 passages
- Training framework: Search-R1 with veRL vendored (upstream
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`), FSDP + vLLM 0.6.3
- Recipe: `Search-R1/train_grpo.sh` under `SEARCH_R1_SHORT_GRPO=1`, driven by
  `run.py`
- Primary source evaluation metric: `val/em`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation. Metric names preserve the source experiment.

## Canonical failure reason

`Encoder.encode` in `Search-R1/search_r1/search/retrieval_server.py`
(lines 117-120) returns an all-zero query embedding. The zero-filled **CPU**
array it returns is never written: `query_emb` is on CUDA (`load_model` calls
`model.cuda()`), so `torch.as_tensor(response_buffer, device=query_emb.device)`
*copies* instead of aliasing, and `writer.copy_(query_emb)` fills a throwaway GPU
tensor. Under `METRIC_INNER_PRODUCT` a zero query scores `0.0` against every
document, so FAISS returns an arbitrary but constant top-3 — legal shape, legal
dtype, no exception. Held-out `val/em` reaches 15.6% at the 32-step horizon
against 32.8% for the identical recipe with the embedding copied to the host.

```python
# defective (staged)                        # control (sealed in the oracle)
response_buffer = np.zeros(...)             query_emb = query_emb.detach().cpu().numpy()
writer = torch.as_tensor(                   query_emb = query_emb.astype(
    response_buffer, device=query_emb.device)    np.float32, order="C")
writer.copy_(query_emb)
query_emb = response_buffer
```

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible snapshot of the defective side, including its completed run under `results/baseline32/`; evidence, not ground truth. |
| `environment/` | Agent-visible read-only at `/ptdbench/environment`: `prepare.sh`, `build_index.py`, and the two manifests. Deliberately free of any retrieval-quality assertion — see "Environment requirement". |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, task-local LLM-judge verifier. |
| `evaluations/solution/` | Repair-and-run instruction, sealed reference material (including the control's `retrieval_server.py` and the recorded comparison package), metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM judge.
`solution` grading reads only the new experiment logs.

## Source and preserved context

```yaml
evidence_kind: historical_run
packaging_ready: true
source: raw_tasks/hsz_search-r1-delivery          # a paired delivery, both sides real runs
control: the delivery's healthy twin, sealed under evaluations/solution/oracle/
verify:
  num_gpus: 1
  total_training_steps: 32
  eval_every: 10
  last_eval_step: 32
  eval_metric_key: val/em
historical_run:
  metric_status: trainer_log_extracted
  wandb_run_id: tojxuay5                          # offline, binary history dropped
  exit_code: 0
  eval_steps: [0, 10, 20, 30, 32]
  eval_first: 0.03125
  eval_last: 0.15625
  eval_max: 0.15625
  f1_first: 0.08970
  f1_last: 0.24703
control_at_horizon:
  val/em: 0.328125
  val/f1: 0.41430
  val/em_at_step_0: 0.1328125
solution_rerun:
  total_training_steps: 32                        # same horizon as the frozen evidence
  required_best_val_em: 0.24
  calibration_runs_per_side: 1
```

The workspace preserves the launch entry point (`run.py`, which owns both the
retrieval server and the trainer), the retriever service wrapper, the full
Search-R1 + veRL source with the defect in place, the shipped data, and the
defective run's own record: `train.log`, `launch.log`, `retrieval_server.log`,
`experiment.json`, a `retrieval.jsonl` excerpt, all 32 `rollouts/metrics_*.json`,
four `rollouts/step_*.jsonl` samples, `metrics_summary.json`, and a slimmed
offline W&B directory (no `.wandb` binary history).

## Original failure evidence

- five held-out points at steps 0, 10, 20, 30, 32: `val/em` 3.125% → 4.688% →
  11.719% → 13.281% → 15.625%, `val/f1` 8.970% → 24.703%
- the control, same seed and data and corpus and index, reaches **32.813%**
  `val/em` / **41.430%** `val/f1`, and already scores **13.281%** at step 0
  before any training
- every document score in the defective `retrieval.jsonl` is `0.0`, and the same
  three documents are returned for every query in a batch:
  `"Scott Derrickson nationality"` → `"Sheep–goat hybrid"`. The delivery's audit
  counted 13959 zero scores and exactly one document combination across the
  trace, against 14193 non-zero scores and 3285 combinations for the control.
- held-out retrieval precheck with the question as query: supporting-title
  recall@3 `0.0` and any-supporting-document hit@3 `0.0`, against `0.809` and
  `0.977` for the control (`oracle/recorded_comparison/retrieval_precheck.json`)
- the search protocol is healthy: `val/searched` 1.0, `val/format` ~1.0,
  `env/number_of_valid_search` 0.93-1.02. The model searches correctly and reads
  the observations; the observations are unrelated to the question.
- responses stay long — `response_length/mean` ~670 tokens against the control's
  ~380 — consistent with a policy that keeps searching and re-reading because
  the returned text never answers the question
- training reward rises too (`critic/score/mean` 0.016 → 0.164) and does not
  establish held-out health
- the fix is discoverable in-tree: `search_r1/search/retrieval.py:127-128` is the
  same `Encoder.encode` in a sibling module nothing serves, and it ends with the
  correct two lines (`index_builder.py:237` does the same for documents). Two
  copies of one encoder disagreeing is the cleanest diagnostic path, and the
  issue oracle credits it as evidence.

## Control baseline

The delivery is a *paired* experiment, not a single run plus a protocol
estimate: the control differs from the staged side only in those four lines
(`diff` over the two `Search-R1/` trees matches exactly one file), ran on the
same host with the same seed, split, corpus, index, encoder and budget, and
reached **32.8% `val/em` at 32 steps**. The staged side's 15.6% is 17.2 points
below it, and 10.2 points of that gap exist at step 0.

Both runs completed (`exit_code: 0`). Logged step time was 3393 s (defective)
and 3056 s (control), so roughly an hour of training each on one A100-40G, plus
retriever and vLLM startup. They ran concurrently on a shared 8-GPU host, so
those timings include contention; `oracle/recorded_comparison/ISOLATION_AUDIT.md`
is the delivery's own record of what that sharing could and could not have
affected, and it explicitly declines to claim complete isolation.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `retriever_query_device_copy` —
  `Search-R1/search_r1/search/retrieval_server.py::Encoder.encode`

Required fix targets:

- `retriever_query_device_copy`: copy the embedding to the host before FAISS
  sees it (`query_emb.detach().cpu().numpy()`, then
  `astype(np.float32, order="C")`).

Documentation reconciliation:

- The defective run **improves** by 12.5 points of EM. "The run does not learn"
  is a misreading of the evidence; the defect is the shortfall against the
  control. The oracle and `standard_answer.txt` both make that a score-0
  attribution so a judge does not reward it.
- `faiss_gpu=False` is intentional. "Encoder on GPU, index on CPU" counts as an
  attribution only when it names the unwritten buffer as the reason the data
  never crosses.
- `max_turns=2`, `topk=3`, `answer_only_reward`, `state_masking`, FSDP offload,
  `gpu_memory_utilization=0.35`, the single-GPU profile, the 32-step budget and
  the 1024/128 split are recipe configuration, not the cause.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause and evidence agreement (weights
0.2 / 0.6 / 0.2); pass is a weighted score of at least `0.8`. `judge.py`,
`judge_prompt.md` and `test.sh` are byte-identical to
`tasks/slime_search_r1_nomask`'s.

`solution` does not grade a diagnosis. The agent must repair the workspace and
run `python3 run.py --run-name graded --runs-dir "$PTDBENCH_RESULTS/experiment_logs"`
at the budget already in `config.json` (32 steps, eval every 10), giving eval
points 0, 10, 20, 30, 32. `score_mode` is `best`; the highest finite `val/em`
must land in `[0.24, 1.0]`. Pass is score `1`, otherwise `0`.

`0.24` sits 0.084 above the defective run's ceiling (`0.15625`) and 0.088 below
the control (`0.32813`), and above the control's untrained step-0 value
(`0.13281`) so that a correct fix without a completed run also fails.

**The calibration is one run per side.** `slime_search_r1_nomask` sets its bar
from four measured runs and can quote a thinnest margin; this task cannot.
Nothing here establishes seed-to-seed spread, and the defective side is the kind
of defect whose ceiling depends on how fast the policy learns to ignore useless
observations. Consequences:

- a graded best near `0.24` is inside the unmeasured band; read
  `experiment_logs/graded/train.log` and `retrieval.jsonl` before trusting the
  verdict either way,
- running a second pair at this budget is the follow-up that would justify
  tightening, widening, or moving the bar,
- reruns are not expected to match the recorded curve digit for digit: the FAISS
  index is rebuilt at prepare time rather than shipped, and GPU nondeterminism
  applies.

## Environment requirement

- Image: `search-r1-verl` (`docker/images/search-r1-verl.Dockerfile`), the stack
  `results/baseline32/experiment.json` records — Python 3.10, torch 2.4.0,
  vLLM 0.6.3, flash-attn 2.6.3, transformers 4.46.3, ray 2.40.0, faiss-cpu
  1.14.2. It is installed from the delivery's own `uv.lock`, staged at
  `docker/search-r1-runtime/`; that lock, not the Dockerfile, is the pin. The
  shared `verl` image (vLLM 0.17 / torch 2.10) is five vLLM minors past the API
  this workspace is written against.
- **The image has never been built.** It was written on a machine with no GPU and
  no reachable Docker Hub or GitHub, so the `FROM` is pinned by tag rather than
  by digest and the prebuilt flash-attn wheel URL is unverified (the build falls
  back to compiling the lock's sdist if it 404s). Pin the digest at first build;
  the Dockerfile carries the exact commands.
- `environment/prepare.sh` verifies the three shipped data files against
  `environment/data_manifest.json`, fetches `Qwen2.5-1.5B-Instruct` (revision
  `989aa798…`, `model.safetensors` sha256 `dd924a11…`, the hash the recorded run
  logged) and `intfloat/e5-base-v2` (revision `f52bf8ec…`, sha256 `d0d559c4…` —
  byte-identical to the delivery's copy and to the one
  `slime_search_r1_nomask` pins, so the two tasks share one cache entry), then
  rebuilds `shared/data/e5_Flat.index` with `environment/build_index.py`.
  Roughly 3.5 GB of weights, cached under `$PTDBENCH_MODELS`; the index is
  cached under `$PTDBENCH_DATASETS` and hardlinked per run.
- The index is rebuilt rather than shipped because it is derivable from the
  6.6 MB corpus the task does ship and is 37 MB of binary otherwise.
  `build_index.py` reproduces upstream's indexing exactly (the `passage: `
  prefix, mean pooling, L2 normalisation, `Flat` + inner product, max_length
  256) and runs on CPU when prepare gets no GPU. Its post-build assertions are
  **document-side only** — count, dimension, finiteness, unit norm — because
  `environment/` is agent-readable and a query-side retrieval-quality gate there
  would point straight at the defect. The document embeddings are identical on
  both sides of the pair, so that is exactly the part that can be asserted for
  free.
- `metadata.yaml`'s `environment.env` deliberately omits `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE`: the framework sets them per stage
  (`_network_policy_env` in `ptdbench/runtimes/base.py`), while a task-level copy
  would also disable prepare's downloads.
- One GPU. `run.py` refuses to start on a GPU that already has memory in use and
  requires 10 GiB of free disk; it also starts the retrieval server itself and
  reaps it on exit, so no stage has to keep a service alive across containers.
- Use the Docker agent backend.

The workspace code and the Search-R1 recipe are distributed under Apache-2.0.

---

## Task card

**【已构建】检索编码器把查询向量留在 GPU，FAISS 收到的是全零向量**

> @chenyinghao · `tasks/search_r1_zero_query_vector`

**一句话**：`Encoder.encode` 返回的查询向量全为零（返回的 CPU 零数组从未被写入），
内积索引下所有文档得分都是 0，FAISS 对任何问题都返回同一组 top-3；同一配方下
held-out EM 只到 15.6%，而把向量拷回主机的对照组到 32.8%，且 step 0 就已经是
3.1% vs 13.3%。

**设定**

GRPO / Qwen2.5-1.5B-Instruct / HotpotQA distractor（1024 训练 + 128 验证，12468
段语料）/ Search-R1 + 内置 veRL `598e61bd` + vLLM 0.6.3 + torch 2.4.0

镜像：`search-r1-verl`（`ptdbench/search-r1-verl:vllm0.6.3-torch2.4-cu121`，未构建）

训练耗时：约 1h（日志记录的 step 时间 3393 s，另加检索服务与 vLLM 启动）

资源：1 x A100-40G

**Bug**

- 症状：检索结果与问题无关——`retrieval.jsonl` 里所有文档得分都是 0.0，每个 query
  返回同一组文档；held-out `val/em` 32 步只到 15.6%，对照组 32.8%。搜索协议本身正常
  （`val/searched` 1.0，`val/format` ~1.0）。
- 根因：`Search-R1/search_r1/search/retrieval_server.py::Encoder.encode` 117-120 行。
  `response_buffer` 是 CPU 上的零数组，而 `query_emb` 在 GPU 上
  （`load_model` 无条件 `model.cuda()`），`torch.as_tensor(..., device=query_emb.device)`
  因跨设备而**拷贝**出一个新的 GPU tensor，不与原数组共享存储，所以
  `writer.copy_()` 写不回 `response_buffer`。
- 为何复现触发：零向量在 `METRIC_INNER_PRODUCT` 下是合法输入，所有文档得分相等，
  FAISS 正常返回 top-k 不报错；形状、dtype、HTTP 契约全都对。同一段代码在 CPU-only
  的编码器上是正确的（那时 `as_tensor` 会共享内存），所以读起来像对的。训练和验证
  共用一个检索服务，两条路径都受影响。
- 修复：两行——`query_emb = query_emb.detach().cpu().numpy()` 再
  `astype(np.float32, order="C")`。起作用的是 `.cpu()`；`.detach()` 只是解除梯度关联。

**评测（两级）**

- `issue` 级（复现 + 诊断）：decider = LLM judge 对 `oracle.md`，verdict/root_cause/
  evidence 权重 0.2/0.6/0.2，加权 ≥ 0.8 通过。`agent_answer.md` 需指出
  `Encoder.encode` 里那块没被写入的 buffer 导致查询向量为零，且**不得**说成
  "训练没有提升"（故障组实际涨了 12.5 个点），也不得只说 "索引在 CPU、编码器在 GPU"。
- `solution` 级（应用修复）：decider = 修好后按 `config.json` 原预算重跑 32 步，
  只读 `$PTDBENCH_RESULTS/experiment_logs` 里的新日志。
- 指标 / 阈值：`val/em`，`score_mode: best`，阈值 **0.24**（故障组上限 0.15625 在阈值
  下方 0.084，对照组 0.32813 在上方 0.088；对照组 step 0 的 0.13281 也在阈值下方，
  所以"只改代码不跑完"同样不过）。
- 防退化：verifier 只认 `experiment_logs` 下的日志，上界 1.0 挡伪造值，`--runs-dir`
  由 instruction 固定；`run.py` 拒绝复用 run 名、拒绝在已占用的卡上启动。
- 已知弱点：阈值只有**每侧一次**运行作为标定，没有方差估计（对照任务
  `slime_search_r1_nomask` 有四次）。补一对重跑是下一步。

**关键结果**

| 指标 | buggy | fixed |
|---|---|---|
| `val/em` best（decider） | 0.15625 | 0.32813 |
| `val/em` step 0 | 0.03125 | 0.13281 |
| `val/f1` step 32 | 0.24703 | 0.41430 |
| supporting-title recall@3 | 0.000 | 0.809 |
| `response_length/mean` step 32 | 668 | 390 |

单 seed、128 题验证集的成对短程实验，不构成统计显著性声明。两组均 exit 0、完成
全部 32 步、消耗同样的 1024 道训练题；两边 `Search-R1/` 源码树的差异只有
`retrieval_server.py` 一个文件。
