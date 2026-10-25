# Cybertron / Harbor 运行手册（runbook）


平台镜像（内建 image_map）：

`--image` 只认 **image 名**，即任务 `metadata.yaml` 的 `image` 字段值；后端按下表的
`image_map`（`harbor/src/harbor/environments/cybertron.py` 的 `_DEFAULT_IMAGE_MAP`）
解析成 Cybertron `canonicalName` 交给 `cctl`，**不再用 id 指定镜像**。评测路径（harbor-evaluate）
本就从任务 metadata 取 image 名，一般无需手填；手动 stage / preheat 时才需要 `--image <名>`。

| image 名（metadata.image） | canonicalName |
|---|---|
| `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` | `share-dev/ngc-docker/ptdbench-slime:dev-96bfe26-cb-55f572c7` |
| `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` | `share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb` |
| `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` | `infra-platform/ptdbench-vllm0.11-torch2.8-cu128:latest` |
| `slime-v0.5.0rc0-sglang0.5.0rc0-torch2.8.0-flashattn2.7.4-cu126-py3.12` | `share-dev/ngc-docker/ptdbench-slime-rc0:dev-f110ee9-cb-8452036a` |

> canonicalName 里带内容 tag，镜像重建后用 `cctl image get <id> -F canonical_name` 刷新 `_DEFAULT_IMAGE_MAP`。
> 其余任务镜像（`miles` / `search-r1-verl` / `verl-moe-token` / `verl-qwen3vl-cu130`）暂未登记平台镜像，
> 映射缺失时报错列出已知 image 名；登记后把一行加进 `_DEFAULT_IMAGE_MAP` 即可。

面向 PTDBench 任务的两条常用流程：

1. **手动实验** —— 拉起 devspace 并构建到「agent 开始 debug 之前」的状态，供你手动跑训练验证环境；
2. **评测 agent** —— 端到端：建 devspace → 起 agent → verify 打分。

> 关键默认值（`harbor/src/harbor/environments/cybertron.py`）：
> cluster `wuxi_train`、resource-pool `speech-wx`、billing `N00069`、gpu-model `h100`、
> cpu/memory 随 GPU 数缩放（默认每卡 16 CPU、152GB，按 `max(gpus,1)` 算；显式传 `cpu`/`memory_gb` 可覆盖）、
> cluster-user `chenyinghao`、teleport-cluster `teleport.cybertron.modelbest.co`、
> 缓存目录 `/user/<user>/ptdbench-cache/{models,datasets}`（`/user` 为集群持久盘，自动挂载到每个 pod）。
>
> 前置：`cctl login` 已登录、`tsh login` 已登录（`tsh` 在 PATH 上，或用 Cursor Cybertron 插件自带的那份）。

---

## 1. 手动实验：构建到「agent 开始 debug 之前」的状态

### 1a. 本地 docker 版（最快，docker 直连分析）

`harbor-shell` 会跑完 `prepare.sh`、把 workspace 物化好，然后把你丢进
**与 agent 阶段同一个容器**（但不启动 agent）的交互式 shell：

```bash
python3 -m ptdbench harbor-shell tasks/reward/retool_length_hack \
  --evaluation-level solution \
  --workspace-dir outputs/harbor-shell/retool_length_hack
# 可选：--force-prepare 重跑 prepare、--gpus all、--shm-size 32g、--shell bash
```

进 shell 后直接手动跑该任务的训练脚本即可复现 / 调试环境。
（`harbor-shell` 等价于 `harbor-stage --backend docker`，后者是跨后端的统一入口。）

### 1b. Cybertron devspace 版（真集群、多卡）

一条命令跑到 pre-agent 手动态（建/attach devspace → `install-ptdbench-tooling.sh` →
上传任务 → 在 devspace 上跑 `prepare.sh`），然后**留着 devspace 不停**，打印 `tsh ssh` 入口：

```bash
python3 -m ptdbench harbor-stage tasks/reward/retool_length_hack --backend cybertron \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069 \
  --run                                   # 不加 --run 即 dry-run，只打印将做什么
# --image / --gpus 默认读任务 metadata.yaml（image 字段 + CUDA_VISIBLE_DEVICES 卡数），
#   想覆盖再显式传，如 --gpus 4 --image <别的 image 名>
# attach 已有 devspace（不新建、不销毁）：--devspace-id tasks/1098539
# 停掉自建的 devspace：--stop（attach 的永不停）
```

命令打印出 `tsh ssh --cluster=… root@<node>` 后，ssh 进去 `cd /workspace` 即是 prepare 好的工作区，
按任务 README / 启动脚本手动起训练、观察日志、复现 bug。用完 `cctl devspace stop <id>` 释放资源。

> harbor-stage 复用后端 `CybertronEnvironment`，一条命令里自动完成建/attach devspace、
> `install-ptdbench-tooling.sh`、上传任务、在 devspace 上跑 `prepare.sh`，无需再手动
> `cctl devspace create` + `tsh ssh` + 导出缓存环境变量那条链路。

---

## 2. 评测 agent（建 devspace → 起 agent → verify）

### 2a. 一条命令（推荐，走 ptdbench harbor-evaluate + cybertron 后端）

`harbor-evaluate` 会：跑 `prepare.sh` → 物化 Harbor 任务 → 起 agent → 自动跑 verifier 打分。
task 路径必须在所有转发给 `harbor run` 的参数**之前**：

```bash
python3 -m ptdbench harbor-evaluate tasks/reward/retool_length_hack \
  --evaluation-level solution \
  --judge-model gpt-5.5 \
  --agent codex --model gpt-5.5 \
  --env cybertron \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069
# 其余参数（--models-dir/--datasets-dir/--agent-network/--harbor-command…）见 `python3 -m ptdbench harbor-evaluate -h`
```

后端选择 `--env cybertron`、后端 kwargs 用可重复的 `--environment-kwarg key=value`
（会并入 `config.environment.kwargs`；也可写死在任务 `task.toml` 的 `[environment].kwargs`）。

### 2b. 复用现有 helper 脚本

`run_harbor_fit.sh` 用 `--env "${PTDBENCH_BACKEND:-docker}"`，额外参数经 `"$@"` 透传：

```bash
PTDBENCH_BACKEND=cybertron PTDBENCH_TASK=tasks/reward/retool_length_hack \
  ./run_harbor_fit.sh \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069
```

（脚本已设好 `OPENAI_API_KEY`/`OPENAI_BASE_URL`、清华 pip 镜像、npmmirror、judge/agent=gpt-5.5 等。）
多任务用 `run_harbor_multi.sh`（`--` 前是 task 根，`--` 后透传给 `harbor run`）。

### 2c. attach 模式（复用已存在的 devspace，不新建）

先按 §1b 建好 devspace，拿到 `devspace_id`，评测时加一个 kwarg 复用它：

```bash
... --environment-kwarg devspace_id=tasks/1098539
```

不传 `devspace_id` 时后端自建、结束自动 `stop`；传了则复用且不销毁（调试更省事）。

### 2d. 原生 harbor run（已物化任务时）

```bash
harbor run -t <materialized-task-dir> -e cybertron \
  --agent codex --model gpt-5.5 \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069
```

（`harbor run` 是 `harbor job start` 的别名。）

### verify / 结果

- Harbor 在 agent 结束后自动跑 verifier 阶段（`tests/test.sh`），reward 写到
`/logs/verifier/reward.txt`（或 `reward.json`）。
- PTDBench 的 metric-only verifier 会从 `$PTDBENCH_RESULTS/experiment_logs`（含离线 W&B）
重建 `eval/aime`，按 `resources/grader_spec.json` 打分（如 retool_length_hack：
`score_mode=best`、`required_final_score=-0.55`、`required_final_score_max=0.5`）。
- 分数与 trace 落在 `--jobs-dir`（helper 默认 `outputs/harbor-slime-retool`）下的 trial 输出里。

---

## 常用运维

```bash
cctl devspace get <id> -F status      # 看状态
cctl devspace stop <id>               # 用完释放（省钱、还资源）
tsh login / cctl login                # 登录过期时重登
```

Cybertron 侧一次性 setup（后端 `start()` 自动做，手动 ssh 时通常无需关心）：
编译 `llrun`（Landlock 文件隔离启动器，因 devspace 禁 mount / bwrap）、
装 rename shim（`/etc/ld.so.preload`）、设 `no_proxy` 走节点内网 IP。
