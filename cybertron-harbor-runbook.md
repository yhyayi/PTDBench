# Cybertron / Harbor 运行手册（runbook）

面向 PTDBench 任务的三条常用流程：

1. **准备数据与模型** —— 跑通所有 `prepare.sh`，把权重 / 数据集缓存好；
2. **手动实验** —— 拉起 devspace 并构建到「agent 开始 debug 之前」的状态，供你手动跑训练验证环境；
3. **评测 agent** —— 端到端：建 devspace → 起 agent → verify 打分。

> 关键默认值（`harbor/src/harbor/environments/cybertron.py`）：
> cluster `wuxi_train`、resource-pool `speech-wx`、billing `N00069`、gpu-model `h100`、cpu 16、memory 128GB、
> cluster-user `chenyinghao`、teleport-cluster `teleport.cybertron.modelbest.co`、
> 缓存目录 `/user/<user>/ptdbench-cache/{models,datasets}`（`/user` 为集群持久盘，自动挂载到每个 pod）。
>
> 平台镜像 id（内建 image_map）：
> | image 名 | id | 说明 |
> |---|---|---|
> | `ptdbench-verl-vllm017` / `ptdbench-verl` | `25607123` | verl，CUDA 12.9 |
> | `ptdbench-slime-rc0` | `25607122` | slime rc0，CUDA 12.6 |
> | `ptdbench-slime-v0.3.2-cu129` / `ptdbench-slime-v0.3.2` / `ptdbench-slime` | `25607131` | slime v0.3.2，**已重建到 CUDA 12.9**，可在 wuxi_train/speech-wx 跑 |
>
> 前置：`cctl login` 已登录、`tsh login` 已登录（`tsh` 在 PATH 上，或用 Cursor Cybertron 插件自带的那份）。

---

## 1. 准备数据与模型（跑所有 prepare.sh）

### 1a. 集群预热到共享持久盘（推荐，一次性）

用 `harbor/scripts/cybertron_preheat.py` 生成并提交一个 `cctl job`（BATCH），它会：
导出 `PTDBENCH_MODELS` / `PTDBENCH_DATASETS` 指到 `/user/<user>/ptdbench-cache/{models,datasets}`，
`mkdir` 好这两个目录，作业启动时由 `cctl --code-type git --git-path` 把仓库 clone 到节点，
然后对每个匹配到的 `prepare.sh` 执行 `(cd <dir> && bash prepare.sh)`。

> **必须按镜像分组跑**：`prepare.sh` 依赖镜像内的 slime / verl / Megatron 环境（还会在有 GPU 时
> 就地做 torch_dist 转换），所以每个任务只能在它 `metadata.yaml` 绑定的镜像里预热。三个镜像分三次提交，
> 各自用 `--image` + `--prepare-glob` 圈定对应任务：

| 镜像（metadata `image`） | 平台 id | 覆盖的任务（`--prepare-glob`） |
|---|---|---|
| `slime-rc0`（cu126） | `25607122` | `tasks/rlve_hyper/*/environment/prepare.sh` |
| `slime-v0.3.2`（cu129） | `25607131` | `tasks/slime_retool/environment/prepare.sh tasks/slime_retool_length_hack/environment/prepare.sh` |
| `verl`（cu129） | `25607123` | `tasks/reward_design/*/environment/prepare.sh tasks/verl_hyper/*/*/environment/prepare.sh tasks/verl_implementation/*/*/environment/prepare.sh` |

```bash
# 公共参数（每条命令都带）；不加 --run 即 dry-run，只打印将提交的 cctl job 命令
COMMON=(--project xinfer-chenyinghao
        --cluster wuxi_train --resource-pool speech-wx --billing-account-id N00069
        --gpu 1 --gpu-model h100 --cpu 16 --memory 128
        --git-path git@github.com:<org>/ptdbench.git --git-ref main
        --cluster-user chenyinghao)

# ① slime-rc0（cu126）: rlve_hyper 四个子任务
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607122 \
  --prepare-glob 'tasks/rlve_hyper/*/environment/prepare.sh' \
  --run

# ② slime-v0.3.2（cu129）: slime_retool + length_hack（search_r1* 见下方说明，暂缓）
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607131 \
  --prepare-glob 'tasks/slime_retool/environment/prepare.sh tasks/slime_retool_length_hack/environment/prepare.sh' \
  --run

# ③ verl（cu129）: reward_design ×2 + verl_hyper ×1 + verl_implementation ×3
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607123 \
  --prepare-glob 'tasks/reward_design/*/environment/prepare.sh tasks/verl_hyper/*/*/environment/prepare.sh tasks/verl_implementation/*/*/environment/prepare.sh' \
  --run
```

- 默认 `--cache-dir /user/chenyinghao/ptdbench-cache`；`--prepare-glob` 是一个字符串、内含空格分隔的
  多个 glob，相对仓库根（node 上 shell 展开，匹配不到的模式自动跳过）。默认值
  `"tasks/*/environment/prepare.sh environment/prepare.sh"` 只覆盖直接子目录，**匹配不到 rlve_hyper /
  verl_implementation 这类嵌套任务**，所以上面按镜像显式给全路径。
- **torch_dist 转换**：`prepare.sh` 里带 GPU 探测——有可用 GPU 才就地把 HF 权重转成 mcore/torch_dist
  ref 权重并缓存；CPU-only 预热会跳过、留给训练首跑兜底。想连转换结果一起缓存就保持 `--gpu 1`（且节点真有卡）。
- **search_r1（`slime_search_r1` / `_nomask`）暂缓**：metadata 也绑 `slime-v0.3.2`，但数据量大（~79GB +
  wiki-18 索引）且有 cu130/确定性复现的未决问题，之前 deferred，未纳入 ② 。要预热时单独一条，`--prepare-glob
  'tasks/slime_search_r1/environment/prepare.sh tasks/slime_search_r1_nomask/environment/prepare.sh'`。
- 已缓存约 ~20GB（slime_retool*/rlve/verl 的 prepare 已跑通）；缓存是宝贵数据，**切勿 `rm -rf`**。

### 1b. 本地 docker（单机自测）

本地路径下 `prepare.sh` 由 `harbor-evaluate` / `harbor-shell` 自动触发；
也可以显式指定缓存目录（默认 `<repo>/models`、`<repo>/datasets`）：

```bash
export PTDBENCH_MODELS_DIR=/abs/models
export PTDBENCH_DATASETS_DIR=/abs/datasets
```

---

## 2. 手动实验：构建到「agent 开始 debug 之前」的状态

### 2a. 本地 docker 版（最快，docker 直连分析）

`harbor-shell` 会跑完 `prepare.sh`、把 workspace 物化好，然后把你丢进
**与 agent 阶段同一个容器**（但不启动 agent）的交互式 shell：

```bash
python3 -m ptdbench harbor-shell tasks/slime_retool \
  --evaluation-level solution \
  --workspace-dir outputs/harbor-shell/slime_retool
# 可选：--force-prepare 重跑 prepare、--gpus all、--shm-size 32g、--shell bash
```

进 shell 后直接手动跑该任务的训练脚本即可复现 / 调试环境。

### 2b. Cybertron devspace 版（真集群、多卡）

**① 建 devspace**（与后端 `_create_devspace` 同参数，手动多要几张卡）：

```bash
cctl devspace create \
  --project xinfer-chenyinghao \
  --cluster wuxi_train --resource-pool speech-wx --billing-account-id N00069 \
  --image 25607131 \
  --gpu 8 --gpu-model h100 --cpu 16 --memory 128 \
  --description manual-slime-retool
# 输出里的 name/id 即 devspace_id（形如 tasks/1098539）
```

**② SSH 进去**（tsh 传输，无需 `~/.ssh/config`）：

```bash
tsh ls --format json                 # 找到名字以 -<数字id> 结尾的节点
tsh ssh --cluster=teleport.cybertron.modelbest.co root@<node>
```

**③ 在 devspace 内构建到 pre-agent 状态并手动跑**：

```bash
# 指到持久盘上的共享缓存（第 1 步已预热）
export PTDBENCH_MODELS=/user/chenyinghao/ptdbench-cache/models
export PTDBENCH_DATASETS=/user/chenyinghao/ptdbench-cache/datasets

cd <repo>/tasks/slime_retool/environment && bash prepare.sh   # 若缓存已在，秒过
# 然后按任务 README / 启动脚本手动起训练，观察日志、复现 bug
```

> 注：`harbor-shell` 是 docker-only 的；在 Cybertron 上「pre-agent 手动态」就是
> 「建 devspace + ssh + 手跑 prepare/训练」这条链路。用完记得 `cctl devspace stop <id>`
> 释放资源。

---

## 3. 评测 agent（建 devspace → 起 agent → verify）

### 3a. 一条命令（推荐，走 ptdbench harbor-evaluate + cybertron 后端）

`harbor-evaluate` 会：跑 `prepare.sh` → 物化 Harbor 任务 → 起 agent → 自动跑 verifier 打分。
task 路径必须在所有转发给 `harbor run` 的参数**之前**：

```bash
python3 -m ptdbench harbor-evaluate tasks/slime_retool \
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

### 3b. 复用现有 helper 脚本

`run_harbor_fit.sh` 用 `--env "${PTDBENCH_BACKEND:-docker}"`，额外参数经 `"$@"` 透传：

```bash
PTDBENCH_BACKEND=cybertron PTDBENCH_TASK=tasks/slime_retool \
  ./run_harbor_fit.sh \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069
```

（脚本已设好 `OPENAI_API_KEY`/`OPENAI_BASE_URL`、清华 pip 镜像、npmmirror、judge/agent=gpt-5.5 等。）
多任务用 `run_harbor_multi.sh`（`--` 前是 task 根，`--` 后透传给 `harbor run`）。

### 3c. attach 模式（复用已存在的 devspace，不新建）

先按 §2b 建好 devspace，拿到 `devspace_id`，评测时加一个 kwarg 复用它：

```bash
... --environment-kwarg devspace_id=tasks/1098539
```

不传 `devspace_id` 时后端自建、结束自动 `stop`；传了则复用且不销毁（调试更省事）。

### 3d. 原生 harbor run（已物化任务时）

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
  重建 `eval/aime`，按 `resources/grader_spec.json` 打分（如 slime_retool_length_hack：
  `score_mode=best`、`required_final_score=0.18`）。
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
