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
- **HF 镜像 + 出口代理**：节点无法直连 `huggingface.co`，且 `hf-mirror.com` 也只能经集群白名单
  代理出网。脚本默认 `--hf-endpoint https://hf-mirror.com` +
  `--http-proxy http://whitelist-proxy.cybertron.svc.cluster.local:7891`，跑 `prepare.sh` 前先
  `export http(s)_proxy=… no_proxy=localhost,127.0.0.1,.modelbest.co,.svc.cluster.local` 再
  `export HF_ENDPOINT=… HF_HUB_DISABLE_XET=1`。**代理是关键**：Teleport 只给交互式 `tsh ssh`
  会话注入这个代理，BATCH `--entry` 不继承，漏了它 HF 拉取会 `ConnectTimeout`。要改/关代理传
  `--http-proxy <url>` / `--http-proxy ''`；镜像同理用 `--hf-endpoint`。
- **torch_dist 转换 + 工作区播种**：slime/rlve 的 `prepare.sh` 在有 GPU 时会就地把 HF 权重转成
  mcore/torch_dist，转换脚本（`$PTDBENCH_WORKSPACE/scripts/models/*.sh`、`tools/convert_hf_to_torch_dist.py`）
  是**任务自带在 `tasks/<t>/workspace/` 里**的，不在镜像里。所以脚本在跑每个 `prepare.sh` 前会把该任务的
  `workspace/` 树 `cp -a` 进临时 `$PTDBENCH_WORKSPACE`（与 harbor materialize 一致），否则 GPU 转换会报
  `scripts/models/*.sh: No such file or directory`。转换结果写在 throwaway 工作区（不进共享缓存），
  CPU-only 会跳过、留给训练首跑兜底。
- **materialize 不能留悬空软链（已修）**：`ptdbench_fetch_hf` 下载进 HF hub 缓存后，用
  `docker/ptdbench_fetch.py` 的 `materialize()` 把快照 `--dest` 到 throwaway 工作区。HF 快照里每个文件
  本身是指向 `blobs/<sha>` 的**相对软链**，而集群 `/user` 盘上 `os.link` **不跟随软链**，于是 dest 落的是
  指向软链自身的硬链——换了目录相对目标就失效，成了**悬空链**。表现是下载明明 `ready`，后面读它的步骤却炸：
  `config.json` 读不出（`Unrecognized model … model_type`）、`tokenizer.json` 读不出
  （`Couldn't instantiate the backend tokenizer`）、taco 的 `tar: No such file`。修复：`materialize()` 在硬链
  前先 `os.path.realpath` 解到真实 blob（跨设备回退到真字节拷贝，`open()` 必跟随软链，不会悬空）。
- **运行时从 clone 安装 fetch/dataset 工具（关键）**：`prepare.sh` 调的 `ptdbench_fetch_hf` /
  `ptdbench_fetch_url` / `ptdbench_dataset` 是 shim，exec 的是 `/usr/local/libexec/*.py`。这些**不再**烤进镜像
  （Dockerfile 只保留 `ptdbench-activate.sh` 钩子和 `/etc/ptdbench-image-path`），所以像上面 materialize 这类
  修复**光 git push 就能生效，无需重建镜像**。`build_entry()` 在 `cd` 进仓库后、跑 prepare 循环前先跑
  `bash docker/install-ptdbench-tooling.sh`（装 clone 当前的 `ptdbench_fetch.py` / `ptdbench_dataset.py` /
  `ptdbench_datasets` 到 `/usr/local`，幂等；文件不在则空操作），让循环里真正 exec 的是仓库最新那份。要让
  已提交的作业生效，重跑 `warm_cache.sh` 即可（代码在 `cyh` 分支，作业会重新 clone）。
- **单任务失败不阻断整批，但作业如实失败**：预热的交付物是下载缓存（`ptdbench_fetch_hf` 在任何后处理之前
  就写好了）。某个 `prepare.sh` 失败时，脚本记一笔、打到 stderr 并**继续**预热后面的任务；但整批跑完若有任何
  失败，作业**以非零码退出**（状态 Failed、日志会归档），而不是误导性的 Succeeded。日志末尾列出
  `prepare.sh failed (downloads cached): …` 方便排查。
- **白名单代理只放行 HF 镜像，pypi.tuna 下不动（已处理）**：`reward_design/task_evaluator` 的 `prepare.sh` 要装
  一个打补丁的 `pyext==0.7`，原本从 `pypi.tuna.tsinghua.edu.cn` 下 sdist——该 host 不在出口白名单里，TLS 握手被
  重置（`SSLEOFError`，job 1108023）。已把那份 sha 钉死的 7772B sdist vendored 进
  `tasks/reward_design/task_evaluator/environment/vendor/pyext-0.7.tar.gz`，`prepare.sh` 优先用它（无外网也能过），
  URL 作兜底，两条路都校 sha。其余 URL/pip 下载若也命中白名单，同法 vendor。
- **search_r1（`slime_search_r1` / `_nomask`）暂缓**：metadata 也绑 `slime-v0.3.2`，但数据量大（~79GB +
  wiki-18 索引）且有 cu130/确定性复现的未决问题，之前 deferred，未纳入 ② 。要预热时单独一条，`--prepare-glob
  'tasks/slime_search_r1/environment/prepare.sh tasks/slime_search_r1_nomask/environment/prepare.sh'`。
- 已缓存约 ~20GB（slime_retool*/rlve/verl 的 prepare 已跑通）；缓存是宝贵数据，**切勿 `rm -rf`**。

### 1b. 本地 docker（单机自测）

本地路径下 `prepare.sh` 由 `harbor-evaluate` / `harbor-shell` / `harbor-stage` 自动触发；
也可以显式指定缓存目录（默认 `<repo>/models`、`<repo>/datasets`）：

```bash
export PTDBENCH_MODELS_DIR=/abs/models
export PTDBENCH_DATASETS_DIR=/abs/datasets
```

> prepare 容器会把仓库 `docker/` 只读挂到 `/ptdbench/tooling`，跑 `prepare.sh` 前先执行
> `install-ptdbench-tooling.sh`——和集群路径一样，用的始终是 clone 当前的 fetch/dataset 工具，
> 改完无需 `build-images --rebuild`（`ptdbench-activate.sh` 仍烤在镜像里，改它才需要重建）。

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
（`harbor-shell` 等价于 `harbor-stage --backend docker`，后者是跨后端的统一入口。）

### 2b. Cybertron devspace 版（真集群、多卡）

一条命令跑到 pre-agent 手动态（建/attach devspace → `install-ptdbench-tooling.sh` →
上传任务 → 在 devspace 上跑 `prepare.sh`），然后**留着 devspace 不停**，打印 `tsh ssh` 入口：

```bash
python3 -m ptdbench harbor-stage tasks/slime_retool --backend cybertron \
  --gpus 8 --image 25607131 \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=wuxi_train \
  --environment-kwarg resource_pool=speech-wx \
  --environment-kwarg billing_account_id=N00069 \
  --run                                   # 不加 --run 即 dry-run，只打印将做什么
# attach 已有 devspace（不新建、不销毁）：--devspace-id tasks/1098539
# 停掉自建的 devspace：--stop（attach 的永不停）
```

命令打印出 `tsh ssh --cluster=… root@<node>` 后，ssh 进去 `cd /workspace` 即是 prepare 好的工作区，
按任务 README / 启动脚本手动起训练、观察日志、复现 bug。用完 `cctl devspace stop <id>` 释放资源。

> harbor-stage 复用后端 `CybertronEnvironment`，一条命令里自动完成建/attach devspace、
> `install-ptdbench-tooling.sh`、上传任务、在 devspace 上跑 `prepare.sh`，无需再手动
> `cctl devspace create` + `tsh ssh` + 导出缓存环境变量那条链路。

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
