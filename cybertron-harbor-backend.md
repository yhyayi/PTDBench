# 在 Cybertron 上跑 harbor：可行性分析与权限申请

> 写给：接手「把 ptdbench/harbor 评测搬到 Cybertron」的同学。
> 记录日期：2026-09-29。探测集群 `wuxi_train` / 资源池 `speech-wx` / 命名空间 `training`。

---

## 0. 结论速览

- **cctl 单独当 harbor backend——不行。** 它是任务提交 CLI，不是容器运行时；缺 harbor
  backend 契约最核心的两个原语：向运行中容器 `exec` 任意命令、向 pod 双向传文件。
- **照搬 harbor 现成的 ACK/GKE（k8s-native）backend——当前也不行。** 因为两条 k8s 路线的
  权限都被平台堵死（见 §3 实测）：
  - 外部没有可用 kubeconfig（`cctl` 无相关命令，`kubeconfigPath` 全是服务端路径）；
  - pod 内 in-cluster SA（`system:serviceaccount:training:default`）RBAC 为空，连建 pod / exec 都不行。
- **可行前提**：向运维申请 **pod exec/管理权限**（§6）。拿到后，一个 `cybertron` backend
  可复用 ACK 的思路实现（§5），但还要解决 prepare 联网 + 缓存经集群文件系统喂进只读挂载
  的问题（§5 两个坑）。

---

## 1. 背景

- 本仓库是 benchmark **PTDBench**：以「复现特定训练失败」为评分，任务横跨三套互不兼容的
  训练栈（verl / 两个版本 slime，CUDA 12.9 / 12.6 / 13.0）。
- 评测框架是 **harbor**（`harbor/`），通过 **environment（backend）** 抽象启动任务，
  当前默认 `docker` backend（本机 docker）。
- 动机：这些是重训练任务，需要 GPU。想用 Cybertron 的 GPU 池来跑，于是问 **cctl 能否作为
  一种启动方式（类似 docker）**。
- 三个镜像已在 Cybertron 构建好并注册（平台数字镜像 id）：
  - `ptdbench-verl-vllm017`（cu129）
  - `ptdbench-slime-rc0`（cu126）= 仓库里的 `slime-rc0`，平台 id `25607122`
  - `ptdbench-slime`（v0.3.2）= 仓库里的 `slime-v0.3.2`，已重建到 cu129，平台 id `25607131`
    - ⚠️ 用户口头说的 `ptdbench-slime-v0.3.1` 与仓库 `slime-v0.3.2` 对不上，需核对。

---

## 2. harbor 对一个 backend 的契约

抽象接口在 `harbor/src/harbor/environments/base.py`，`@abstractmethod`：

| 方法 | 作用 |
|---|---|
| `type()` | backend 标识 |
| `_validate_definition()` | 校验任务定义 |
| `start(force_build)` | 起一个 **长驻 keepalive 容器**（预构建镜像 + GPU） |
| `exec(...)` | **反复**在容器里跑命令，拿 stdout/stderr/exit code（热路径） |
| `upload_file/dir` `download_file/dir` | 本地 ↔ 容器双向传文件 |
| `stop(delete)` | 拆除 |

- docker backend（`harbor/src/harbor/environments/docker/docker.py`）：`docker compose up --wait`
  + `docker compose exec` + `docker cp`。
- ACK backend（`harbor/src/harbor/environments/ack.py`，~2500 行）：k8s Python client
  `connect_get_namespaced_pod_exec` 做 stream exec，`tar` over exec 传文件。→ **cybertron
  可直接参考这份**。

已注册的 backend 类型见 `harbor/src/harbor/environments/models/environment_type.py`
（docker/e2b/modal/gke/ack/ec2/openshift/skypilot…），加新 backend = 加一个枚举 +
factory 注册 + 一个实现类。

---

## 3. 权限实测（关键）

`cctl` 没法 exec，所以用「把探测脚本当 `--entry`，跑完读 `cctl job logs`」验证。

### 3.1 外部是否有可用 kubeconfig
- `cctl` 全部命令里 **没有** `exec`/`attach`/`ssh`/`cp`/`kubeconfig`/`port-forward`。
  用户跟 devspace 交互只能靠 `expose`（应用层端口）。
- `cctl cluster get` 的 `kubeconfigPath` 全是控制面服务端路径（`/etc/kubeconfig/*.yaml`），
  不发给用户。
- → **外部可用 kubeconfig：没有。**

### 3.2 pod 内 in-cluster SA 能否建 pod / exec
探测 job（`tasks/1097135`，ptdbench-slime-rc0 镜像，namespace=`training`）内对 k8s API 做
SelfSubjectAccessReview：

| 项 | 结果 |
|---|---|
| 身份 | `system:serviceaccount:training:default` |
| pods/exec `create` | **false** |
| pods `create` / `get` / `list` / `delete` | **全 false** |
| pods/log `get` | **false** |
| services `create` / configmaps `create` | **false** |

in-cluster API 可达、kubectl 未预装，但这个 default SA **零 RBAC**。
→ **pod 内建 pod/exec：没有。**

### 3.3 附带发现
- **CUDA 13.0 驱动不匹配**：`ptdbench-slime`（v0.3.2/cu130）在 wuxi_train 节点起不来，
  报 `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0`；连
  `--gpu 0` 都会被 nvidia hook 拦。`ptdbench-slime-rc0`（cu126）能起。→ slime v0.3.2 那批
  任务（`slime_retool*`、`slime_search_r1*`）得换驱动够新的集群/池。
- `cctl job create --image` 要用**平台数字镜像 id**（如 `25607122`），不是原始 registry URL。
- `cctl job create` 需要 `--billing-account-id N00069`。
- 短 job 的 stdout 靠 `cctl job logs` 读，成功后要 ~30–60s 才归档。

---

## 4. 任务执行生命周期（agent 在 pod 内）

核实自 `harbor/src/harbor/trial/trial.py` 的 `run()` 与 `harbor/src/harbor/agents/installed/`。

- **跑在 pod 内的是 agent（codex/claude-code 等）和 verifier；harbor 控制器跑在 pod 外**，
  靠 `exec`+`cp` 反复伸进 pod。**要申请的权限正是给 pod 外控制器用的。**

**阶段 0 · prepare（harbor 之前、容器外）**
`environment/prepare.sh` 由 ptdbench 在 `harbor run` 之前单独跑，是**唯一联网、唯一写共享
缓存**的阶段（下模型/数据、写 `prepared/env`）。产物塞进 harbor task 的 `environment/`，
后续由 harbor 上传到 `/workspace`。到 harbor 这步 agent/verifier 已离线。

**阶段 1 · `_prepare()` 起 pod + 装 agent**
1. `environment.start()`：从任务镜像起 keepalive 长驻 pod（隔离边界）。
2. 上传 `/workspace`；`prepared/models/datasets/environment` **只读**挂进去。
3. `agent.install()`：exec 进 pod 装 nodejs/npm/curl + agent CLI（如 `npm install -g @openai/codex`）。

**阶段 2 · `_run()` agent 在 pod 内自主跑**
4. `agent.run()` = 一次长阻塞 `environment.exec`：`codex exec --dangerously-bypass-approvals-and-sandbox …`
   （容器本身即沙盒）。codex 读说明、改 `/workspace`、本地执行命令；LLM base_url/key 经 env 传入。

**阶段 3 · verifier**
5. 同一 pod（shared 模式）exec `verifier/test.sh`，输出结构化结果。

**阶段 4 · `_finalize()` 拉结果 + 销毁**
6. `download_*` 把 agent 轨迹 / verifier 结果 / artifacts 拉回宿主。
7. `environment.stop(delete)` 销毁 pod。

并发（`--n-concurrent`）= 同时起 N 个这样的 pod。

**backend 契约 → k8s 层动作 → 所需权限**

| harbor 调用 | k8s 动作 | 权限 |
|---|---|---|
| `start()` | 建 pod | pods **create** |
| `exec()`（装 agent / 跑 codex / 跑 verifier，多次） | `connect_get_namespaced_pod_exec` | **pods/exec create** ← 核心 |
| `upload/download` | tar over exec | pods/exec（同上） |
| `stop(delete)` | 删 pod | pods **delete** |
| healthcheck/状态 | get/list pod | pods **get/list** |

---

## 5. 若拿到权限：cybertron backend 设计要点

- 新增 `EnvironmentType.CYBERTRON` + factory 注册 + `CybertronEnvironment`（从 `ack.py` fork）。
- `exec/upload/download` 走 k8s API（`connect_get_namespaced_pod_exec` + tar），与 ACK 一致。
- `start/stop/镜像/配额/认证` 用 cctl。两种子形态：
  - **(A) cctl 建 pod + k8s exec**：`cctl job/devspace create` 起 keepalive 任务（尊重平台调度/
    配额/池），再对该 pod 用 k8s API exec/cp。平台集成最干净。
  - **(B) 纯 k8s（ACK 式）**：连建 pod 都用 kubeconfig。代码复用最大，但绕过 cybertron 调度语义。
- 镜像名映射：harbor `environment.image`（`ptdbench/<name>:latest`，见
  `ptdbench/core/types.py` 的 `EnvironmentImage.tag`）→ cybertron 数字镜像 id。

**两个必须解决的坑：**
1. **prepare 联网 + 缓存喂进只读挂载**：agent/verifier 离线没问题，但 prepare 要联网、产物要落到
   一个后续 pod 能只读读到的地方。cybertron 无宿主 bind mount → prepare 产物和 models/datasets
   缓存要放**集群文件系统**（用 `cctl file-task` 同步），建 pod 时挂进去。移植改动最大处。
2. **只读挂载形态**：docker 靠 host bind mount 给 `/ptdbench/{prepared,models,datasets}`；
   cybertron 要换成 PVC / 集群 fs volume，backend 建 pod 时声明。

---

## 5.5 备选路线：应用层 exec 通道（不需要 RBAC）

§3 的死结是「拿不到 k8s exec」。但 exec 的本质只是「有通道把命令送进运行中的容器、把输出拿回来」，
`kubectl exec`（走 k8s API server）只是其中一种。**换一条通道即可绕开整个 RBAC 问题**：在容器里自跑一个
监听端口的 **exec-agent**，用 cctl 的 `expose` 把端口引出来，harbor 从网络侧驱动它。

> 关键认知：这里的「执行任意命令」不是 cctl / k8s 给的，是**你烤进镜像的 agent** 给的；cctl/expose 只负责
> 起机器 + 开端口。因此**不用向运维申请 `pods/exec`**。

链路：`harbor 控制器 --HTTP/ws--> expose 出来的 URL --> 容器内 exec-agent --> 在容器内跑命令`。
harbor 的 `start/stop/GPU/配额/计费` 仍走 cctl，只有 `exec/upload/download` 改走这个 agent。
等价于把 §2 契约里 exec 的**传输层**从「k8s exec」换成「app 层网络」，保留 `start→exec→exec→stop` 生命周期。

**通用改动**
- 三个镜像（verl/slime-rc0/slime-v0.3.2）各烤进一个小 exec-agent：HTTP/websocket server，接口 = 「跑命令
  → 回 stdout/stderr/exit code」+「收/发文件」，单端口多路复用，**必须带 token 鉴权**（暴露 URL 可被他人访问，
  否则等于开放 GPU 任意执行）。
- `CybertronEnvironment.exec/upload/download` → 对 URL 发 HTTP；`start/stop` → cctl。
- 并发 `--n-concurrent=N` = 起 N 个容器 + N 个 URL。
- prepare 联网 + 数据只读挂载那两个坑（§5）**不变**：仍要 `cctl file-task` 预热 + PVC 挂载。

### 形态 (C) BATCH job + expose（推荐）
```bash
cctl job create --image <id> --gpu N --resource-pool speech-wx \
  --billing-account-id N00069 --expose-port 8666 \
  --entry "start-exec-agent && sleep infinity"   # 长驻：起 agent 后永远阻塞
```
- **优点**：无生命周期负担。batch 跑 `--entry` 到结束，只要 entry 永不退出就是长驻容器；没有心跳、没有到点快照。
- start=`cctl job create --expose-port`；stop=`cctl job stop`；exec/cp 走 agent URL。

### 形态 (D) devspace + expose
```bash
cctl devspace create --image <id> --gpu N --resource-pool speech-wx --expose-port 8666
cctl devspace expose <id> -q   # 取 URL
```
- 语义上更贴「交互式长驻」，且带 snapshot/restore（§4.6）。
- **坑**：devspace 有 deadline，需**持续心跳**续期（`POST .../:heartbeat`，**cctl 无子命令**，得自己打 HTTP）；
  **申请了 GPU 的 devspace 到 deadline 会被自动快照并释放**——harbor 长跑中途会被回收。控制器必须为每个
  devspace 打心跳。除非确实要 snapshot/restore，否则 (C) 更省心。

**(C) vs (D)**：默认选 **(C) BATCH**，规避心跳与到点回收；只有需要 devspace 的 restore 语义时才用 (D)。

**仍需实测/确认的点**
1. expose 出的 URL 从 harbor 控制器所在网络**是否可达**（平台 ingress 级，未证实）。
2. 单任务基本「一个端口」→ exec 与传文件复用同一端口（agent 多路复用解决）。
3. agent 端口鉴权方案（token 经 `--env` 注入）。

**与 §6（申请 RBAC）的取舍**：本路线不求人、可立即自证，代价是每个镜像加一层 agent + 自管传输；§6 路线
代码最省（直接 fork ack.py），但依赖运维授权。建议：先按 (C) 做可行性验证，同时并行走 §6 申请。

---

## 5.6 文件挂载与隔离：现状调研与实现方案

拿到 exec 通道（§5.5）后，剩下的缺口就是「文件怎么进 pod、并发怎么互不干扰」。调研结论：
**隔离几乎白送，真正的硬骨头只有「大的、共享的、只读的 models/datasets 缓存怎么挂进 pod」。**

### 5.6.1 harbor 已有的两套文件模型（关键先验）

harbor 的 `BaseEnvironment` 有个 `capabilities.mounted` 开关，两个后端走两条完全不同的路：

| | docker 后端（`mounted=True`） | **ACK 后端（`mounted=False`）** |
|---|---|---|
| 文件进出 | compose bind mount（运行时生成 `ServiceVolumeConfig` 列表） | **完全不挂载，全部 tar-over-exec 上传/下载** |
| `/logs/*` | host bind mount RW | `mkdir -p` + `chmod 777`，结束时 `download_dir` 拉回 |
| `/workspace`、`/ptdbench/*` | 由 **task 级** `[environment].mounts` 声明的 bind | 只能靠 `pod_overrides` 注入 PVC/hostPath（harbor 不默认声明） |

> 重要纠正：`/ptdbench/{prepared,models,datasets}`、`/workspace` **不是 harbor 内置**，是
> ptdbench adapter 通过 task.toml `[environment].mounts` 塞进去的（docker 后端）。
> **ACK 后端根本不挂任何卷**——这正是 cybertron 该抄的样板：cctl 无挂载 flag（见 §5.6.3），
> 所以 cybertron backend 应设 `mounted=False`，走 ACK 的 tar-over-exec 模型。

### 5.6.2 ptdbench 的文件布局与体量（决定策略）

per-stage 挂载规则（`ptdbench/runtimes/providers/docker.py`）+ 缓存模型（`evaluation/evaluator.py`）：

| 目录 | prepare 阶段 | agent/verifier 阶段 | 共享性 | 体量 |
|---|---|---|---|---|
| `/workspace` | RW | RW(agent)/RO(verifier) | **per-run**（每 trial 独立） | 小～中（多数任务几 GB） |
| `/ptdbench/prepared` | RW | **RO** | per-run | 小（pysite、env 文件） |
| `/ptdbench/models` | RW | **RO** | **跨 task/run 共享** | 大 |
| `/ptdbench/datasets` | RW | **RO** | **跨 task/run 共享** | **很大** |

体量实测：整套 benchmark 缓存约 **110–120GB**，其中 **search_r1 的 wiki-18 索引一家就 ~90GB**
（`e5_Flat.index` 60GiB + `wiki-18.jsonl` 14GB，首次构建峰值再 +20GB）；其余任务各几 GB
（qwen3-4b ~9GB、qwen2.5-3b ~6GB、llama-1b ~2.5GB、若干 0.5b ~1GB）。缓存按
`repo+revision` 的 HF-hub 布局 key，跨 task 命中不重下；prepare 时**硬链接**进 workspace
（不是软链，因为 harbor 会把 workspace 打包上传，软链会 dangle）。

### 5.6.3 cctl 挂载能力实测：没有

`cctl {job,pytorchjob,notebook,devspace} create --help` 全部 flag 已核：数据进容器只有
`--code-type git --git-path`（拉 git 仓库）、`--env`、`--image`。**没有
`--volume/--mount/--pvc/--filesystem`**；`-f` 的 Task JSON schema（kind/project/cluster/
image/resources/entry/envVars/gpuModel…）**也没有 volumes 字段**。→ 用户侧无任何声明式挂载入口。

但 wuxi_train 集群**有** file 型文件系统：`wuxi_train`(id 218, 1.2 PiB)、`wuxi_gpfs`(223)、
`uds-shared-wx`(388)。

**✅ 已探测证实（2026-09-29，job `tasks/1097565`，1×H100 node gpu24）：**
Cybertron **自动把资源池的集群文件系统挂进每个 pod 的 `/user`**。
- `/user` = `drwxrwxrwx`、199 个条目、owner `nobody:nogroup`，就是共享 fs（job 响应里的
  `filesystemId: 218` = `wuxi_train`）；下面是按用户名（拼音）分的 home 目录
  （`anhongran/`、`baibingsong/`… 多为 world-writable）。本人目录约为 `/user/chenyinghao`。
- 日志落盘路径 `/wuxi_train/training/projects/3067-<proj>/<jobid>/logs` 表明 fs 也可按
  `wuxi_train:/...` / projects 结构寻址（与 cctl URI、`cctl file-task` 一致）。
- pod 内 `uid=0(root)`。

**⚠️ 同时探测出的硬约束（另一个失败 job `tasks/1097517`）：** 0-GPU pod 起 rc0 直接报
`临时存储超出限制，请将数据保存在 /user 等共享文件存储中` —— **容器自身 ephemeral 存储配额很小，
一切大数据（models/datasets/checkpoints/workspace 大文件）必须落在 `/user`，不能写容器 rootfs。**
这正好和 §5.6.4 的分流一致，且是平台官方指引。

### 5.6.4 两个缺口分别怎么填

**A. 文件隔离——几乎白送。** 用 ACK 式 `mounted=False`：每个 trial 一个独立 pod，`/workspace`
就是该 pod 自己的文件系统、由 tar 上传填充，**天然隔离**，并发不会撞。共享的 models/datasets 是
**只读**，多 pod 共读同一份完全安全（k8s 层就是 PVC `ReadOnlyMany` / 只读挂载）。→ 无需额外设计。

**B. 文件挂载——按体量分流，这才是要干的活：**

- **小/中、per-run、可写（`/workspace`、`/ptdbench/prepared`）**：走 §5.5 的 exec 通道
  tar 上传，和 ACK 的 `upload_dir` 一模一样。够用。
- **大、共享、只读（`/ptdbench/models`、`/ptdbench/datasets`，最多 ~90GB）**：**绝不能每个 pod
  拷一份**，必须真挂载集群 fs。这就是 §5 坑 1/2 的实质。三条子路：
  1. **✅(最优，已证实可行) 资源池 fs 自动挂载于 `/user`**：把 110GB 缓存**一次性**预热到
     `/user/chenyinghao/ptdbench-cache/{models,datasets}`（见下），所有评测 pod 直接读同一路径；
     只读性与隔离靠「约定不往那写」（同一份被多 pod 只读共享本就安全）。**无需求平台任何额外权限。**
  2. Task JSON / 平台侧声明 PVC：cctl 不给，需走管控台或 v2 API——**既然子路 1 已通，不必走这条。**
  3. 塞进镜像——否决（太大、随 task 变）。

**缓存预热怎么做（解决 §5「prepare 联网 + 落集群 fs」坑）：** 不要从笔记本上传 110GB。而是
起一个**一次性 BATCH job（`--code-type git` 拉本仓库、联网）**，在 pod 内跑全部 `prepare.sh`，
把 `PTDBENCH_MODELS`/`PTDBENCH_DATASETS` 指到 **`/user/chenyinghao/ptdbench-cache/{models,datasets}`**
（`/user` 是自动挂载的持久 fs，写这里才不撞 ephemeral 限制）。之后所有评测 pod 读同一路径。
跨集群/池搬运用 `cctl file-task sync`。⚠️ search_r1(cu130) 还卡在
[[cybertron-cuda13-driver-mismatch]]，其 90GB 预热得等换驱动够新的集群。

### 5.6.5 落到 CybertronEnvironment 的改动

- `capabilities = EnvironmentCapabilities(mounted=False)`（照 ACK）。
- `upload/download` → §5.5 exec-agent 的 tar 通道；只喂 `/workspace` + `prepared`。
- models/datasets **不走 upload**：backend 起 pod 时让其读集群 fs 的固定/约定路径；把 ptdbench
  期望的 `/ptdbench/models`、`/ptdbench/datasets` 用容器内 `ln -s`（或 entry 里 bind）指到那个
  只读路径。ptdbench 硬链接进 workspace 的那部分权重仍随 `/workspace` 上传（多数任务几 GB，可接受）。
- 若走子路 1，需先确认自动挂载路径（§7 探测）；若走子路 2，需平台开 PVC 声明。

### 5.6.6 把 agent 关进 /workspace（隐藏 /user）

问题：`/user` 是**全集群共享盘**自动挂进每个 pod，agent（root）默认能看/改整个 `/user`（含你的家目录
和别人 777 的目录）。要让 agent **只看到 `/workspace`**、看不到 `/user/chenyinghao`，且 models/datasets
以只读出现在 `/workspace` 下。

原设想：pod 内是 root，用 **private mount namespace** 给 agent 套受限视图（bind 模型进 workspace +
tmpfs 盖 `/user`）。**❌ 已探测证否，此路不通。**

**❌ 挂载类隔离已探测排除（2026-09-29，jobs `tasks/1097839` + `tasks/1097871`，node gpu10）：**
- 容器**有 CAP_SYS_ADMIN**（甚至 userns 内 CapEff=全 `000001ffffffffff`），`unshare/mount/nsenter/capsh`
  都在（`bwrap` 未装）。
- 但 **`mount` / `unshare(CLONE_NEWNS)` 被拒**：`unshare -m`、`unshare -U --map-root-user --mount`、
  `unshare -m --propagation unchanged`+`mount -t tmpfs` **全部 `Permission denied`/EPERM**。根挂载被锁 +
  seccomp 过滤，**跟 cap 无关**。→ **mount namespace / bind / tmpfs 方案不可用，bubblewrap 同理没用**
  （它做的是同样的 syscall）。
- `df`：容器 `/` overlay ~134G 空闲（但真实 per-pod ephemeral limit 更小，0-GPU pod 曾因此失败）；
  `/user` = `thuwayfs` 10 PB 共享盘。

**✅ 可行的替代（不需要 mount）：**
1. **模型/数据集进 `/workspace`：用符号链接**（bind 不可用）：`ln -s /user/chenyinghao/ptdbench-cache/models
   /workspace/models`。物理仍在 `/user`，但出现在 `/workspace/models`。（ptdbench 现用硬链接；这里免上传，软链即可。）
2. **隐藏你自己的 `/user/chenyinghao`：靠文件权限**——agent 以**专用非 root uid** 跑，把家目录设 `711`
   （不可列目录、仅可穿到已知子路径）、敏感子目录 `700`、只把 cache 子目录开 `755`。这样 agent 读得到
   给它的 cache 路径，但 `ls /user/chenyinghao` 看不到你其余文件。
   - **注意**：⑴ 只挡住**你的**目录，**整个 `/user` 仍挂着**，别人 world-readable 的目录 agent 仍能看到——
     全盘隐藏做不到；⑵ agent 需改成非 root 跑，要验证训练/GPU 在非 root 下正常（RL 任务要跑 GPU）。
3. **平台侧（最干净，需求人）**：请平台给 harbor pod 提供「挂 `/user` 的**子路径**」或「不挂 `/user`」或
   放开 mount syscall 之一。任一即可回到 §5.6.6 原设想的强隔离。

结论：**「agent 只看 `/workspace`、完全看不到 `/user`」目前做不到**；能做到的是「模型软链进 `/workspace`
+ 用权限挡住你自己的目录」，全盘隔离要么等平台支持，要么接受弱隔离。

---

## 6. 向运维申请权限（可直接转发）

> 用途：在 Cybertron 上跑 harbor 评测。控制进程在 pod 外，需反复 `exec` 进长驻 pod
> 装 agent、跑 rollout、跑 verifier，并经 exec 通道传文件。当前 `training:default` SA
> RBAC 为空（`pods`/`pods/exec`/`pods/log` 全 `allowed:false`）。

**二选一（哪种符合安全策略都行）：**
- **方案 A（推荐）**：给一个按命名空间收敛的**外部 kubeconfig**。集群 `wuxi_train`（或建议的 H100
  池所在集群）；命名空间建议**单开一个 `harbor`**，避免和训练任务混。
- **方案 B**：命名空间内建**专用 SA**（`harbor-runner`，别改 `default`）绑同一 Role，harbor 控制器
  也跑在集群内。

最小 RBAC（传文件走 exec，不需额外资源）：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: harbor-runner
  namespace: training        # 或单开的 harbor 命名空间
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["create"]         # ← 最核心
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: harbor-runner
  namespace: training
subjects:
  - kind: ServiceAccount     # 方案 A 换成签发 kubeconfig 的 user/group
    name: harbor-runner
    namespace: training
roleRef:
  kind: Role
  name: harbor-runner
  apiGroup: rbac.authorization.k8s.io
```

**顺带确认**：prepare 阶段要联网下模型/数据，产物需落到某**集群文件系统**再只读挂进 pod——
该池/命名空间能否挂共享 fs（PVC 或你们的 fs），读写路径怎么申请。

**归属**：项目 `<填>`，billing account `N00069`（MiniCPM 5.0）。只要命名空间级，不要集群级。

**收敛提示**（内部备注，不用发）：真正不能少的是 `pods/exec: create` 和 `pods: create/delete`；
`list/watch` 可砍但留着排障方便。别申请 ClusterRole。

---

## 7. 待确认 / 下一步

1. 运维能否给 §6 的权限（A 或 B）？→ 决定 backend 走 k8s-native 还是必须换架构。
2. slime v0.3.2（cu130）要跑在哪个驱动够新的集群？wuxi 不行。
3. ~~集群文件系统挂载~~ **✅ 已解决**：pod 内 `/user` 自动挂载集群 fs（fs 218 wuxi_train），
   走 §5.6.4 子路 1。剩下的仅是 prepare 缓存预热的**执行**：起一次性 BATCH job 跑全部
   `prepare.sh` 写到 `/user/chenyinghao/ptdbench-cache`。（speech-wx 已确认是 **H100** 池。）
4. 核对镜像名 `ptdbench-slime-v0.3.1` vs 仓库 `slime-v0.3.2`。
5. 若拿到权限，据 §5 展开 `cybertron` backend 的实现计划（volume/挂载声明、prepare 走
   `cctl file-task` 预热、exec/cp 的两种权限形态分支）。
6. **§5.6.6 的决定性未知**：Cybertron pod 给不给挂载特权？→ 一个 GPU 探测 job 依次试
   `unshare -m`、`mount --bind`、`mount -t tmpfs`、`unshare -U`（capsh --print 看 cap），
   读 logs 定论。能 mount → agent 隔离走 mount namespace 方案；否则退 bubblewrap / 弱隔离。
7. ephemeral 余量未测：GPU pod 上 `df -h / ; df -h /user`，定 `/workspace` 能放多大（决定
   §5.6.4 情况 1/2 分界）。可与第 6 项合并到同一个探测 job。

---

## 8. 方案 E：devspace + SSH 作为 harbor backend（推荐主线）

> 记录日期：2026-09-29。本节把 §5.5（BATCH + 自烤 exec-agent）升级为一条**不用自烤 agent、
> 不用求 RBAC**的主线：**用 devspace 当长驻主机、用它自带的 SSH 通道当 exec/传输层**。
> 对应用户诉求：「持续 exec 用 devspace 实现，文件隔离用 bubblewrap 实现」。

### 8.0 关键新认知：devspace 自带真 SSH 通道

§3/§5.5 的死结是「`cctl` 没有 exec」。但 **devspace 不一样**：Cursor 的 Cybertron/Teleport
集成会在 `~/.ssh/config` 里生成 `Host devspace-<id>` 条目，于是可以直接
`ssh devspace-<id> "<cmd>"`、`rsync`/`scp` 传文件（见 `devspace-driver` skill 的
`scripts/devspace`：`ssh_exec`、`run`、`sync/pull` 全走标准 ssh+rsync）。

→ **这正是 harbor backend 契约缺的两个原语（反复 exec + 双向传文件），devspace 白送。**
不用像 §5.5 那样往镜像里烤 HTTP exec-agent，也不用向运维申请 `pods/exec`（§6）。

### 8.1 直接照抄的样板：EC2 backend（不是 ACK）

harbor 已有一个**纯 SSH** 的 backend：`harbor/src/harbor/environments/ec2.py`。它的
`exec/upload/download` 全部走 `ssh` 子进程，与 devspace 完全同构：

| harbor 原语 | EC2 的实现 | devspace backend 照抄 |
|---|---|---|
| `exec()` | `_ssh_exec` → `_ssh_base_cmd`（`ssh dest "cmd"`） | `ssh devspace-<id> "cmd"` |
| `upload_file/dir` | tar 打包 → `ssh "tar xf - -C <dir>"`（`pack_dir_to_bytes`） | 同左，或 `rsync` |
| `download_file/dir` | 反向 tar-over-ssh（`tar_transfer`） | 同左 |
| 就绪等待 | `_wait_for_ssh()` 轮询 | 轮询 `ssh ... true` |
| `capabilities.mounted` | `False`（无 host bind，全 tar 传） | `False`（同 ACK/EC2） |

→ **新 backend = fork `ec2.py`**，把「起 EC2 实例 + 拿 host/key」换成「`cctl devspace create`
+ 解析 Teleport host `devspace-<id>`」。exec/传输/tar 那套一行不用改。这比 §5 fork 2500 行的
`ack.py` 省得多。

### 8.2 契约映射（backend 骨架）

新增 `EnvironmentType.CYBERTRON = "cybertron"`（`harbor/src/harbor/models/environment_type.py`）
+ 在 `factory.py` 的 `_ENVIRONMENT_REGISTRY` 注册 `_EnvEntry("harbor.environments.cybertron",
"CybertronEnvironment", None)` + 实现类 `CybertronEnvironment(fork ec2)`：

| 方法 | 实现 |
|---|---|
| `start(force_build)` | `cctl devspace create --image <数字id> --gpu N --resource-pool speech-wx --billing-account-id N00069`；轮询 `cctl devspace get <id> -F status` 到 `Running`；从 `~/.ssh/config` 解析 `devspace-<id>`（照 `devspace-driver` 的 `host_for_id`）；`_wait_for_ssh`；**起心跳线程**（§8.3）。 |
| `exec(...)` | `ssh devspace-<id> "cmd"`（EC2 的 `_ssh_exec`）。agent/verifier 若要隔离，命令包一层 `bwrap ...`（§8.4）。 |
| `upload/download` | tar-over-ssh（EC2 原样）或 rsync。只喂 `/workspace` + `prepared`；models/datasets 走 §5.6 的 `/user` 软链，不上传。 |
| `stop(delete)` | `cctl devspace stop <id>`（+ 停心跳）。 |
| 健康检查 | `cctl devspace get -F status` + `ssh ... true`。 |

镜像名映射同 §5：harbor `ptdbench/<name>:latest` → cybertron 数字镜像 id（verl=vllm017 /
slime-rc0=25607122 / slime-v0.3.2=25607131）。

### 8.3 devspace 的三个生命周期坑（backend 必须处理）

1. **心跳续期**：devspace 有 deadline，需持续 `POST .../:heartbeat` 续命；**cctl 无该子命令**，
   backend 得自己打 HTTP（或靠 SSH 周期性活动是否足够续期——**待实测**）。控制器为每个 devspace
   起一个后台心跳。
2. **GPU devspace 到点自动快照并释放**：申请了 GPU 的 devspace 到 deadline 会被平台快照+回收，
   harbor 长跑中途会被抽走。→ 心跳必须可靠；或评估 restore 语义能否让 backend 自愈。
3. **Teleport SSH 依赖**：`ssh devspace-<id>` 依赖 **控制器所在机器** 有 `cctl` 登录态 +
   Teleport 签发的 SSH 证书 + `~/.ssh/config` 里的 `Host devspace-<id>`。
   → **控制器不能跑在无 Teleport 的匿名机器**；证书会过期，长跑要考虑 `tsh login` 续签。
   这条替换了 §6 的 RBAC 依赖——不求运维，但依赖本机 Teleport 登录态。

> 与 §5.5 形态 (C)(BATCH) 的取舍：BATCH 无心跳/无到点回收，但要自烤 exec-agent；devspace 免烤
> agent（SSH 白送）但要管心跳。**若心跳可稳定维持，devspace 主线更干净**；否则回退 (C)。

### 8.4 文件隔离：bubblewrap（目标形态，但有硬前提）

**动机**：devspace 是**长驻单机**。并发跑 N 个 trial 时，若共用一台 devspace 的文件系统，各 trial
会互相看见/污染；且跑 `codex/claude-code --dangerously-bypass-approvals` 的 agent 是 root，
默认能看/改整个自动挂载的 `/user` 共享盘（含你和别人的家目录，§5.6.6）。bubblewrap 的作用 =
**给每个 trial 的 agent 套一个受限文件视图**：只暴露该 trial 的 `/workspace` + 只读的
models/datasets，用 tmpfs/空目录盖住 `/user` 与别的 trial。

理想链路：`exec()` 把 agent 命令包成
```
bwrap --unshare-all --die-with-parent \
  --bind /path/to/trialN/workspace /workspace \
  --ro-bind /user/chenyinghao/ptdbench-cache /ptdbench-cache \
  --tmpfs /user --proc /proc --dev /dev \
  -- codex exec --dangerously-bypass-approvals-and-sandbox ...
```

**❌ 已在真 devspace 实测证否（2026-09-29，devspace `tasks/711108`，H100，cluster
`sichuan_train`）：bubblewrap 在 devspace 上用不了。** 证据链：

| 探测 | 结果 |
|---|---|
| 镜像里有 `bwrap`？ | **无**，且装不上（whitelist-only 代理 `whitelist-proxy.cybertron.svc`，aliyun/ubuntu 源 404/502） |
| `unshare -Ur`（user ns） | ✅ **能**（`unprivileged_userns_clone=1`）——**与 batch pod 不同**（batch pod 连 userns 都禁） |
| `unshare(CLONE_NEWNS)`（mount ns） | ✅ 能 |
| `mount(/ , MS_REC\|MS_PRIVATE)` = **bwrap 的强制第一步** | ❌ **EPERM**，即使在已成功的 userns 里以 mapped-root 身份 |
| `mount(tmpfs)` / `mount(--bind)` / `MS_REC\|MS_SLAVE` | ❌ 全 **EPERM** |

用 gcc（devspace 自带）编译一个复刻 bwrap `setup_newroot` 系统调用序列的 C 程序、在 `unshare -Ur`
里跑，**每个 `mount()` 都 EPERM**。

**根因已定位——是 AppArmor，不是 seccomp、不是缺 cap：** 进程跑在 profile
**`cri-containerd.apparmor.d (enforce)`**（containerd 默认）下，它 **deny `mount` 操作**；同时
`Seccomp: 0`（无 seccomp 过滤）、userns 里有 CAP_SYS_ADMIN。所以 bwrap 在第一步
「make / private」就死，跟权限无关。harbor 的 Singularity/Apptainer backend（靠 mount/overlay）**同理挂**。

> ⚠️ 修正 §5.6.6 / [[cybertron-user-fs-mount]] 的旧结论：拦截点在 devspace 上**变了**——userns 现在放开了，
> 但 AppArmor 挡住所有 mount。对 bubblewrap 的净结果不变：**用不了**。

**要真正启用 bwrap → 只能靠平台**：给 devspace pod 设 AppArmor `unconfined`（pod annotation
`container.apparmor.security.beta.kubernetes.io/<c>: unconfined`）或一个放开 `mount` 的自定义
profile。这是平台级申请，与 §5.6.6「subpath 挂载 / 不挂 `/user`」同类。

### 8.5 隔离用 **Landlock LSM**（✅ 已在 devspace 实测可行，推荐主线）

bwrap 挂在「要 mount」上；**Landlock 根本不用 mount / 不用 namespace / 不用任何特权**——它是
**进程自我约束**的 LSM：进程声明「我今后只允许访问这些路径」，`landlock_restrict_self` 后规则外的
文件访问一律 EACCES，被所有子进程继承、不可撤销。正好绕开 §8.4 的 AppArmor mount 封锁。

**✅ 已在真 devspace 实测（2026-09-29，`tasks/711108`，kernel 5.15）：**

| 探测 | 结果 |
|---|---|
| Landlock ABI 版本 | **v1 可用**（5.15 内核；ABI v1 自 5.13 起） |
| `landlock_restrict_self` | ✅ 成功，**root 与非 root 都行**（无需 CAP/mount/userns） |
| 只放行 `/tmp`、拒 `/user` `/etc` | 受限后 `/tmp` 可开；`/user`、`/etc/hostname` → **Permission denied** |
| 放行 `/usr /lib` 后 `execvp` | ✅ 受限进程能正常 exec 真二进制 |
| 放行 `/dev` → `open(/dev/nvidiactl)` | ✅ OK；不放行 `/dev` → **denied** ← GPU 设备节点就是普通文件 open，加进 allowlist 即可 |
| 放行系统盘+`/dev`、拒 `/user`，跑 `nvidia-smi` | 进程正常执行（本 devspace 当前无 GPU 节点 `/dev/nvidia0`，bare `nvidia-smi` 也是 "No devices found"，与 Landlock 无关） |

> 机制：`landlock_create_ruleset`（handled_access_fs=v1 全 13 位）→ 对每个允许目录
> `landlock_add_rule(PATH_BENEATH, O_PATH fd, 权限位)` → `prctl(PR_SET_NO_NEW_PRIVS,1)` →
> `landlock_restrict_self`。raw syscall（444/445/446）即可，devspace 自带 gcc 可现编。

**落到 backend：** `exec()` 把 **agent 命令**（及其派生的训练/rollout）包一层 `llrun`——一个 ~60 行 C
启动器（编译进镜像或 setup 时现编）：建 ruleset → 放行「系统只读盘 `/usr /lib /lib64 /bin /sbin /etc /opt
/proc /sys /run`（RO+exec）+ `/dev /tmp /dev/shm`（RW，含 GPU 节点）+ 本 trial 的 `/workspace` +
只读的 models/datasets cache 路径」→ **不放行 `/user`（及别的 trial 的 workspace）** → restrict → exec agent。
于是 agent 读不到你和别人在 `/user` 的数据，但 GPU、训练、cache 全通。

**Landlock 恢复了「一台 devspace 跑 N 个隔离 trial」的并发模型**（bwrap 本想给的）：每个 agent 在自己的
Landlock 规则里，互相看不到对方 workspace，共享 RO cache 安全。→ **省 GPU/省 devspace 数，不必「每
trial 一台 devspace」。**

**Landlock v1 的边界（诚实说明）：**
- 只管**文件路径访问**，不隐藏路径存在性（`ls /` 仍见 `/user` 名字，但 `readdir`/`open` 其内容 →
  EACCES）。安全属性是「拒绝访问」而非「不可见」，对「别让 agent 读别人数据」已足够。
- **不做路径重映射**（无 bind）→ cache 出现在 `/ptdbench/models` 仍靠 §5.6.6 的 `ln -s`；Landlock 只管
  open 成不成。
- 不管 ioctl（v1）、不管网络（v1；网络管控是 ABI v4/6.7，本内核没有）、不管 `/proc` 里看别的进程
  （要进程隔离另配 PID ns——userns 在 devspace 可用，`unshare -Up` 可考虑）。
- restrict 前已打开的 fd 不受限 → llrun 别泄漏敏感目录 fd（正常写法天然满足）。
- allowlist 要覆盖 CUDA/NCCL 全部路径（`/dev/nvidia*`、驱动 lib、`/dev/shm`、HF cache、workspace），
  漏一个就 EACCES——**这份 allowlist 是主要工程量**，一次调好可复用。

**mount-free 备选：proot（ptrace）。** devspace `ptrace` 可用，proot 能做 chroot 式视图+路径重映射，
但逐 syscall 拦截、开销大、易与 GPU/训练冲突 → 只适合轻量 agent，**不如 Landlock**，备而不用。

**兜底（不想用 Landlock 时，§5.6.6，零依赖）：** 每 trial 一台 devspace（rootfs 天然隔离）+ agent 跑非
root uid + `chmod 711/700` 自己的 `/user/<you>` + 软链 cache。缺点：每 trial 一台（费 GPU/心跳）、
只挡自己目录。

> 一句话：**隔离用 Landlock——已实测可行、零特权、绕开 mount 封锁、支持一机多 trial，唯一成本是调
> allowlist。** bwrap 需平台放开 AppArmor；proot/权限兜底次之。

### 8.6 数据/缓存（沿用 §5.6，不变）

- `capabilities.mounted=False`，走 tar/rsync-over-ssh 传 `/workspace`+`prepared`（EC2 原样）。
- 110GB models/datasets 缓存**一次性预热**到 `/user/chenyinghao/ptdbench-cache/{models,datasets}`
  （§5.6.4 子路 1，`/user` 自动挂载、免 PVC/RBAC），评测时软链进 `/workspace`。
- ⚠️ search_r1(cu130) 的 90GB 预热仍卡 [[cybertron-cuda13-driver-mismatch]]，等换驱动够新的池。

### 8.7 实现清单 & 待实测

**代码（fork ec2.py）**
- [ ] `EnvironmentType.CYBERTRON` 枚举 + `factory.py` 注册。
- [ ] `CybertronEnvironment`：`start`=`cctl devspace create`+解析 Teleport host+心跳；
      `exec/upload/download`=EC2 的 ssh/tar；`stop`=`cctl devspace stop`。
- [ ] 镜像名→数字 id 映射；`--billing-account-id N00069`、`--resource-pool speech-wx`。
- [ ] agent 命令按 §8.4/§8.5 决定是否包 `bwrap`。

**必须先实测（决定架构分叉）**
1. ~~devspace 是否放开 mount / 隔离怎么做~~ **✅ 已定（§8.4/§8.5）：devspace 禁 mount（AppArmor
   `cri-containerd` deny mount）→ bwrap 用不了；但 **Landlock LSM 实测可行**（ABI v1，零特权、无 mount、
   支持一机多 trial、GPU 设备节点加进 allowlist 即通）。→ 隔离主线 = Landlock `llrun` 包 agent；主要
   工程量是调 CUDA/NCCL allowlist。**
2. **心跳能否稳定维持 GPU devspace 不被到点回收**（§8.3）→ 定 devspace 主线 vs 回退 §5.5 (C) BATCH。
3. **Teleport SSH 证书**在 harbor 控制器所在机器的续签方式（长跑不掉线）。
4. expose/SSH 从控制器网络可达性（devspace 走 Teleport，一般本机 `~/.ssh/config` 即可）。
