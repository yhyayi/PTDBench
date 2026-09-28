# PTDBench Docker 环境设计

本文介绍 `docker/` 目录下评测镜像的设计：它解决什么问题、镜像怎么组织、任务怎么选镜像、
怎么构建、运行时怎么塑造每个阶段的环境。

---

## 1. 要解决的问题

这是一个**以复现特定训练失败为评分标准**的 benchmark。这一句话决定了环境设计的取舍：

- 环境如果会漂，被复现的"ground truth"就会被悄悄改写，而且不会有任何东西报错。
- 因此"可复现"优先于"统一"、"镜像少"。
- 因此宁可**显式失败**，也不要**静默降级**。

同时，任务集横跨了三套互不兼容的训练栈（verl 与两个版本的 slime），CUDA 分别是
12.9 / 12.6 / 13.0。

---

## 2. 总体形态：一个环境一个镜像

`docker/images/` 下每个 `<name>.Dockerfile` 就是一个环境，构建出的镜像 tag 固定为
`ptdbench/<name>:latest`：

| 环境名 | 上游镜像 | 栈 | 任务 |
|---|---|---|---|
| `verl` | `verlai/verl:vllm017.latest` | vLLM 0.17，torch 2.10 / CUDA 12.9 | `reward_design/*`、`verl_hyper/*`、`verl_implementation/*` |
| `slime-rc0` | `slimerl/slime:v0.5.0rc0-cu126` | sglang 0.5.0rc0，torch 2.8 / CUDA 12.6 | `rlve_hyper/*` |
| `slime-v0.3.2` | `slimerl/slime:v0.3.2-cu130` | sglang 0.5.15.post1，CUDA 13.0 | `slime_retool*`、`slime_search_r1*` |
| `python3.11-slim` | `python:3.11-slim` | 纯 CPython，无 GPU | `example/swe_slugfix` |

任务在 `metadata.yaml` 里只写环境名：

```yaml
environment:
  image: verl
```

可选值就是 `docker/images/` 下的文件名，代码里没有第二份常量表（`known_images()` 直接列目录）。
写了不存在的名字、或者还在用旧的 `conda` / `requirements` / `dockerfile` 字段，
`ptdbench validate` 会直接报错。

**上游镜像原样使用。** 每个 Dockerfile 只有两部分：一行钉死 digest 的 `FROM`，加上一段
所有镜像完全相同的 PTDBench 工具块（见第 4 节）。不在镜像里重新编译框架栈 —— 上游镜像就是
这些任务录制时的运行时底座，框架源码则随任务的 workspace 一起来。

**为什么 slime 有两个环境**：不是新旧补丁关系，而是两套端到端不同的栈。retool /
search-r1 的 recipe 要 import `sglang.srt.constants.GPU_MEMORY_TYPE_CUDA_GRAPH`，
0.5.0rc0 里没有这个常量；而 rlve 四个任务的既有成绩是在 0.5.0rc0 上录的，往前挪就等于
换 torch 重写它们的 ground truth。谈不拢的两套栈就拿两个名字。

同理，一个镜像被多个任务共享，镜像内容就是这些任务共同的 ground truth 环境。改一个 Dockerfile
等于同时改所有用它的任务；需要不同依赖的任务应该开一个新环境名。只缺一两个纯 Python 包的任务
在 `environment/prepare.sh` 里 `pip install --target` 到 workspace，再通过 `prepared/env` 的
`PYTHONPATH` 交给后两个阶段（见 `tasks/reward_design/task_evaluator`）——prepare 容器对镜像
site-packages 的改动不会带到 agent / verifier 容器里。

---

## 3. 构建

### 按需构建

评测时 docker backend 先检查任务的镜像 `ptdbench/<name>:latest` 是否已存在：存在就直接用，
不存在才用 `docker/images/<name>.Dockerfile` 构建（构建上下文是 `docker/`）。
prepare / agent / verifier 三个阶段用同一个镜像，各自起容器。

### 显式预构建：`ptdbench build-images`

```sh
python3 -m ptdbench build-images                  # docker/images/ 下所有缺失的镜像
python3 -m ptdbench build-images verl slime-rc0   # 只处理指定的
python3 -m ptdbench build-images --rebuild verl   # 已存在也重建
```

**tag 是固定的，不从内容派生。** 所以改了 Dockerfile 或工具块（`ptdbench_fetch.py` 等）之后，
已有的镜像不会被自动发现过期，需要 `--rebuild`。

### FROM 必须钉 digest

上游的 tag 会移动（`vllm017.latest` 这种尤其如此），而任务的录制结果不能跟着移动。所以
`FROM` 一律写成 `image:tag@sha256:...`，测试会检查每个 Dockerfile 都满足这一点。
换 digest 是一次有意识的升级：换完之后要把用这个环境的任务重跑一遍。

### 构建上下文是白名单

`docker/.dockerignore` 先排除一切，再只放行镜像需要 COPY 的几个文件：
`ptdbench-activate.sh`、`ptdbench_fetch.py`、`ptdbench_dataset.py`、`ptdbench_datasets/`。
`docker/` 下其余东西（`warm-cache.sh`、本文档、残留的下载缓存）都跑在宿主机上，不进上下文。

### 加一个新环境

1. 新建 `docker/images/<name>.Dockerfile`，开头注释写清上游镜像、栈和用它的任务；
2. `FROM` 钉 digest；
3. 从已有 Dockerfile 里把 `# ---- PTDBench tooling: keep identical in every docker/images/*.Dockerfile ----`
   之后的部分原样复制过来 —— 测试会检查所有 Dockerfile 这一段逐字相同；
4. 任务的 `metadata.yaml` 写 `environment.image: <name>`。

---

## 4. 工具块与运行时

### 工具块

每个镜像末尾都有同一段：

- `mkdir -p /ptdbench/{prepared,models,datasets}`，让它们是挂载点而不是 docker 临时造的目录；
- 把镜像自己的 `PATH` 记到 `/etc/ptdbench-image-path`；
- 装 activate hook 到 `/etc/profile.d/ptdbench-activate.sh`；
- 装 `ptdbench_fetch_hf` / `ptdbench_fetch_url` / `ptdbench_dataset` 三个命令。

### `ptdbench-activate.sh`

harness 跑的每条命令都走 `sh -lc`，所以这个文件是塑造"每阶段环境"的唯一入口。用 POSIX sh。
上游镜像原样使用，没有环境需要"激活"，所以它只做三件事：

0. **补回镜像的 PATH** —— login shell 会先跑 `/etc/profile`，在 Debian 系上它会把 PATH 重置成
   默认列表，丢掉镜像 `ENV PATH` 加的 `/usr/local/cuda/bin` 之类。hook 对照
   `/etc/ptdbench-image-path` 把缺的补回前面。
1. **接上 prepare 的产物** —— 读 `$PTDBENCH_PREPARED/env` 的 `KEY=VALUE` 行。
   这个文件被当作**数据**逐行解析，不是 `.` 进 shell：它在 agent 和 verifier 的每条命令里都会被读，
   任务包不该能借此在所有命令里执行任意 shell。它放在 `$PTDBENCH_PREPARED` 而不是 workspace，
   因为 agent 能写 workspace，而 verifier 也读这个文件 —— 评分不能参考 agent 改得动的东西。
2. **给乱装的包找个落点（仅 agent 阶段）** —— 设 `PYTHONUSERBASE=/workspace/.ptdbench-user`、
   `PIP_USER=1`，以及 `PIP_BREAK_SYSTEM_PACKAGES=1`（Ubuntu 24.04 的系统解释器标了
   externally managed，不设的话连 `--user` 安装都会被拒）。agent 真去 `pip install` 的话
   落在自己的 workspace 里，不改镜像的 site-packages，verifier 也看不见。

### 三个阶段与挂载

| 阶段 | 干什么 | 网络 | `/ptdbench/{models,datasets}` |
|---|---|---|---|
| `prepare` | 跑任务的 `environment/prepare.sh`：下权重、下数据集、写 `prepared/env` | 需要 | 可写 |
| `agent` | agent 在 workspace 里干活 | 按 backend 策略 | 只读 |
| `verifier` | 评分 | — | 只读 |

每个阶段的离线相关变量（`HF_HUB_OFFLINE` 等）由 `SandboxRuntime._network_policy_env` 统一给出。

### 镜像内置的抓取工具

| 命令 | 作用 |
|---|---|
| `ptdbench_fetch_hf` / `ptdbench_fetch_url` | 按 pin 死的 revision 取权重和数据集；缓存优先、sha256 校验、**不匹配即硬失败，绝不静默重下** |
| `ptdbench_dataset` | 跑镜像内置的数据集配方（`docker/ptdbench_datasets/`），处理"需要变换而非下载"的那类数据集 |

这几个 shim 在调用时才解析 `python3`，所以跑在镜像自带的解释器里，用的是那个镜像的
huggingface_hub / datasets。

### 离线评测：`warm-cache.sh`

把下载从镜像构建挪到 prepare 阶段，代价是评测机要能联网 —— 对 airgapped 的 GPU 节点是笔烂账。
`./docker/warm-cache.sh [--models-dir DIR] [--datasets-dir DIR]` 在有网的机器上把每个任务的
`prepare.sh` 各跑一遍（缺镜像时先 `build-images`），填满两个缓存根目录；把它们搬到评测机上，
之后每个 prepare 阶段都是缓存命中。镜像本身用 `docker save` / `docker load` 搬过去。
它不产出可运行的 workspace：每个任务拿到的 `/workspace` 和 `prepared` 用完即弃，只有两个缓存活下来。

---

## 5. 边界与已知事项

- **Harbor 路径还没跟上。** `harbor_evaluator.py` 及其测试仍按旧的 conda / 逐任务 Dockerfile
  两种形态写，下一阶段再改。
- **`docker/wheels/`、`docker/gitmirror/` 是统一镜像时代的下载缓存残留**，已不被任何东西使用，
  只是还没从磁盘上删；删掉之后 `.gitignore` 里对应的两行也可以去掉。
- **旧环境的 Megatron 补丁**（原 `docker/envs/*/patches/`）随统一镜像一起删除了，
  需要时可以从 git 历史里找回。
