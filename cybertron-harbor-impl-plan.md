# Cybertron harbor backend 实现计划（devspace + SSH + Landlock）

> 写给：接手实现 `cybertron` 这个 harbor environment backend 的同学。
> 本文是**可直接照着写代码**的实现计划，自成一份；背景可行性/探测证据见
> `cybertron-harbor-backend.md`（本文不再重复推导，只引用结论）。
> 记录日期：2026-09-29。目标池 `speech-wx`（H100 / cluster `wuxi_train`）、billing `N00069`。

---

## 0. 一句话方案

把 **Cybertron devspace 当长驻主机**，用它自带的 **Teleport SSH 通道**当 harbor 的
`exec`/`upload`/`download`（照抄 `ec2.py` 那套纯 SSH 实现），用 **cctl 管 devspace 生命周期**，
用 **Landlock LSM** 把在 devspace 里跑的 agent 关进「本 trial 的 workspace + 只读 cache」、
挡住共享盘 `/user` 上别人的数据。**不需要 k8s RBAC、不需要 mount 特权、不需要往镜像烤 exec-agent。**

---

## 1. 为什么是这套（关键事实基线，均已实测）

| 事实 | 结论 | 证据 |
|---|---|---|
| harbor backend 契约 = 长驻容器上反复 `exec` + 双向传文件 | 需要一个持久 exec 通道 | `harbor/src/harbor/environments/base.py` 的 `@abstractmethod` |
| `cctl` 无 `exec/attach/cp`；pod 内 SA 零 RBAC | k8s-native（ACK 式）路线走不通 | `cybertron-harbor-backend.md` §3 |
| **devspace 有真 SSH**（Teleport 在 `~/.ssh/config` 写 `Host devspace-<id>`） | `ssh devspace-<id> "cmd"` + rsync/scp 即 exec + 传文件 | `devspace-driver` skill；实测 `ssh` 进 `tasks/711108` 成功 |
| harbor 已有**纯 SSH backend** `ec2.py`（`_ssh_exec` + tar-over-ssh） | 新 backend = fork 它，改 provisioning 即可 | `harbor/src/harbor/environments/ec2.py` |
| Cybertron 把资源池集群 fs **自动挂在 `/user`**；容器 ephemeral 很小 | 大数据/workspace 必须落 `/user`，不能写 rootfs | `cybertron-harbor-backend.md` §5.6.3 |
| devspace pod **禁 mount**（AppArmor `cri-containerd` deny mount） | bubblewrap / apptainer / bind / tmpfs 全不可用 | 实测 `tasks/711108`：每个 `mount()` EPERM |
| devspace **Landlock ABI v1 可用**（kernel 5.15），零特权自我约束 | 用 Landlock 做 agent 文件隔离 | 实测：restrict 后 `/user` open→EACCES，GPU 设备节点加白名单即通 |

> 净结论：**exec 用 devspace SSH，隔离用 Landlock。** 两者都已在真 devspace 上验证。

---

## 2. 总体架构

```
harbor 控制器（本机/有 Teleport 登录的机器，pod 外）
  │  cctl devspace create/get/stop         ← 生命周期、配额、计费、GPU
  │  ssh devspace-<id> "cmd"               ← exec（EC2 的 _ssh_exec）
  │  tar-over-ssh / rsync                  ← upload/download
  ▼
Cybertron devspace（长驻主机，一 trial 一台）
  ├─ /user  (自动挂载的集群共享盘, 10PB)   ← 只读 cache + 本 trial workspace 都放这
  │    └─ /user/<cluster_user>/ptdbench-cache/{models,datasets}   （110GB，一次性预热，多 trial 只读共享）
  │    └─ /user/<cluster_user>/harbor-runs/<session>/{workspace,logs}
  ├─ /workspace -> (symlink) 本 trial 的 /user/.../workspace
  ├─ /ptdbench/{models,datasets} -> (symlink) cache 路径
  └─ agent/verifier 进程：被 `llrun`（Landlock）关进 {workspace + RO cache + 系统盘 + /dev}
                          → 看不到 /user 上别人的数据
```

关键设计点：
- **一 trial 一台 devspace**（忠实照搬 EC2/ACK「一 host 一 trial」；`--n-concurrent=N` = N 台 devspace）。
  Landlock 控制的是**访问权**不是**路径名**，无法把两个 trial 的 workspace 都映射成 `/workspace`
  （那需要 bind mount，已被禁），所以并发靠多台 devspace，不靠一台多沙盒。见 §7 的可选优化。
- **workspace 放 `/user`**（ephemeral 太小），用 `ln -s` 让它出现在 ptdbench 期望的 `/workspace`。
- **Landlock 只包 agent/verifier 这类不可信执行**，不包 setup 阶段（装 agent 要写 `/usr`）。

---

## 3. 代码改动清单（fork `ec2.py`）

### 3.1 注册新 backend 类型

`harbor/src/harbor/models/environment_type.py`：
```python
class EnvironmentType(str, Enum):
    ...
    CYBERTRON = "cybertron"
```

`harbor/src/harbor/environments/factory.py` 的 `_ENVIRONMENT_REGISTRY`：
```python
EnvironmentType.CYBERTRON: _EnvEntry(
    "harbor.environments.cybertron",   # 新模块
    "CybertronEnvironment",
    None,                               # 无需额外 pip extra
),
```

### 3.2 `CybertronEnvironment`（新文件 `harbor/src/harbor/environments/cybertron.py`）

**复用 EC2 的**：`_ssh_base_cmd` / `_ssh_raw` / `_ssh_exec` / `_wait_for_ssh` /
`upload_file`/`upload_dir`/`download_file`/`download_dir`（tar-over-ssh，`pack_dir_to_bytes`/
`extract_dir_from_bytes`）。这些**一行不用改**——只要 `self._host` 指向 `devspace-<id>` 且用
本机 `~/.ssh/config`（Teleport 证书 + ControlMaster 复用，照 `devspace-driver` 的 SSH_OPTIONS）。

**要改的只有 provisioning / 生命周期：**

| 方法 | 实现 |
|---|---|
| `type()` | `return EnvironmentType.CYBERTRON` |
| `capabilities` | `EnvironmentCapabilities(gpus=True, mounted=False, disable_internet=False)`（`mounted=False` 照 ACK/EC2，全 tar 传） |
| `_validate_definition()` | 校验 image 能映射到平台数字 id；校验 resource-pool/billing 已配置 |
| `start(force_build)` | ① **解析集群用户名**（§4.0）缓存到 `self._cluster_user`；② `cctl devspace create ...`（§4.1）拿 devspace id；③ 轮询 `cctl devspace get <id> -F status` 到 `Running/Started`；④ 从 `~/.ssh/config` 解析 host（§4.2）；⑤ `_wait_for_ssh()`；⑥ 起**心跳线程**（§4.3）；⑦ **一次性 setup**（§4.4：建 per-trial 目录、软链 cache、编译 `llrun`）。 |
| `exec(cmd, *, confine=False, ...)` | 走 EC2 `_ssh_exec`；`confine=True` 时把 `cmd` 包成 `llrun -- cmd`（§5）。 |
| `upload_file/dir` `download_*` | EC2 原样（tar-over-ssh）；上传目标 = 本 trial 的 workspace 真实路径（`/user/.../workspace`）。 |
| `stop(delete)` | 停心跳线程；`delete=True` → `cctl devspace stop <id>`（只停 pod）。per-trial 目录在 `/user`（持久盘）上默认**保留**，仅在 `cybertron.keep_workspace=False` 时才 `rm -rf`（见 §6.2）。 |
| healthcheck | `cctl devspace get -F status` + `ssh ... true`。 |

**镜像名映射**：harbor `environment.image`（`ptdbench/<name>:latest`，见 `ptdbench/core/types.py`
`EnvironmentImage.tag`）→ cybertron 平台数字镜像 id。维护一张表（配置文件或 env）：
```
ptdbench/ptdbench-verl-vllm017  -> 25607123          # cu129
ptdbench/ptdbench-slime-rc0     -> 25607122          # cu126
ptdbench/ptdbench-slime         -> 25607131          # v0.3.2，已重建到 cu129（见 §8）
```

### 3.3 让 harbor 在「跑 agent / verifier」时传 `confine=True`

`exec()` 大部分调用（装 nodejs/npm、装 agent CLI）**不能**被 Landlock 限制（要写 `/usr`）。
只有 **agent.run()** 和 **verifier** 这两类「不可信执行」要 confine。三选一：

- **(推荐) 后端内部状态位**：`start()` 后 `self._sealed=False`；`agent.install()` 完成的信号之后
  置 `True`；`exec()` 里 `confine = self._sealed`。缺点：需要一个「install 完成」钩子。
- **exec kwarg**：给 `BaseEnvironment.exec` 加一个可选 `sandbox: bool=False`，agent/verifier
  的调用点传 `True`。改动小但动到基类签名（各 backend 忽略即可）。
- **命令前缀约定**：agent/verifier 命令由 ptdbench 侧自己前缀 `llrun --`。最省 harbor 改动，
  但把隔离逻辑漏到 ptdbench，不推荐。

推荐走 **exec kwarg**：`BaseEnvironment.exec(..., sandbox: bool = False)`，只有
`CybertronEnvironment` 消费它（其它 backend 无视），agent/verifier 调用点显式传 `sandbox=True`。

---

## 4. devspace 生命周期细节

### 4.0 解析集群用户名（不要用 `whoami`）

`/user/<user>` 目录按**平台账号拼音 id** 命名，与 pod 内 unix 用户无关。devspace pod 以
**root**（uid=0）运行，所以 `$(whoami)` 会返回 `root` 而不是 `chenyinghao` —— **绝不能用它拼
`/user` 路径**。`start()` 里解析一次真实用户名，缓存到 `self._cluster_user`，后续所有
`/user/<user>/...` 都用它。解析来源（按优先级，取第一个成功的）：

1. **cctl 登录身份**（首选，随登录自动变）：`cctl whoami` / `cctl user get` / 读 cctl 配置里的
   登录账号 —— ⚠️ 确切子命令需对 `cctl --help` 复核（§10）。
2. **显式配置**：backend 配置项 `cybertron.cluster_user`（兜底/覆盖用）。
3. **从 Teleport host 名回推**：host 形如 `devspace-<user>-<proj>-<user>-<id>`，可解析出 `<user>`。

```python
# 伪码
self._cluster_user = (
    cfg.get("cybertron", {}).get("cluster_user")
    or self._resolve_user_via_cctl()          # cctl whoami/user get（待定确切命令）
    or self._parse_user_from_ssh_host(self._host)
)
if not self._cluster_user:
    raise EnvironmentError("无法确定 Cybertron 集群用户名；请配置 cybertron.cluster_user")
```
> 这样换人登录（`cctl login` 成别的账号）时，`/user/<user>` 会自动跟着变，无需改代码。

### 4.1 create

```bash
cctl devspace create \
  --image <平台数字 id> \
  --gpu <N> --gpu-model h100 \
  --resource-pool speech-wx \
  --billing-account-id N00069 \
  --project <ptdbench 项目> \
  -o json     # 拿回 tasks/<id>
```
> ⚠️ `cctl devspace create` 的确切 flag 需对 `cctl devspace create --help` 复核（本文按已知
> job/devspace 通用 flag 写）。拿不到某 flag 时退回 `-f <Task JSON>`。

### 4.2 解析 SSH host / transport（M3：默认走 tsh）

devspace 的 Teleport node 形如 `devspace-<user>-<proj>-<id>`（实测样例
`devspace-chenyinghao-xinfer-chenyinghao-1098539`，末尾即 devspace 数字 id）。

**默认 transport = `tsh`（M3 起）**：`tsh ls --format json` 里找唯一以 `-<id>` 结尾的 node，
再 `tsh ssh --cluster=teleport.cybertron.modelbest.co root@<node> <cmd>` 直接执行。
**不依赖 `~/.ssh/config`**——所以 `--n-concurrent=N` 起 N 台新 devspace 时无需 Cursor 插件
或手工写 Host block（M1 曾靠手工补 block）。stdin/stdout 二进制透传已实测（tar-over-ssh 照走）。

**fallback transport = `ssh <alias>`（`use_tsh=False`）**：在 `~/.ssh/config` 里找唯一以
`-<id>` 结尾的 `Host devspace-*`（alias 须已由 Cursor 插件写好）。SSH 选项照 `devspace-driver`：
```
-o ControlMaster=auto -o ControlPersist=10m -o ControlPath=~/.ssh/devspace-%C
-o ServerAliveInterval=30 -o ServerAliveCountMax=3
```
可用 `ssh_host` kwarg 直接指定 node/alias 跳过解析；`tsh_path` / `teleport_cluster` 可覆盖。

### 4.3 心跳（必须）

devspace 有 deadline；**申请了 GPU 的 devspace 到点会被自动快照并释放**，长跑中途会被抽走。
- backend 为每台 devspace 起一个后台心跳线程，周期性续期。
- **cctl 无 heartbeat 子命令** → 直接 `POST <控制面>/.../:heartbeat`（需确认端点），或验证
  「周期性 SSH 活动是否足以续期」——**这是必须先实测的点**（§8）。
- `stop()` 要停掉心跳线程。

### 4.4 一次性 setup（`start()` 末尾，unconfined exec）

在 devspace 上跑一段初始化（不被 Landlock 限制）。`<cluster_user>` 由 §4.0 解析后由后端填入
（**不是** `$(whoami)`）：
```bash
set -e
U=/user/<cluster_user>          # 后端用 self._cluster_user 填，非 whoami（pod 里是 root）
RUN=$U/harbor-runs/<session_id>
mkdir -p "$RUN/workspace" "$RUN/logs"
# ptdbench 期望的绝对路径 → 软链到 /user（bind 不可用，软链可）
ln -sfn "$RUN/workspace"            /workspace
mkdir -p /ptdbench
ln -sfn "$U/ptdbench-cache/models"   /ptdbench/models
ln -sfn "$U/ptdbench-cache/datasets" /ptdbench/datasets
# 编译 Landlock 启动器（镜像里若已带则跳过）
command -v llrun || gcc -O2 -o /usr/local/bin/llrun /tmp/llrun.c   # llrun.c 见 §5
```
> `/workspace`、`/ptdbench/*` 是符号链接，实际落在 `/user`（持久、大容量、免 ephemeral 限制）。
> Landlock 按**解析后的真实路径**判权，所以白名单里放的是 `/user/.../workspace` 与 cache 真实路径。

---

## 5. 文件隔离：Landlock（核心）

### 5.1 原理

Landlock 是**进程自我约束**的 LSM：进程建一个 ruleset（声明要管哪些访问类型）、对允许的目录
逐个 `add_rule`、`prctl(PR_SET_NO_NEW_PRIVS,1)` 后 `landlock_restrict_self`。此后该进程**及其所有
子进程**对白名单外路径的访问一律 `EACCES`，且不可撤销。**不用 mount、不用 namespace、不用任何
capability**（root/非 root 都行）——因此完全绕开 devspace 的 AppArmor mount 封锁。

内核 5.15 → **ABI v1**（只管文件路径访问；无网络/ioctl/refer 管控，够用）。syscall 号
`landlock_create_ruleset=444`、`landlock_add_rule=445`、`landlock_restrict_self=446`。

### 5.2 `llrun`：给 agent 套 Landlock 的小启动器

用法：`llrun [--ro DIR]... [--rw DIR]... -- <cmd> [args...]`
（内置一份系统只读盘 + `/dev` 白名单；额外目录用 `--ro/--rw` 传）。

后端拼装的典型调用（wrap agent.run）：
```bash
llrun \
  --rw /user/<cluster_user>/harbor-runs/<session>/workspace \
  --ro /user/<cluster_user>/ptdbench-cache \
  --rw /tmp --rw /dev/shm \
  -- codex exec --dangerously-bypass-approvals-and-sandbox ...
```
内置白名单（`llrun` 里 hardcode）：
- **RO+exec**：`/usr /lib /lib64 /bin /sbin /etc /opt /sys /run`
- **RW（含 GPU 节点）**：`/dev /tmp /dev/shm /var/tmp /proc`。`/proc` 必须 RW——NVIDIA 驱动在 CUDA context 初始化时会写 `/proc`，只读会得到 `cudaErrorOperatingSystem`(304)、torch `cuda False`。
- **不放行 `/user` 根**（只放行本 trial 的 workspace + cache 子路径）→ agent 看不到别人数据。

### 5.3 `llrun.c`（可直接编，raw syscall，无需内核头）

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <linux/types.h>

#define NR_create   444
#define NR_add      445
#define NR_restrict 446
#define RULE_PATH_BENEATH 1

struct ruleset_attr { __u64 handled_access_fs; };
struct pb_attr { __u64 allowed_access; __s32 parent_fd; } __attribute__((packed));

/* ABI v1: 全部 13 个 fs 访问位 */
#define FS_ALL   0x1fffULL
#define FS_EXEC  (1ULL<<0)
#define FS_WRITE (1ULL<<1)
#define FS_READ  (1ULL<<2)
#define FS_RDIR  (1ULL<<3)

static long l_create(const void*a,size_t s,__u32 f){return syscall(NR_create,a,s,f);}
static long l_add(int fd,const struct pb_attr*a){return syscall(NR_add,fd,RULE_PATH_BENEATH,a,0);}
static long l_restrict(int fd){return syscall(NR_restrict,fd,0);}

static int allow(int rs, const char* path, __u64 acc){
    int fd = open(path, O_PATH|O_CLOEXEC);
    if (fd < 0) return 0;                       /* 路径不存在就跳过 */
    struct pb_attr a = { .allowed_access = acc, .parent_fd = fd };
    long r = l_add(rs, &a);
    close(fd);
    if (r) fprintf(stderr, "llrun: add_rule %s: %s\n", path, strerror(errno));
    return 0;
}

int main(int argc, char** argv){
    struct ruleset_attr ra = { .handled_access_fs = FS_ALL };
    int rs = l_create(&ra, sizeof(ra), 0);
    if (rs < 0){ fprintf(stderr,"llrun: create_ruleset: %s\n", strerror(errno)); return 90; }

    const __u64 RO = FS_READ|FS_RDIR|FS_EXEC;
    const char* ro[] = {"/usr","/lib","/lib64","/bin","/sbin","/etc","/opt",
                        "/sys","/run", NULL};
    for (int i=0; ro[i]; i++) allow(rs, ro[i], RO);
    /* /proc 必须 RW：NVIDIA 驱动在 CUDA context 初始化时写 /proc，
       只读会导致 cudaErrorOperatingSystem(304) / torch cuda False。 */
    const char* rw[] = {"/dev","/tmp","/dev/shm","/var/tmp","/proc", NULL};
    for (int i=0; rw[i]; i++) allow(rs, rw[i], FS_ALL);

    /* 解析 --ro/--rw ... -- cmd */
    int i = 1;
    for (; i < argc; i++){
        if (!strcmp(argv[i],"--")) { i++; break; }
        else if (!strcmp(argv[i],"--ro") && i+1<argc) allow(rs, argv[++i], RO);
        else if (!strcmp(argv[i],"--rw") && i+1<argc) allow(rs, argv[++i], FS_ALL);
        else { fprintf(stderr,"llrun: bad arg %s\n", argv[i]); return 91; }
    }
    if (i >= argc){ fprintf(stderr,"llrun: no command\n"); return 92; }

    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)){ perror("prctl"); return 93; }
    if (l_restrict(rs)){ fprintf(stderr,"llrun: restrict_self: %s\n", strerror(errno)); return 94; }

    execvp(argv[i], &argv[i]);
    fprintf(stderr,"llrun: exec %s: %s\n", argv[i], strerror(errno));
    return 95;
}
```

### 5.4 已实测（devspace `tasks/711108`, kernel 5.15）

- ABI v1 可用；`restrict_self` 成功（root 与非 root 都行）。
- 受限后：`/tmp`（放行）可开；`/user`、`/etc/hostname`（未放行）→ **Permission denied**。
- 放行 `/usr /lib` 后受限进程能正常 `execvp` 真二进制。
- `open(/dev/nvidiactl)`：放行 `/dev` → OK，不放行 → denied → **GPU 设备节点就是普通文件 open**，
  进白名单即通（CUDA/NCCL 走 `/dev/nvidia*` + 驱动 lib + `/dev/shm`）。

### 5.5 边界与注意（诚实说明）

- **只管访问不管命名**：`ls /` 仍见 `/user` 这个名字，但读其内容 → EACCES。安全属性是「拒绝访问」。
- **不做路径重映射**：cache 出现在 `/ptdbench/models` 靠 `ln -s`，不是 Landlock。
- **allowlist 要覆盖 CUDA/NCCL 全部路径**（`/dev/nvidia*`、驱动 `.so`、`/dev/shm`、HF cache、
  workspace），**漏一个就 EACCES** —— 这是主要调试量，一次调好可复用。建议先宽（放 `/dev` 整个），
  跑通后再按需收敛。
- **agent 运行时装包**：Landlock 下写 `/usr` 会被拒。若某 agent 运行中 `pip/npm -g` 安装，需要
  给它一个可写前缀（`--rw <venv/prefix>`）或在 setup 阶段预装好。
- **网络不受影响**（ABI v1 不管网络）→ agent 照常访问 LLM API。
- restrict 前已打开的 fd 不受限 → `llrun` 在 restrict 前只 `open(O_PATH)` 目录并随即 close，不泄漏。
- 进程可见性（`/proc` 里看别的进程）Landlock 不管；一 trial 一 devspace 时无所谓。

---

## 6. 数据与缓存

### 6.1 只读大缓存（models/datasets, ~110GB）

- **一次性预热**到 `/user/<cluster_user>/ptdbench-cache/{models,datasets}`：起一个一次性 BATCH job
  （`--code-type git` 拉本仓库、联网、跑全部 `environment/prepare.sh`），把
  `PTDBENCH_MODELS`/`PTDBENCH_DATASETS` 指向该路径（写 `/user` 才不撞 ephemeral 限制）。
- 之后所有 trial 的 devspace **只读共享**同一份（多读安全，无需 PVC/RBAC）。
- 跨集群/池搬运用 `cctl file-task sync`。

### 6.2 per-trial workspace（小/中, 可写）—— 生命周期：本地上传 → agent 改 → 保留

- 真实路径 `/user/<cluster_user>/harbor-runs/<session>/workspace`，`/workspace` 软链过去。
- **初始态从本地上传**：agent 运行前，harbor 的 `upload_dir`（tar-over-ssh）把本地 prepared
  workspace 灌进这个真实路径 → agent 拿到的就是初始 workspace（每 trial 一份干净副本）。
- ptdbench 原本**硬链接**权重进 workspace 的那部分（几 GB）仍随 workspace 走；大 cache 不进 workspace，
  用 §6.1 的软链。
- **agent 操作后 workspace 被保留（双重）**：
  1. workspace 实际落在 `/user`（**持久**集群盘）。`stop(delete=True)` 停的是 **devspace pod**，
     `/user` 上的数据不受影响——**除非** backend 显式 `rm -rf` 该 per-trial 目录。默认**不删**，
     可用配置项 `cybertron.keep_workspace`（默认 True）控制是否保留 / 何时清理。
  2. harbor 结束时照常 `download_dir` 把该 workspace 拉回控制器做评分 / 存档。
- 因此 agent 结束后可随时回到 `/user/<cluster_user>/harbor-runs/<session>/workspace` 复查最终态，
  或重新 SSH 进 devspace 接着看（只要该目录没被清理）。

### 6.3 logs / 产物

- `/logs` 照 ACK：`mkdir -p` + `chmod 777`，结束 `download_dir` 拉回控制器。放 `/user/.../logs`。

---

## 7. 并发模型

- **默认：一 trial 一台 devspace**（忠实 EC2/ACK 语义）。`--n-concurrent=N` → N 台 devspace，
  各自 `/workspace`、各自心跳。Landlock 在每台里挡住 `/user` 别人的数据。
- **代价**：N 份 GPU 配额 + N 条心跳 + N 次 devspace create（比起 pod 慢）。
- **可选优化（gated）：一台 devspace 跑多 trial**。前提是让 ptdbench/harbor 的 workspace 路径
  **可 per-trial 参数化**（不再写死 `/workspace`），这样一台 devspace 上多个 trial 落在不同子目录，
  Landlock 再给每个 agent 只放行自己的子目录 → trial 间互不可见。**需要 harbor/ptdbench 侧改动**，
  非首版目标。

---

## 8. 已知硬约束（实现时必须绕开）

- **CUDA 13 驱动不匹配（已解决）**：slime v0.3.2 原先的 cu130 构建在 wuxi_train/speech-wx
  的节点起不来（`cuda>=13.0` 驱动太旧），连 `--gpu 0` 也被 nvidia hook 拦。现已把 v0.3.2
  重建到 cu129（`ptdbench-slime-v0.3.2-cu129`, id 25607131），`slime_retool*` / `slime_search_r1*`
  可直接在 speech-wx 跑；`slime-rc0`(cu126) 与 verl(cu129) 一直 OK。
- **ephemeral 很小**：一切大文件必须落 `/user`，不能写容器 rootfs（否则「临时存储超限」）。
- **mount 全禁**：不要在实现里依赖任何 bind/tmpfs/overlay/pivot_root/bwrap/apptainer。
- **search_r1 的 90GB wiki-18 索引**预热量大，且卡上面的 cu130 驱动问题。

---

## 9. 分阶段落地

**M1 — 骨架跑通（无隔离、单并发）**
- [ ] 枚举 + factory 注册；`CybertronEnvironment` fork ec2。
- [ ] `start` = `cctl devspace create` + 解析 host + `_wait_for_ssh` + setup 软链。
- [ ] `exec`/`upload`/`download` 直接用 EC2 的 SSH/tar。
- [ ] 手动预热一个小任务的 cache，跑通 `hello-world` 级 trial（prepare→start→install→run→verify→finalize）。
- [ ] 心跳最简实现 + 实测「能否保住 GPU devspace 不被回收」。

**M2 — 加 Landlock 隔离**
- [ ] setup 阶段编译 `llrun`（或烤进镜像）。
- [ ] `exec(sandbox=True)` 包 `llrun`；agent.run / verifier 调用点传 `sandbox=True`。
- [ ] 用一个真 GPU 任务把 CUDA/NCCL allowlist 调完整（先放 `/dev` 整个跑通，再收敛）。
- [ ] 验证：agent 内 `ls /user/<别人>` → 拒绝；训练/rollout 正常。

**M3 — 全量缓存 + 并发**
- [x] 一次性 BATCH job 预热全部 `prepare.sh` 到 `/user/.../ptdbench-cache`：脚本
  `harbor/scripts/cybertron_preheat.py`（`cctl job create --code-type git ... --entry`，
  默认 dry-run 打印命令，`--run` 才提交）。cu130 那批仍待新驱动池，勿在 speech-wx 跑。
- [x] `--n-concurrent=N` 多 devspace：改用 tsh transport（§4.2）后每 trial 独立 devspace/
  node/心跳/run 目录，构造上即并发安全、无需手工 SSH block。单节点 transport 已实测
  （node 解析 + exec + stdin 透传）；**N-GPU 端到端长跑 + 心跳保活仍需真机验证（§10.1）**。
- [x] 镜像名→数字 id 映射落配置：内置 `_DEFAULT_IMAGE_MAP` + `image_map`（inline dict）
  + `image_map_path`（JSON 文件）三级合并，优先级 `image_id` > `image_map` > 文件 > 内置 > 透传。

---

## 10. 必须先实测/确认的点

1. **心跳**：GPU devspace 到 deadline 会被快照+释放；确认续期端点（`:heartbeat`）或「周期 SSH
   活动是否足以续期」。→ 决定 devspace 主线是否成立（否则退 BATCH+expose，见
   `cybertron-harbor-backend.md` §5.5 形态 C）。
2. **Teleport SSH 证书**在控制器机器上的续签方式（`tsh login`），保证长跑不掉线。
3. `cctl devspace create` 的确切 flag（对 `--help` 复核 gpu/pool/billing/project/image）。
   同时确认**取当前登录用户名**的 cctl 命令（`cctl whoami` / `cctl user get` / 配置文件），供 §4.0 用。
4. **CUDA/NCCL 的 Landlock allowlist** 是否完整（真 GPU 任务下逐一补齐）。
5. 控制器网络能否 SSH 到 Teleport host（一般本机 `~/.ssh/config` 即可，跨网需确认）。

---

## 附录 A · 路径约定

> `<cluster_user>` = §4.0 从 cctl 登录解析出的账号名（如 `chenyinghao`），**不是** pod 里的 `whoami`（root）。

| 逻辑路径（ptdbench 期望） | 真实路径（devspace 上） | 谁填 |
|---|---|---|
| `/workspace` | `→ /user/<cluster_user>/harbor-runs/<session>/workspace`（软链） | harbor tar 上传 |
| `/ptdbench/models` | `→ /user/<cluster_user>/ptdbench-cache/models`（软链，RO） | 预热 job |
| `/ptdbench/datasets` | `→ /user/<cluster_user>/ptdbench-cache/datasets`（软链，RO） | 预热 job |
| `/logs` | `/user/<cluster_user>/harbor-runs/<session>/logs` | harbor 建 + 拉回 |

## 附录 B · 复用 vs 新写

| 来自 `ec2.py` 直接复用 | 新写 |
|---|---|
| `_ssh_base_cmd` `_ssh_raw` `_ssh_exec` `_wait_for_ssh` | `start`（cctl devspace create + host 解析 + setup） |
| `upload_file/dir` `download_file/dir`（tar-over-ssh） | 心跳线程 |
| tar 打包/解包（`tar_transfer`） | `exec` 的 `sandbox=True` → `llrun` 包裹 |
| `capabilities`（改 `mounted=False, gpus=True`） | `stop`（cctl devspace stop + 停心跳） |
|  | 镜像名→数字 id 映射；`llrun.c`（§5.3） |
