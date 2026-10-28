from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..core.types import CommandResult, EnvironmentImage


# 以前这段逻辑装在镜像的 /etc/profile.d/ptdbench-activate.sh，靠登录 shell（`sh -lc`）
# 自动 source。现在不再烤进镜像：harness 在运行时把它拼到每条 shell 命令前面（见
# CommandSpec.to_argv），改一次 push 一次就生效，不用重建十几个镜像。
#
# 做三件事：
#   0. 恢复镜像的 PATH（section 0，从老 /etc/profile.d 钩子移植回来）。本地 docker
#      后端确实不再用登录 shell、/etc/profile 不会重置 PATH，但 Harbor/cybertron 后端
#      的命令经 ssh 走非登录 `bash -c`，PATH 是 sshd 的默认列表，没有镜像 ENV 加的
#      /opt/venv/bin、/usr/local/cuda/bin —— ray/nvidia-smi 一律 command not found。
#      Dockerfile 仍把镜像 PATH 记录在 /etc/ptdbench-image-path（老镜像里也有），
#      缺的目录按镜像顺序补回 PATH 前面；镜像里没这个文件或 PATH 完整时是 no-op。
#   1. 把 prepare 阶段写进 $PTDBENCH_PREPARED/env 的 KEY=VALUE 行 export 出来
#      （PYTHONPATH 之类），给 agent / verifier / rerun 阶段用。只认 bare NAME=...，
#      挡掉命令替换——任务不该能往每个阶段塞任意 shell。
#   2. 把 pip 的 --user 落点指到 agent 自己的 workspace（agent / rerun 阶段）。
#
# POSIX sh：拼进 `sh -c` 执行，不能用 `return`（那只在被 source 时有效）。用 if 包住，
# 让控制流自然落到后面拼接的真实命令。Harbor 后端不走这里，它把同样的内容
# materialize 成文件、靠 BASH_ENV 加载（见 harbor_evaluator.ACTIVATE_HOOK）。
PTDBENCH_ACTIVATE_BODY = r"""if [ -n "${PTDBENCH_STAGE:-}" ]; then
    if [ -r /etc/ptdbench-image-path ]; then
        _ptdbench_image_path=""
        IFS= read -r _ptdbench_image_path < /etc/ptdbench-image-path || :
        _ptdbench_missing=""
        _ptdbench_ifs=$IFS
        IFS=:
        for _ptdbench_dir in $_ptdbench_image_path; do
            [ -n "$_ptdbench_dir" ] || continue
            case ":$_ptdbench_missing:$PATH:" in
                *":$_ptdbench_dir:"*) ;;
                *) _ptdbench_missing="${_ptdbench_missing:+$_ptdbench_missing:}$_ptdbench_dir" ;;
            esac
        done
        IFS=$_ptdbench_ifs
        if [ -n "$_ptdbench_missing" ]; then
            PATH="$_ptdbench_missing${PATH:+:$PATH}"
            export PATH
        fi
        unset _ptdbench_image_path _ptdbench_missing _ptdbench_ifs _ptdbench_dir
    fi
    PTDBENCH_PREPARED="${PTDBENCH_PREPARED:-/ptdbench/prepared}"
    export PTDBENCH_PREPARED
    if [ -f "$PTDBENCH_PREPARED/env" ]; then
        while IFS= read -r _ptdbench_line || [ -n "$_ptdbench_line" ]; do
            case "$_ptdbench_line" in
                ''|'#'*) continue ;;
            esac
            _ptdbench_key="${_ptdbench_line%%=*}"
            case "$_ptdbench_key" in
                ''|*[!A-Za-z0-9_]*)
                    echo "ptdbench: ignoring malformed line in prepared/env: $_ptdbench_line" >&2
                    continue ;;
            esac
            export "$_ptdbench_line"
        done < "$PTDBENCH_PREPARED/env"
        unset _ptdbench_line _ptdbench_key
    fi
    if [ "$PTDBENCH_STAGE" = "agent" ] || [ "$PTDBENCH_STAGE" = "rerun" ]; then
        PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.ptdbench-user}"
        PIP_USER=1
        PIP_BREAK_SYSTEM_PACKAGES=1
        export PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
    fi
fi
"""


@dataclass(frozen=True)
class CommandSpec:
    """一次命令调用的描述，不包含任何平台专属参数。"""

    argv: Optional[Sequence[str]] = None
    shell: Optional[str] = None
    cwd: Optional[Path] = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: Optional[int] = None
    user: Optional[str] = None
    live_log_name: Optional[str] = None
    live_log_to_stderr: bool = False

    def to_argv(self) -> List[str]:
        if self.argv is not None:
            return list(self.argv)
        if self.shell is not None:
            # 非登录 shell（`sh -c`）：不跑 /etc/profile，PATH 保持镜像 ENV 的值。
            # 先跑 activate body（导出 prepared/env 等），再跑真实命令。
            return ["sh", "-c", PTDBENCH_ACTIVATE_BODY + "\n" + self.shell]
        raise ValueError("command spec needs argv or shell")


@dataclass(frozen=True)
class EnvironmentArtifact:
    """构建后的环境，可以在 agent/verifier 两个阶段复用。"""

    backend: str
    reference: Optional[str] = None


@dataclass(frozen=True)
class SandboxInstance:
    """A running, mutable sandbox created from an environment artifact."""

    backend: str
    reference: str
    request: "SandboxRequest"


@dataclass(frozen=True)
class SandboxRequest:
    """一次 prepare / agent / rerun / verifier 运行需要的宿主机资源。

    agent 之后、verifier 之前多了一个 rerun 阶段：solution 层协议里 agent
    只修代码不跑实验，实验由 harness 用 agent 留下的 workspace 重跑一遍，
    判分只认这一遍的新日志。
    """

    stage: str
    task_id: str
    host_task_root: Path
    host_visible_task: Path
    host_workspace: Path
    host_results: Path
    evaluation_level: str = "solution"
    host_oracle: Optional[Path] = None
    host_verifier: Optional[Path] = None
    # 任务的 environment/ 目录：prepare.sh 在里面。三个阶段都
    # 以只读挂，因为三个阶段都可能要读它——prepare 要执行 prepare.sh，agent 要知道
    # 环境里有什么，verifier 判分时可能要对着同一份清单。只读是因为没有任何一个阶段
    # 有理由改仓库里的任务定义。
    host_environment: Optional[Path] = None
    # prepare 阶段的产物落点（模型、数据集、写给后两个阶段的 env 文件）。
    # prepare 阶段可写，agent / verifier 阶段只读——判分不能依赖 agent 能改的东西。
    host_prepared: Optional[Path] = None
    # 跨 run、跨任务复用的下载缓存，按资产种类分两个根。**三个阶段都挂**：prepare
    # 可写（下载就发生在那里），agent / verifier 只读。后两个阶段能看见，是为了
    # 调试时对得上账，不是给任务绕开 prepare 用的——交付路径仍然只有 --dest 一条，
    # harbor 路径下这两个目录根本不在镜像里。
    host_models: Optional[Path] = None
    host_datasets: Optional[Path] = None
    workdir: str = "workspace"
    env: Mapping[str, str] = field(default_factory=dict)
    network: str = "bridge"
    timeout_s: Optional[int] = None
    # metadata.environment.env：任务自己声明的变量（如 CUDA_VISIBLE_DEVICES）。
    # 逐任务不同的变量只能从这里来。
    task_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_env", dict(self.task_env))
        if self.stage not in {"prepare", "agent", "rerun", "verifier"}:
            raise ValueError(
                "sandbox stage must be 'prepare', 'agent', 'rerun' or 'verifier'"
            )
        if self.workdir not in {"workspace", "task_root", "prepared"}:
            raise ValueError(
                "sandbox workdir must be 'workspace', 'task_root' or 'prepared'"
            )
        if self.stage == "prepare" and self.host_prepared is None:
            # prepare 阶段没有落点就等于白跑：脚本会成功，什么也没留下，直到 agent
            # 阶段找不到模型才暴露，而那时报的是任务的错。
            raise ValueError("the prepare stage needs host_prepared")


def sees_environment_dir(request: SandboxRequest) -> bool:
    """这个阶段该不该看得到任务的 environment/。

    prepare 要跑里面的 prepare.sh；verifier 偶尔要读里面的资源。**agent 不给**：
    prepare.sh 写着任务是怎么搭起来的——下哪个 HF 仓库、解哪个归档、校验哪些
    sha256——其中任何一个名字都可能直接点出要诊断的缺陷。

    rerun 是 harness 自己发起的，不是 agent 的延伸：它跑的是任务声明的
    run_spec，读 environment/ 没有额外风险，但它也没有需要——保守起见和
    verifier 一样看得见（判分脚本和 run_spec 资源同级，都在 evaluation
    目录里，不在 environment/ 里）。

    以前三个阶段共用一张挂载表，agent 也挂着；"别把答案写进 prepare.sh" 是写在
    prepare.sh 文件头里的一条约定，靠任务作者自觉。约定覆盖不到"仓库名本身就是
    答案"这种情况，所以改成机制。
    """
    return request.stage != "agent"


def sees_download_cache(request: SandboxRequest) -> bool:
    """这个阶段该不该看得到 $PTDBENCH_{MODELS,DATASETS}。

    prepare 要写（下载落点），verifier 要读（判分常要读 checkpoint 或对齐数据集
    样本）。**agent 不给**：HF 缓存的目录名就是仓库名（datasets--<owner>--<repo>），
    而仓库名有时就是照着它承载的那个缺陷起的，等于把答案写在目录名里。
    rerun 同理不挂——它跑的命令来自任务声明，不需要，也少一条向 agent 泄漏
    仓库名的通路（agent 阶段虽然看不到，但 rerun 的日志会留在 results 里，
    agent 改不了 results，这条纯粹是少挂一个没用的东西）。

    agent 也不需要它：prepare 交付的权重/数据集已经被
    materialize() 硬链进 workspace，是真实字节，缓存挂不挂都读得到。

    挂载表和环境变量都走这一个判据，免得两边漂移——漂移的表现是 PTDBENCH_MODELS
    指着一个没挂上的路径，报出来是"目录是空的"而不是"你看不到"。
    """
    return request.stage in {"prepare", "verifier"}


class SandboxRuntime(ABC):
    """运行平台的统一接口。

    agent 和 evaluator 只依赖这个接口，不直接依赖 Docker 或本机进程。
    Harbor、云沙盒等后端可以实现同一组方法。
    """

    name: str

    @property
    @abstractmethod
    def needs_environment_image(self) -> bool:
        """该平台是否需要按 metadata.environment.image 准备镜像。"""

    @abstractmethod
    def ensure_environment(
        self, image: EnvironmentImage, *, rebuild: bool = False
    ) -> EnvironmentArtifact:
        """准备任务声明的环境镜像并返回可以复用的 artifact。

        镜像按环境名共享：同一个环境的任务用同一个镜像，已存在就不再构建。
        """

    @abstractmethod
    def path(self, request: SandboxRequest, name: str) -> str:
        """返回逻辑路径在当前平台中的实际路径。"""

    @abstractmethod
    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        """在指定阶段执行命令。"""

    def create_sandbox(
        self,
        request: SandboxRequest,
        *,
        artifact: EnvironmentArtifact,
    ) -> SandboxInstance:
        """Create and start a mutable sandbox for one workflow stage."""
        raise NotImplementedError(f"{self.name} does not support persistent sandboxes")

    def exec(
        self,
        sandbox: SandboxInstance,
        spec: CommandSpec,
    ) -> CommandResult:
        """Execute a command inside an existing sandbox."""
        raise NotImplementedError(f"{self.name} does not support sandbox exec")

    def close_sandbox(self, sandbox: SandboxInstance) -> None:
        """Stop and remove a sandbox, best effort after failures."""
        raise NotImplementedError(f"{self.name} does not support sandbox cleanup")

    def standard_env(self, request: SandboxRequest) -> Dict[str, str]:
        """生成三个阶段都可以依赖的 PTDBench 环境变量。"""
        visible_task = self.path(request, "visible_task")
        workspace = self.path(request, "workspace")
        results = self.path(request, "results")
        if request.stage == "verifier":
            task_root = self.path(request, "task_root")
            visible = ""
            oracle = self.path(request, "oracle")
            verifier = self.path(request, "verifier")
        else:
            # prepare / agent / rerun 都只看得到 visible_task：prepare 跑的是任务
            # 自己提供的脚本，agent 和 rerun 都不该读到 oracle 或判分脚本。
            task_root = visible_task
            visible = visible_task
            oracle = ""
            verifier = ""
        env = {
            # agent 和 rerun 阶段给空：任务 id 往往直接点出缺陷
            # （verl_grpo_uid_after_repeat），和 oracle 一样不该让 agent 读到；
            # rerun 是 harness 发起的，不给它额外多任何信息。
            "PTDBENCH_TASK_ID": (
                ""
                if request.stage in {"agent", "rerun"}
                else request.task_id
            ),
            "PTDBENCH_EVALUATION_LEVEL": request.evaluation_level,
            "PTDBENCH_TASK_ROOT": task_root,
            "PTDBENCH_VISIBLE_TASK": visible,
            "PTDBENCH_WORKSPACE": workspace,
            "PTDBENCH_ORACLE": oracle,
            "PTDBENCH_VERIFIER": verifier,
            "PTDBENCH_RESULTS": results,
            "PTDBENCH_STAGE": request.stage,
            "PTDBENCH_ENVIRONMENT": (
                self.path(request, "environment") if request.host_environment else ""
            ),
            "PTDBENCH_PREPARED": (
                self.path(request, "prepared") if request.host_prepared else ""
            ),
            # 两个下载缓存根。给空字符串表示"这次看不到"，和 prepared 一致——
            # agent 阶段总是空的，见 sees_download_cache()。
            "PTDBENCH_MODELS": (
                self.path(request, "models")
                if request.host_models and sees_download_cache(request)
                else ""
            ),
            "PTDBENCH_DATASETS": (
                self.path(request, "datasets")
                if request.host_datasets and sees_download_cache(request)
                else ""
            ),
        }
        env.update(self._network_policy_env(request))
        return env

    @staticmethod
    def _network_policy_env(request: SandboxRequest) -> Dict[str, str]:
        """离线开关，按阶段给。

        这些变量以前烤在镜像的 ENV 里，现在不行了：prepare 是唯一**应该**联网的阶段，
        而镜像级的 ENV 对三个阶段一视同仁，等于让 prepare 自己关掉自己要用的网络。

        agent 和 verifier 反过来：任何一次 silent 的联网下载，都可能把评测跑在一份
        没人校验过的权重上，还看不出来。所以这里把它们关死，任务要的数据必须是
        prepare 阶段交付的那份。rerun 与它们同一边：实验必须用 prepare 交付的
        那份资产重跑，而且它的输出就是判分输入，更不能联网。
        """
        if request.stage == "prepare":
            # prepare 也不该往 W&B 发东西——它只是在下载。
            return {"WANDB_MODE": "offline", "WANDB_SILENT": "true"}
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            # 有些框架在 WANDB_MODE=offline 下仍然会因为没有 key 而报错退出。
            "WANDB_API_KEY": "ptdbench-offline",
            "WANDB_SILENT": "true",
        }

    def _merged_spec(self, request: SandboxRequest, spec: CommandSpec) -> CommandSpec:
        """合并标准变量和调用方变量，避免 agent 重复处理路径。"""
        env = self.standard_env(request)
        # 任务声明的 env（metadata.environment.env）排在标准变量之后、调用方之前：
        # 它比框架默认值更具体，但不该盖掉调用方为这一次运行显式指定的东西。
        # PTDBENCH_* 不可能出现在这里，EnvironmentPlan 已经挡掉了。
        env.update({str(key): str(value) for key, value in request.task_env.items()})
        env.update({str(key): str(value) for key, value in request.env.items()})
        env.update({str(key): str(value) for key, value in spec.env.items()})
        timeout_s = spec.timeout_s if spec.timeout_s is not None else request.timeout_s
        return replace(spec, env=env, timeout_s=timeout_s)
