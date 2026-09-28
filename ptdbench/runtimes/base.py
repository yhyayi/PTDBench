from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..core.types import CommandResult, EnvironmentImage


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
            return ["sh", "-lc", self.shell]
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
    """一次 prepare / agent / verifier 运行需要的宿主机资源。"""

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
        if self.stage not in {"prepare", "agent", "verifier"}:
            raise ValueError("sandbox stage must be 'prepare', 'agent' or 'verifier'")
        if self.workdir not in {"workspace", "task_root", "prepared"}:
            raise ValueError(
                "sandbox workdir must be 'workspace', 'task_root' or 'prepared'"
            )
        if self.stage == "prepare" and self.host_prepared is None:
            # prepare 阶段没有落点就等于白跑：脚本会成功，什么也没留下，直到 agent
            # 阶段找不到模型才暴露，而那时报的是任务的错。
            raise ValueError("the prepare stage needs host_prepared")


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
            # prepare 和 agent 都只看得到 visible_task：prepare 跑的是任务自己提供的
            # 脚本，没有理由让它读到 oracle 或判分脚本。
            task_root = visible_task
            visible = visible_task
            oracle = ""
            verifier = ""
        env = {
            "PTDBENCH_TASK_ID": request.task_id,
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
            # 两个下载缓存根，三个阶段都导出。给空字符串表示"这次没有配"，和
            # prepared 的处理一致。
            "PTDBENCH_MODELS": (
                self.path(request, "models") if request.host_models else ""
            ),
            "PTDBENCH_DATASETS": (
                self.path(request, "datasets") if request.host_datasets else ""
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
        prepare 阶段交付的那份。
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
