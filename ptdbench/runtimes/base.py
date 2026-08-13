from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..core.types import CommandResult


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
    """一次 agent 或 verifier 运行需要的宿主机资源。"""

    stage: str
    case_id: str
    host_case_root: Path
    host_visible_case: Path
    host_workspace: Path
    host_results: Path
    evaluation_level: str = "solution"
    host_oracle: Optional[Path] = None
    host_verifier: Optional[Path] = None
    workdir: str = "workspace"
    env: Mapping[str, str] = field(default_factory=dict)
    network: str = "none"
    timeout_s: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stage not in {"agent", "verifier"}:
            raise ValueError("sandbox stage must be 'agent' or 'verifier'")
        if self.workdir not in {"workspace", "case_root"}:
            raise ValueError("sandbox workdir must be 'workspace' or 'case_root'")


class SandboxRuntime(ABC):
    """运行平台的统一接口。

    agent 和 evaluator 只依赖这个接口，不直接依赖 Docker 或本机进程。
    Harbor、云沙盒等后端可以实现同一组方法。
    """

    name: str

    @property
    @abstractmethod
    def needs_environment_image(self) -> bool:
        """该平台是否需要通过 environment/Dockerfile 准备镜像。"""

    @abstractmethod
    def build_environment(
        self,
        environment_dir: Path,
        *,
        dockerfile: str = "Dockerfile",
        tag_prefix: str = "ptdbench",
    ) -> EnvironmentArtifact:
        """准备环境并返回可以复用的 artifact。"""

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
        """生成 agent/verifier 都可以依赖的 PTDBench 环境变量。"""
        visible_case = self.path(request, "visible_case")
        workspace = self.path(request, "workspace")
        results = self.path(request, "results")
        if request.stage == "agent":
            case_root = visible_case
            visible = visible_case
            oracle = ""
            verifier = ""
        else:
            case_root = self.path(request, "case_root")
            visible = ""
            oracle = self.path(request, "oracle")
            verifier = self.path(request, "verifier")
        return {
            "PTDBENCH_CASE_ID": request.case_id,
            "PTDBENCH_EVALUATION_LEVEL": request.evaluation_level,
            "PTDBENCH_CASE_ROOT": case_root,
            "PTDBENCH_VISIBLE_CASE": visible,
            "PTDBENCH_WORKSPACE": workspace,
            "PTDBENCH_ORACLE": oracle,
            "PTDBENCH_VERIFIER": verifier,
            "PTDBENCH_RESULTS": results,
            "PTDBENCH_STAGE": request.stage,
        }

    def _merged_spec(self, request: SandboxRequest, spec: CommandSpec) -> CommandSpec:
        """合并标准变量和调用方变量，避免 agent 重复处理路径。"""
        env = self.standard_env(request)
        env.update({str(key): str(value) for key, value in request.env.items()})
        env.update({str(key): str(value) for key, value in spec.env.items()})
        timeout_s = spec.timeout_s if spec.timeout_s is not None else request.timeout_s
        return replace(spec, env=env, timeout_s=timeout_s)
