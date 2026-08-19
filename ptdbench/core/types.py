from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit


@dataclass(frozen=True)
class EvaluationPaths:
    """A task evaluation's public instruction and hidden grading paths."""

    level: str
    root: Path
    instruction_md: Path
    oracle: Path
    verifier: Path
    verifier_test: Path
    verifier_tests: Optional[Path]
    verifier_resources: Optional[Path]


@dataclass(frozen=True)
class TaskPaths:
    """一个 task package 解析后的文件系统布局。"""

    root: Path
    readme_md: Optional[Path]
    metadata_yaml: Path
    workspace: Path
    environment: Path
    evaluations: Dict[str, EvaluationPaths]


@dataclass(frozen=True)
class ValidationIssue:
    """validator 产出的一条目录或 schema 问题。"""

    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    """汇总所有校验问题，方便贡献者一次性修复。"""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, severity: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity=severity, path=path, message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [
                {"severity": item.severity, "path": item.path, "message": item.message}
                for item in self.issues
            ],
        }


@dataclass(frozen=True)
class TaskPackage:
    """加载后的 task：路径、metadata 和校验报告。"""

    paths: TaskPaths
    metadata: Dict[str, Any]
    report: ValidationReport

    @property
    def task_id(self) -> str:
        return self.paths.root.name

    @property
    def evaluation_levels(self) -> Sequence[str]:
        return tuple(self.paths.evaluations)


@dataclass(frozen=True)
class AgentSpec:
    """统一后的 agent 配置。"""

    kind: str = "codex"
    command: Optional[str] = None
    model: Optional[str] = None
    judge_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    network: str = "bridge"
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_args", tuple(self.extra_args))
        object.__setattr__(self, "env", _normalize_env(self.env))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "command": self.command,
            "model": self.model,
            "judge_model": self.judge_model,
            "reasoning_effort": self.reasoning_effort,
            "base_url": self.base_url,
            "api_key": None if self.api_key is None else "<redacted>",
            "extra_args": list(self.extra_args),
            "network": self.network,
            "env": _redact_env(dict(self.env)),
        }


@dataclass(frozen=True)
class CommandResult:
    """一次命令执行的统一结果，agent 和 verifier 共用。"""

    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    cwd: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "cwd": self.cwd,
        }


@dataclass(frozen=True)
class StageArtifact:
    """给后续评测阶段预留的 artifact 记录结构。"""

    name: str
    path: Path
    command: Optional[CommandResult] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    """调用方运行时配置，与 task package 自身数据分开。"""

    agent: Optional[AgentSpec] = None
    agent_cmd: Optional[str] = None
    verifier_cmd: Optional[str] = None
    backend: str = "docker"
    verifier_network: str = "bridge"
    timeout_s: Optional[int] = None
    keep_artifacts: bool = False
    image_tag_prefix: str = "ptdbench"
    work_dir: Optional[Path] = None
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    debian_mirror: Optional[str] = None
    verifier_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.debian_mirror is not None:
            object.__setattr__(self, "debian_mirror", _normalize_plain_url(self.debian_mirror))
        object.__setattr__(self, "verifier_env", _normalize_env(self.verifier_env))


@dataclass(frozen=True)
class EvaluationResult:
    """PTDBench 输出的最终评测摘要。"""

    task_id: str
    evaluation_level: str
    passed: bool
    score: Optional[float]
    metrics: Dict[str, Any]
    report_dir: Path
    agent_spec: Optional[AgentSpec]
    agent_setup: Optional[CommandResult]
    agent: Optional[CommandResult]
    verifier: Optional[CommandResult]
    workspace_before_hash: str
    workspace_after_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "evaluation_level": self.evaluation_level,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "report_dir": str(self.report_dir),
            "agent_spec": None if self.agent_spec is None else self.agent_spec.to_dict(),
            "agent_setup": None if self.agent_setup is None else self.agent_setup.to_dict(),
            "agent": None if self.agent is None else self.agent.to_dict(),
            "verifier": None if self.verifier is None else self.verifier.to_dict(),
            "workspace_before_hash": self.workspace_before_hash,
            "workspace_after_hash": self.workspace_after_hash,
            "metadata": self.metadata,
        }


def _normalize_plain_url(value: str) -> str:
    url = str(value).rstrip("/")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "|" in url
        or any(char.isspace() for char in url)
    ):
        raise ValueError("URL must be a plain HTTP(S) base URL")
    return url


def _normalize_env(env: Mapping[str, str]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in env.items()}


def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in env.items():
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
