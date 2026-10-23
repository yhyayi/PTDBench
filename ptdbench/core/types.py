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
    environment: "EnvironmentPlan" = field(default_factory=lambda: EnvironmentPlan())

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


# 容器里三个阶段共享的目录。prepare 阶段可写 prepared，agent / verifier 阶段只读。
# 与 docker/ptdbench-activate.sh 和 docker/ptdbench_fetch.py 的默认值必须一致。
PREPARED_DIR = "/ptdbench/prepared"
# 下载缓存，按资产种类分两个根：宿主机上就是仓库根的 models/ 和 datasets/。
# 分开不是为了好看——一个显式的、跟仓库并排的目录漏挂时一眼看得出来，而藏在
# work_dir 底下的单个 cache 漏挂了只会安静地把几个 G 留在容器可写层。
# prepare 阶段可写，agent / verifier 只读。
MODELS_DIR = "/ptdbench/models"
DATASETS_DIR = "/ptdbench/datasets"
# 任务的 environment/ 目录挂进来的位置。三个阶段都只读——prepare.sh 是环境说明，
# 不是答案，但也没有哪个阶段有理由改仓库里的任务定义。
PTDBENCH_ENVIRONMENT_DIR = "/ptdbench/environment"
# 仓库 docker/ 目录（见 DOCKER_DIR）在 prepare 容器里挂进来的位置。只读、仅 prepare 阶段。
# prepare 跑 prepare.sh 前先 `bash $TOOLING_DIR/install-ptdbench-tooling.sh`，把 fetch/
# dataset 工具按仓库当前版本装进 /usr/local，避免用到烤进镜像的旧副本。
TOOLING_DIR = "/ptdbench/tooling"
# 这个安装脚本在仓库 docker/ 里，挂进来后的容器路径。
TOOLING_INSTALLER = f"{TOOLING_DIR}/install-ptdbench-tooling.sh"

# task package 里的约定文件名。写成常量是因为 loader 和 evaluator 都要用，而
# "默认值散落多处"正是 schema 漂移的起点。
DEFAULT_PREPARE = "environment/prepare.sh"

# 有哪些环境，唯一事实来源是 docker/images/ 下的 <name>.Dockerfile——镜像就是照它
# 构建的。不在这里再抄一份常量：抄一份就会漂，而漂的结果是任务指定了一个没有
# Dockerfile 的环境，直到评测跑起来才发现。
_REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = _REPO_ROOT / "docker"
IMAGES_DIR = DOCKER_DIR / "images"
IMAGE_DOCKERFILE_SUFFIX = ".Dockerfile"
# 镜像 tag 写在 Dockerfile 头部的 `# ptdbench-image: <tag>` 一行里，例如
# ptdbench/verl:vllm0.17-torch2.10-cu129。tag 描述的是栈（框架、推理引擎、CUDA
# 版本），多个任务共用；不从内容派生 hash。升级栈就改 tag，只改工具脚本则保持
# tag 不变、显式跑 `ptdbench build-images --rebuild`。
IMAGE_TAG_HEADER = "# ptdbench-image:"

# 下载缓存在宿主机上的默认位置。跟仓库并排而不是藏在 run 目录里，是因为 run 目录
# 跑完就删，而这两个目录的全部意义就是活得比 run 长——下一个 run、下一个任务都要
# 命中它们。默认值放在这里而不是各入口各写一份，cli 和 harbor 才不会漂。
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
DEFAULT_DATASETS_DIR = _REPO_ROOT / "datasets"


def known_images() -> frozenset:
    """docker/images/ 下有 Dockerfile 的环境名；拿不到目录时返回空集合。

    空集合表示"无法判断"，不是"一个都没有"。装成 wheel 之后 `docker/` 不一定跟着
    走，那时候校验环境名这件事只能推迟到构建镜像那一步。
    """
    if not IMAGES_DIR.is_dir():
        return frozenset()
    return frozenset(
        item.name[: -len(IMAGE_DOCKERFILE_SUFFIX)]
        for item in IMAGES_DIR.iterdir()
        if item.is_file()
        and item.name.endswith(IMAGE_DOCKERFILE_SUFFIX)
        and not item.name.startswith(".")
    )


@dataclass(frozen=True)
class EnvironmentImage:
    """一个环境对应的镜像：docker/images/<name>.Dockerfile，tag 由其头部声明。

    构建上下文是整个 docker/ 目录，因为每个镜像都要 COPY 同一组工具脚本
    （ptdbench-activate.sh、ptdbench_fetch.py 等），见 docker/.dockerignore。
    docker 评测、build-images 和 Harbor 的 `docker_image` 都用同一个 tag。
    """

    name: str

    @property
    def tag(self) -> str:
        """Dockerfile 头部 `# ptdbench-image: <tag>` 声明的 tag。

        找不到就报错而不是回退到某个默认值：回退会让两个环境悄悄共用一个 tag，
        后构建的那个覆盖先构建的。
        """
        try:
            text = self.dockerfile.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"cannot read {self.dockerfile} to resolve the image tag of {self.name!r}"
            ) from exc
        for line in text.splitlines():
            if line.startswith(IMAGE_TAG_HEADER):
                tag = line[len(IMAGE_TAG_HEADER):].strip()
                if tag:
                    return tag
        raise RuntimeError(
            f"{self.dockerfile} has no '{IMAGE_TAG_HEADER} <tag>' header line"
        )

    @property
    def dockerfile(self) -> Path:
        return IMAGES_DIR / f"{self.name}{IMAGE_DOCKERFILE_SUFFIX}"

    @property
    def context(self) -> Path:
        return DOCKER_DIR


@dataclass(frozen=True)
class EnvironmentPlan:
    """任务的 `environment` 段解析结果。

    `image` 是环境名，对应 docker/images/<image>.Dockerfile，多个任务共用同一个镜像。
    `prepare` 是跑训练前要准备的东西（下模型、下数据集、写环境变量），默认
    environment/prepare.sh。`env` 是任务自己要导出的变量。
    """

    image: Optional[str] = None
    prepare: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(self.env))
        known = known_images()
        if self.image is not None and known and self.image not in known:
            raise ValueError(
                f"environment.image must be one of {', '.join(sorted(known))}, "
                f"got {self.image!r}; add docker/images/{self.image}"
                f"{IMAGE_DOCKERFILE_SUFFIX} if the task really needs a new one"
            )
        for key, value in self.env.items():
            if not key or not isinstance(key, str):
                raise ValueError("environment.env keys must be non-empty strings")
            if not isinstance(value, str):
                # YAML 里 `CUDA_VISIBLE_DEVICES: 0,1` 会被读成字符串，但 `: 0` 是 int。
                # 悄悄 str() 的话，0 和 "0" 的区别就再也看不见了。
                raise ValueError(
                    f"environment.env[{key}] must be a string; quote it in YAML"
                )
        # PTDBENCH_* 是 harness 自己的通道（stage / task_id / 挂载路径等）。
        # 让任务覆盖它们，等于让任务改写评测框架对自己的判断。
        reserved = sorted(key for key in self.env if key.startswith("PTDBENCH_"))
        if reserved:
            raise ValueError(
                "environment.env must not set PTDBENCH_* variables: "
                + ", ".join(reserved)
            )

    @property
    def environment_image(self) -> Optional[EnvironmentImage]:
        return None if self.image is None else EnvironmentImage(self.image)


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
    work_dir: Optional[Path] = None
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    debian_mirror: Optional[str] = None
    huggingface_mirror: Optional[str] = None
    verifier_env: Mapping[str, str] = field(default_factory=dict)
    # 宿主机上的共享下载缓存，分别挂到容器的 /ptdbench/models 和 /ptdbench/datasets。
    # prepare.sh 下的模型和数据集都落在这里，所以第二个任务要同一个模型时不会再下
    # 一遍，重建镜像也不会。留空则用仓库根的 models/ 和 datasets/。
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.debian_mirror is not None:
            object.__setattr__(self, "debian_mirror", _normalize_plain_url(self.debian_mirror))
        if self.huggingface_mirror is not None:
            object.__setattr__(
                self,
                "huggingface_mirror",
                _normalize_plain_url(self.huggingface_mirror),
            )
        object.__setattr__(self, "verifier_env", _normalize_env(self.verifier_env))
        # 两个缓存根总是有值：调用方没给就落到仓库根。下游因此不必各自再写一遍
        # "没给怎么办"，挂载表也就没有"这次没有缓存"这种分支。
        for name, fallback in (
            ("models_dir", DEFAULT_MODELS_DIR),
            ("datasets_dir", DEFAULT_DATASETS_DIR),
        ):
            given = getattr(self, name)
            resolved = (Path(given) if given is not None else fallback).expanduser().resolve()
            object.__setattr__(self, name, resolved)


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
    # solution 层的系统重跑：agent 交 report 之后、判分之前 harness 跑的那一遍
    # 实验。issue 层和没有 run_spec.json 的任务为 None。放在末尾带默认值，
    # 老的构造点（harbor、测试）按位置传参时不用改也能继续工作。
    rerun: Optional[CommandResult] = None

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
            "rerun": None if self.rerun is None else self.rerun.to_dict(),
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
