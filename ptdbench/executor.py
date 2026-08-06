from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .filesystem import tree_digest
from .types import CommandResult


@dataclass(frozen=True)
class CommandSpec:
    # 命令构造保持显式：调用方可以按任务习惯选择 argv 或 shell 字符串。
    argv: Optional[Sequence[str]] = None
    shell: Optional[str] = None
    cwd: Optional[Path] = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: Optional[int] = None

    def to_argv(self) -> List[str]:
        if self.argv is not None:
            return list(self.argv)
        if self.shell is not None:
            return ["sh", "-lc", self.shell]
        raise ValueError("command spec needs argv or shell")


class LocalExecutor:
    def run(self, spec: CommandSpec) -> CommandResult:
        # local backend 只用于快速 smoke test，不承担隔离或安全边界。
        argv = spec.to_argv()
        cwd = str(spec.cwd or Path.cwd())
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in spec.env.items()})
        start = time.monotonic()
        timed_out = False
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
        end = time.monotonic()
        return CommandResult(
            argv=argv,
            returncode=proc.returncode if proc.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=end - start,
            cwd=cwd,
        )


@dataclass(frozen=True)
class DockerMount:
    host_path: Path
    container_path: str
    read_only: bool = False

    def to_arg(self) -> str:
        # Docker 的 -v 参数需要把一次挂载拼成一个字符串。
        suffix = ":ro" if self.read_only else ""
        return f"{self.host_path}:{self.container_path}{suffix}"


@dataclass(frozen=True)
class DockerImageSpec:
    context_dir: Path
    dockerfile: str = "Dockerfile"
    build_args: Mapping[str, str] = field(default_factory=dict)
    tag_prefix: str = "ptdbench"

    @property
    def digest(self) -> str:
        return tree_digest(self.context_dir)

    @property
    def tag(self) -> str:
        # image tag 带上环境目录内容哈希；environment 变了就会自动换 tag。
        base = self.context_dir.name.replace("_", "-")
        return f"{self.tag_prefix}-{base}-{self.digest[:12]}"


class DockerExecutor:
    def build(self, image: DockerImageSpec) -> str:
        # 相同 environment 哈希的镜像已经存在时，直接复用。
        tag = image.tag
        if self._image_exists(tag):
            return tag
        cmd = ["docker", "build"]
        for key, value in sorted(image.build_args.items()):
            cmd.extend(["--build-arg", f"{key}={value}"])
        cmd.extend([
            "-f",
            str(image.context_dir / image.dockerfile),
            "-t",
            tag,
            str(image.context_dir),
        ])
        subprocess.run(cmd, check=True)
        return tag

    def run(
        self,
        tag: str,
        spec: CommandSpec,
        mounts: Sequence[DockerMount],
        workdir: str = "/workspace",
        network: str = "none",
    ) -> CommandResult:
        # 容器只能看到 evaluator 明确挂载的目录；agent/verifier 的可见性
        # 就在这里收口。
        argv = spec.to_argv()
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-w",
            workdir,
        ]
        if network:
            # agent 往往需要访问模型 API；verifier 默认仍然用 none 断网。
            docker_cmd.append(f"--network={network}")
        for mount in mounts:
            docker_cmd.extend(["-v", mount.to_arg()])
        for key, value in sorted(spec.env.items()):
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.append(tag)
        docker_cmd.extend(argv)
        start = time.monotonic()
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
        end = time.monotonic()
        return CommandResult(
            argv=docker_cmd,
            returncode=proc.returncode if proc.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=end - start,
            cwd=workdir,
        )

    def _image_exists(self, tag: str) -> bool:
        # build 前先轻量检查镜像缓存是否存在。
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
