from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ...core.types import CommandResult
from ...utils.filesystem import tree_digest
from ..base import CommandSpec, EnvironmentArtifact, SandboxRequest, SandboxRuntime
from ..factory import register_runtime


@dataclass(frozen=True)
class DockerMount:
    """一个宿主机到容器的目录挂载。"""

    host_path: Path
    container_path: str
    read_only: bool = False

    def to_arg(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.host_path}:{self.container_path}{suffix}"


@dataclass(frozen=True)
class DockerImageSpec:
    """Docker 镜像构建参数。"""

    context_dir: Path
    dockerfile: str = "Dockerfile"
    build_args: Mapping[str, str] = field(default_factory=dict)
    tag_prefix: str = "ptdbench"

    @property
    def digest(self) -> str:
        return tree_digest(self.context_dir)

    @property
    def tag(self) -> str:
        base = self.context_dir.name.replace("_", "-")
        return f"{self.tag_prefix}-{base}-{self.digest[:12]}"


@register_runtime("docker")
class DockerRuntime(SandboxRuntime):
    """Docker provider，集中处理镜像、挂载和网络。"""

    name = "docker"

    @property
    def needs_environment_image(self) -> bool:
        return True

    def build_environment(
        self,
        environment_dir: Path,
        *,
        dockerfile: str = "Dockerfile",
        tag_prefix: str = "ptdbench",
    ) -> EnvironmentArtifact:
        image = DockerImageSpec(
            context_dir=environment_dir,
            dockerfile=dockerfile,
            tag_prefix=tag_prefix,
        )
        return EnvironmentArtifact(backend=self.name, reference=self._build(image))

    def path(self, request: SandboxRequest, name: str) -> str:
        paths = {
            "case_root": "/case",
            "visible_case": "/case",
            "workspace": "/workspace",
            "results": "/results",
            "oracle": "/case/oracle",
            "verifier": "/case/verifier",
        }
        try:
            return paths[name]
        except KeyError as exc:
            raise ValueError(f"unknown sandbox path: {name}") from exc

    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        if artifact is None or artifact.backend != self.name or not artifact.reference:
            raise ValueError("docker runtime needs a built environment artifact")
        spec = self._merged_spec(request, spec)
        spec = replace(spec, cwd=Path(self.path(request, request.workdir)))
        mounts = self._mounts(request)
        docker_argv = ["docker", "run", "--rm", "-w", str(spec.cwd)]
        if request.network:
            docker_argv.append(f"--network={request.network}")
        for mount in mounts:
            docker_argv.extend(["-v", mount.to_arg()])
        for key, value in sorted(spec.env.items()):
            docker_argv.extend(["-e", f"{key}={value}"])
        docker_argv.append(artifact.reference)
        docker_argv.extend(spec.to_argv())

        start = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            docker_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        return CommandResult(
            argv=docker_argv,
            returncode=process.returncode if process.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            cwd=str(spec.cwd),
        )

    def _mounts(self, request: SandboxRequest) -> Sequence[DockerMount]:
        if request.stage == "agent":
            return [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_case, "/case", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        return [
            DockerMount(request.host_workspace, "/workspace", read_only=True),
            DockerMount(request.host_case_root, "/case", read_only=True),
            DockerMount(request.host_results, "/results"),
        ]

    def _build(self, image: DockerImageSpec) -> str:
        """构建镜像；相同 digest 的镜像直接使用本地缓存。"""
        if self._image_exists(image.tag):
            return image.tag
        command = ["docker", "build"]
        for key, value in sorted(image.build_args.items()):
            command.extend(["--build-arg", f"{key}={value}"])
        command.extend(
            ["-f", str(image.context_dir / image.dockerfile), "-t", image.tag, str(image.context_dir)]
        )
        subprocess.run(command, check=True)
        return image.tag

    @staticmethod
    def _image_exists(tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
