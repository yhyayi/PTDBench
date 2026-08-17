from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ...core.types import CommandResult
from ...utils.filesystem import tree_digest
from ..base import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
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
            "task_root": "/task",
            "visible_task": "/task",
            "workspace": "/workspace",
            "results": "/results",
        }
        selected_paths = {
            "oracle": request.host_oracle,
            "verifier": request.host_verifier,
        }
        if name in selected_paths and selected_paths[name] is not None:
            selected = selected_paths[name]
            try:
                relative = selected.resolve().relative_to(request.host_task_root.resolve())
            except ValueError as exc:
                raise ValueError(f"{name} must stay under the selected task root") from exc
            return f"/task/{relative.as_posix()}"
        paths.update({"oracle": "/task/oracle", "verifier": "/task/verifier"})
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
        sandbox = self.create_sandbox(request, artifact=artifact)
        try:
            return self.exec(sandbox, spec)
        finally:
            self.close_sandbox(sandbox)

    def create_sandbox(
        self,
        request: SandboxRequest,
        *,
        artifact: EnvironmentArtifact,
    ) -> SandboxInstance:
        if artifact.backend != self.name or not artifact.reference:
            raise ValueError("docker runtime needs a built environment artifact")
        container_name = f"ptdbench-{request.stage}-{uuid.uuid4().hex[:12]}"
        env = self._proxy_env(request, self._merged_spec(request, CommandSpec()).env)
        argv = [
            "docker",
            "create",
            "--name",
            container_name,
            "-w",
            self.path(request, request.workdir),
        ]
        if request.network:
            argv.append(f"--network={request.network}")
        for mount in self._mounts(request):
            argv.extend(["-v", mount.to_arg()])
        env_args, process_env = self._docker_env(env)
        argv.extend(env_args)
        argv.extend(
            [
                "--entrypoint",
                "sh",
                artifact.reference,
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )
        subprocess.run(
            argv,
            check=True,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            subprocess.run(
                ["docker", "start", container_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            subprocess.run(["docker", "rm", "-f", container_name], check=False)
            raise
        return SandboxInstance(self.name, container_name, request)

    def exec(self, sandbox: SandboxInstance, spec: CommandSpec) -> CommandResult:
        if sandbox.backend != self.name:
            raise ValueError("sandbox belongs to a different runtime")
        request = sandbox.request
        merged = self._merged_spec(request, spec)
        merged = merged if request.stage != "agent" else self._with_env(
            merged,
            self._proxy_env(request, merged.env),
        )
        cwd = self.path(request, request.workdir)
        argv = ["docker", "exec", "-w", cwd]
        if merged.user is not None:
            argv.extend(["-u", str(merged.user)])
        env_args, process_env = self._docker_env(merged.env)
        argv.extend(env_args)
        argv.extend([sandbox.reference, *merged.to_argv()])
        return self._communicate(
            argv,
            timeout_s=merged.timeout_s,
            cwd=cwd,
            process_env=process_env,
            live_log_name=merged.live_log_name,
            live_log_to_stderr=merged.live_log_to_stderr,
            results_dir=request.host_results,
        )

    def close_sandbox(self, sandbox: SandboxInstance) -> None:
        if sandbox.backend != self.name:
            return
        subprocess.run(["docker", "rm", "-f", sandbox.reference], check=False)

    @staticmethod
    def _communicate(
        argv: list[str],
        *,
        timeout_s: Optional[int],
        cwd: str,
        process_env: Optional[dict[str, str]] = None,
        live_log_name: Optional[str] = None,
        live_log_to_stderr: bool = False,
        results_dir: Optional[Path] = None,
    ) -> CommandResult:
        start = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            argv,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if live_log_name and results_dir is not None:
            stdout, stderr, timed_out = DockerRuntime._stream_process(
                process,
                timeout_s=timeout_s,
                results_dir=results_dir,
                log_name=live_log_name,
                to_stderr=live_log_to_stderr,
            )
        else:
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
        return CommandResult(
            argv=DockerRuntime._redact_argv(argv),
            returncode=process.returncode if process.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            cwd=cwd,
        )

    @staticmethod
    def _stream_process(
        process: subprocess.Popen,
        *,
        timeout_s: Optional[int],
        results_dir: Path,
        log_name: str,
        to_stderr: bool = False,
    ) -> tuple[str, str, bool]:
        """Capture a process while making long setup commands observable."""
        results_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = results_dir / f"{log_name}.stdout.log"
        stderr_path = results_dir / f"{log_name}.stderr.log"
        captured = {"stdout": [], "stderr": []}

        def drain(stream, path: Path, key: str) -> None:
            with open(path, "w", encoding="utf-8") as log_file:
                for line in iter(stream.readline, ""):
                    captured[key].append(line)
                    log_file.write(line)
                    log_file.flush()
                    if to_stderr:
                        sys.stderr.write(line)
                        sys.stderr.flush()
            stream.close()

        threads = [
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout_path, "stdout"),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, stderr_path, "stderr"),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        for thread in threads:
            thread.join()
        return "".join(captured["stdout"]), "".join(captured["stderr"]), timed_out

    @staticmethod
    def _docker_env(env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
        argv: list[str] = []
        process_env = os.environ.copy()
        sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for key, value in sorted(env.items()):
            process_env[key] = value
            if any(token in key.upper() for token in sensitive):
                argv.extend(["-e", key])
            else:
                argv.extend(["-e", f"{key}={value}"])
        return argv, process_env

    @staticmethod
    def _with_env(spec: CommandSpec, env: Mapping[str, str]) -> CommandSpec:
        """Return a command with the supplied environment merged in."""
        return CommandSpec(
            argv=spec.argv,
            shell=spec.shell,
            cwd=spec.cwd,
            env=dict(env),
            timeout_s=spec.timeout_s,
            user=spec.user,
            live_log_name=spec.live_log_name,
            live_log_to_stderr=spec.live_log_to_stderr,
        )

    @staticmethod
    def _proxy_env(request: SandboxRequest, env: Mapping[str, str]) -> dict[str, str]:
        """Optionally expose a host-reachable proxy to task containers.

        Docker does not inherit the host's GUI/system proxy settings. The
        explicit PTDBENCH_DOCKER_PROXY opt-in keeps direct networking as the
        default and avoids forwarding a host localhost address by accident.
        """
        result = dict(env)
        proxy = os.environ.get("PTDBENCH_DOCKER_PROXY", "").strip()
        if proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                result.setdefault(key, proxy)
        no_proxy = os.environ.get("PTDBENCH_DOCKER_NO_PROXY", "").strip()
        if no_proxy:
            result.setdefault("NO_PROXY", no_proxy)
        return result

    @staticmethod
    def _redact_argv(argv: list[str]) -> list[str]:
        redacted = list(argv)
        sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for index, value in enumerate(redacted[:-1]):
            if value != "-e":
                continue
            assignment = redacted[index + 1]
            key = assignment.split("=", 1)[0].upper()
            if any(token in key for token in sensitive) and "=" in assignment:
                redacted[index + 1] = f"{assignment.split('=', 1)[0]}=<redacted>"
        return redacted

    def _mounts(self, request: SandboxRequest) -> Sequence[DockerMount]:
        if request.stage == "agent":
            return [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_task, "/task", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        return [
            DockerMount(request.host_workspace, "/workspace", read_only=True),
            DockerMount(request.host_task_root, "/task", read_only=True),
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
