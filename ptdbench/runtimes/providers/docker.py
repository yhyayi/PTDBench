from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ...core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    TOOLING_DIR,
    CommandResult,
    EnvironmentImage,
)
from ..base import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
    sees_download_cache,
    sees_environment_dir,
)
from ..factory import register_runtime


# ray + sglang 的 worker 之间通过 /dev/shm 传张量，NCCL 也在那里开缓冲区。Docker
# 的默认值是 64MB，第一次 rollout 就会爆，而报出来的是一个看不懂的 bus error。
DEFAULT_SHM_SIZE = "16g"


@dataclass(frozen=True)
class DockerMount:
    """一个宿主机到容器的目录挂载。"""

    host_path: Path
    container_path: str
    read_only: bool = False

    def to_arg(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.host_path}:{self.container_path}{suffix}"


@register_runtime("docker")
class DockerRuntime(SandboxRuntime):
    """Docker provider，集中处理镜像、挂载和网络。"""

    name = "docker"

    @property
    def needs_environment_image(self) -> bool:
        return True

    def ensure_environment(
        self, image: EnvironmentImage, *, rebuild: bool = False
    ) -> EnvironmentArtifact:
        """镜像已存在就直接用，否则照 docker/images/<name>.Dockerfile 构建。

        tag 是固定的，所以"已存在"不代表"和 Dockerfile 一致"：改了 Dockerfile 或
        docker/ 下的工具脚本之后要 rebuild（`ptdbench build-images --rebuild`）。
        rebuild 走 docker 的层缓存，没变的层不会重跑。
        """
        if rebuild or not self._image_exists(image.tag):
            if not image.dockerfile.is_file():
                raise RuntimeError(
                    f"image {image.tag!r} not found locally and {image.dockerfile} "
                    "does not exist to build it from"
                )
            # 构建日志是给人看的进度，不是命令结果；走 stderr，stdout 留给 CLI 的输出。
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    str(image.dockerfile),
                    "-t",
                    image.tag,
                    str(image.context),
                ],
                stdout=sys.stderr,
                check=True,
            )
        return EnvironmentArtifact(backend=self.name, reference=image.tag)

    def path(self, request: SandboxRequest, name: str) -> str:
        paths = {
            "task_root": "/task",
            "visible_task": "/task",
            "workspace": "/workspace",
            "results": "/results",
            "environment": PTDBENCH_ENVIRONMENT_DIR,
            "prepared": PREPARED_DIR,
            "models": MODELS_DIR,
            "datasets": DATASETS_DIR,
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
        argv.extend(self._gpu_args(request))
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
    def _gpu_args(request: SandboxRequest) -> list[str]:
        """把任务声明的 NVIDIA_VISIBLE_DEVICES 翻译成 docker 的 `--gpus`。

        镜像侧（CUDA 基础镜像）一直假设 libcuda.so.1 是 `--gpus` 注入的，缺的只是
        这一半。把变量 `-e`
        进容器并不会让容器看见 GPU：宿主机的 default runtime 是 runc，只有
        `--gpus` 才会让 nvidia 的 hook 真正注入设备节点和 libcuda.so.1。

        没有声明就一个 flag 都不加，纯 CPU 的任务和阶段（判分几乎都是）行为与
        接线前完全相同。`none` / `void` 是 NVIDIA_VISIBLE_DEVICES 自己的"不要
        GPU"取值，按同一条处理。

        PTDBENCH_DOCKER_GPUS 是宿主机侧的覆盖，沿用 _proxy_env 的 opt-in 惯例：
        一台只有 2 张卡的机器不该因为任务卡片写着 8 张就跑不起来。

        prepare は模型・データセットのダウンロードだけで GPU は不要。`docker create`
        は --gpus を受け付けるが `docker start` 時に NVIDIA runtime hook が実際の
        デバイスノードをバインドしようとして失敗する環境がある。
        """
        if request.stage == "prepare":
            return []
        spec = os.environ.get("PTDBENCH_DOCKER_GPUS", "").strip()
        if not spec:
            spec = str(request.task_env.get("NVIDIA_VISIBLE_DEVICES", "")).strip()
        if not spec or spec in {"none", "void"}:
            return []
        # `all` 和 `device=...` 本身就是 --gpus 的语法；任务卡片里写的是裸设备表
        # （"0,1,2,3"），要包一层才能被 docker 认出来。
        if spec != "all" and not spec.startswith("device="):
            spec = f"device={spec}"
        shm = os.environ.get("PTDBENCH_DOCKER_SHM_SIZE", "").strip() or DEFAULT_SHM_SIZE
        return ["--gpus", spec, f"--shm-size={shm}"]

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
        """每个阶段挂什么，以及谁能写。

        三张表的差别就是这套设计的访问控制，值得挨条说清楚：

        * **workspace**：prepare 和 agent 可写，verifier 只读。prepare 要写是因为
          权重就下在任务脚本引用的那个相对路径上（`./Qwen2.5-0.5B-Instruct`）；
          verifier 只读是因为判分不能改现场。
        * **/task**：prepare 和 agent 挂的是 visible_task（只有 instruction），
          verifier 挂的是整个 evaluation 目录（含 oracle 和判分脚本）。这条是"别剧透"——
          prepare 跑的是任务自带的脚本，没有理由让它读到答案。
        * **prepared**：prepare 可写，后两个阶段只读。verifier 要读 prepare 交付的
          模型路径，但不能读到一份 agent 改过的——否则判分依据是可以被应试者编辑的。
        * **models / datasets**：prepare 可写，后两个阶段只读。见
          SandboxRequest.host_models。
        """
        if request.stage == "prepare":
            mounts = [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_task, "/task", read_only=True),
                DockerMount(request.host_prepared, PREPARED_DIR),
            ]
            # 仓库 docker/ 只读挂进来：prepare 跑脚本前先用它装最新的 fetch/dataset 工具
            # （见 TOOLING_DIR / install-ptdbench-tooling.sh）。只有 prepare 需要。
            if request.host_tooling is not None:
                mounts.append(
                    DockerMount(request.host_tooling, TOOLING_DIR, read_only=True)
                )
        elif request.stage == "agent":
            mounts = [
                DockerMount(request.host_workspace, "/workspace"),
                DockerMount(request.host_visible_task, "/task", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        else:
            mounts = [
                DockerMount(request.host_workspace, "/workspace", read_only=True),
                DockerMount(request.host_task_root, "/task", read_only=True),
                DockerMount(request.host_results, "/results"),
            ]
        if request.host_environment is not None and sees_environment_dir(request):
            # 只读：没有任何一个阶段有理由改仓库里的任务定义。
            mounts.append(
                DockerMount(request.host_environment, PTDBENCH_ENVIRONMENT_DIR, read_only=True)
            )
        if request.stage != "prepare" and request.host_prepared is not None:
            mounts.append(DockerMount(request.host_prepared, PREPARED_DIR, read_only=True))
        # 下载缓存：prepare 写，verifier 只读，agent 看不到。判据见
        # sees_download_cache()——环境变量那边用的是同一个。
        if sees_download_cache(request):
            read_only = request.stage != "prepare"
            for host, container in (
                (request.host_models, MODELS_DIR),
                (request.host_datasets, DATASETS_DIR),
            ):
                if host is not None:
                    mounts.append(DockerMount(host, container, read_only=read_only))
        return tuple(mounts)

    @staticmethod
    def _image_exists(tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
