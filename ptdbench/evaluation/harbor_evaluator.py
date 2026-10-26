from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..core.harbor_types import (
    HarborEvaluationConfig,
    HarborEvaluationReport,
    HarborRun,
    HarborTaskUnit,
    PreparedHarborTask,
    harbor_task_name,
)
from ..core.types import (
    DATASETS_DIR,
    DOCKER_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    CommandResult,
    EnvironmentImage,
    EvaluationConfig,
    EvaluationPaths,
    EvaluationResult,
    TaskPackage,
)
from ..runtimes.base import PTDBENCH_ACTIVATE_BODY
from ..tasks import SchemaError, load_task_package
from .evaluator import TaskEvaluator
from .run_spec import (
    REPORT_FILENAME,
    RunSpec,
    RunSpecError,
    load_run_spec,
)
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


# activate hook 的位置。以前烤在镜像的 /etc/profile.d 里，现在不再烤进镜像：
# 内容由 runtimes.base.PTDBENCH_ACTIVATE_BODY 这一处定义，prepare 阶段把它写进
# prepared 目录（见 HarborPreparer.prepare），容器里按 PREPARED_DIR 挂进来。
# Harbor 用非登录 `bash -c`，靠 BASH_ENV 指到这里自动 source。
ACTIVATE_HOOK = f"{PREPARED_DIR}/ptdbench-activate.sh"

# 不在本机跑容器的 Harbor 后端。prepare 阶段要的是任务镜像（x86-64 CUDA）里的
# slime / verl / Megatron 环境，而这些后端的算力在远端：prepare 由后端在远端执行
# （见 harbor.environments.ptdbench_prepare），本地既不跑 prepare 也不挂载。
_REMOTE_BACKENDS = frozenset({"cybertron"})


def _is_remote_backend(harbor_args: Sequence[str]) -> bool:
    """`-e/--env` 选中的后端是否在远端自己跑 prepare。

    `harbor run` 的参数是整体透传的，PTDBench 本来不解析它们；这里只窥探后端名，
    因为「prepare 在哪跑」「要不要挂载」「GPU 数从哪来」三件事都取决于它。
    """
    return HarborRunner._option_value(harbor_args, "-e", "--env") in _REMOTE_BACKENDS


class HarborAdapterError(RuntimeError):
    """PTDBench 与 Harbor 交界处的配置或结果错误。"""


class HarborTaskMaterializer:
    """把选定的 PTDBench evaluation 在线物化为一批 Harbor task。"""

    def materialize(
        self,
        units: Sequence[HarborTaskUnit],
        destination: Path,
        config: HarborEvaluationConfig,
    ) -> PreparedHarborTask:
        if not units:
            raise HarborAdapterError("at least one task is required")
        root = destination.resolve()
        # Harbor 的 -p/--path 接受"父目录下每个子目录是一个 task"，批量与单任务同构。
        tasks_dir = ensure_dir(root / "tasks")
        prepared = PreparedHarborTask(root=root, tasks_dir=tasks_dir, units=units)
        self._reject_duplicate_task_names(units)
        for unit in units:
            self._materialize_one(unit, prepared.task_dir(unit), config)
        return prepared

    @staticmethod
    def _reject_duplicate_task_names(units: Sequence[HarborTaskUnit]) -> None:
        """同名 task 会互相覆盖物化结果，也无法从 trial 反查来源。"""
        seen: Dict[str, Path] = {}
        for unit in units:
            name = unit.harbor_task_name
            previous = seen.get(name)
            if previous is not None:
                raise HarborAdapterError(
                    f"duplicate Harbor task name {name!r} from "
                    f"{previous} and {unit.package.paths.root}"
                )
            seen[name] = unit.package.paths.root

    def _materialize_one(
        self,
        unit: HarborTaskUnit,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        package = unit.package
        evaluation = unit.evaluation
        ensure_dir(task_dir)

        self._write_instruction(evaluation, task_dir)
        self._write_environment(unit, task_dir)
        self._write_tests(unit, task_dir)
        self._write_solution(evaluation, task_dir)
        self._write_task_config(unit, task_dir, config)
        self._write_manifest(package, evaluation, task_dir, config)

    @staticmethod
    def _write_instruction(evaluation: EvaluationPaths, task_dir: Path) -> None:
        instruction = evaluation.instruction_md.read_text(encoding="utf-8")
        write_text(task_dir / "instruction.md", instruction)

    def _write_environment(self, unit: HarborTaskUnit, task_dir: Path) -> None:
        """environment/ 只放 workspace 的内容，不放 Dockerfile。

        task.toml 声明了 `docker_image`，Harbor 直接起预构建镜像；而 environment/
        里没有 Dockerfile / docker-compose.yaml 时，Harbor 会在容器启动后把整个
        目录上传到 workdir（/workspace）。这正好就是 agent 要改的那份副本。

        有 staging 就用 prepare 写过的那份（权重可能就下在里面）；硬链接复制，
        几十 GB 的权重不会在磁盘上再占一份。
        """
        source = unit.workspace or unit.package.paths.workspace
        environment_dir = task_dir / "environment"
        for name in ("Dockerfile", "docker-compose.yaml"):
            if (source / name).exists():
                # 放进 environment/ 就会被 Harbor 当成构建定义，而不是上传的文件。
                raise HarborAdapterError(
                    f"workspace of {unit.package.task_id!r} contains {name}; Harbor "
                    "would build it instead of using the prebuilt image"
                )
        self._link_tree(source, environment_dir)
        if not any(environment_dir.iterdir()):
            # 空目录 Harbor 不会上传，/workspace 也就不存在；留一个占位文件。
            write_text(environment_dir / ".ptdbench-workspace", "")

    @staticmethod
    def _link_tree(source: Path, destination: Path) -> None:
        """尽量硬链接，跨文件系统时退回普通复制。"""

        def link_or_copy(src: str, dst: str) -> None:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        shutil.copytree(source, destination, symlinks=True, copy_function=link_or_copy)

    def _write_tests(self, unit: HarborTaskUnit, task_dir: Path) -> None:
        evaluation = unit.evaluation
        tests_dir = ensure_dir(task_dir / "tests")
        copy_tree(evaluation.verifier, tests_dir / "verifier")
        if evaluation.oracle.is_dir():
            copy_tree(evaluation.oracle, tests_dir / "oracle")
        else:
            shutil.copy2(evaluation.oracle, tests_dir / "oracle.md")

        test_script = tests_dir / "test.sh"
        write_text(test_script, self._verifier_wrapper(self._load_run_spec(unit)))
        test_script.chmod(0o755)

    @staticmethod
    def _load_run_spec(unit: HarborTaskUnit) -> Optional[RunSpec]:
        """读任务的 run_spec.json，坏 spec 当场报 HarborAdapterError。

        Harbor 的 verifier 容器里没有第二次机会加载它（wrapper 是静态脚本），
        所以校验全部在 materialize 时做掉。
        """
        try:
            return load_run_spec(unit.evaluation.root, unit.package.task_id)
        except RunSpecError as exc:
            raise HarborAdapterError(str(exc)) from exc

    @staticmethod
    def _write_solution(evaluation: EvaluationPaths, task_dir: Path) -> None:
        """evaluations/<level>/solution/ 原样成为 Harbor 的 solution/。

        只有 `harbor run -a oracle` 会把它上传到容器并执行 solve.sh，用来端到端
        验证"镜像 + GPU + prepare 产物 + verifier"这条链路；普通 agent 看不到。
        """
        source = evaluation.root / "solution"
        if not (source / "solve.sh").is_file():
            return
        copy_tree(source, task_dir / "solution")
        (task_dir / "solution" / "solve.sh").chmod(0o755)

    # Harbor 的 verifier 容器里这些路径是常量（见 _write_task_config 的
    # shared_env）；run_spec 的 $PTDBENCH_* 引用在这里展开成字面量，wrapper
    # 里就不用做运行时替换。
    _HARBOR_SPEC_PATHS = {
        "PTDBENCH_WORKSPACE": "/workspace",
        "PTDBENCH_RESULTS": "/logs/artifacts",
    }

    @classmethod
    def _verifier_wrapper(cls, spec: Optional[RunSpec]) -> str:
        """solution 层的 wrapper：report 门槛 → 系统重跑 → test.sh。

        Harbor 只有 agent / verifier 两个 stage，rerun 折进 verifier 容器：同一
        台机器、同一个 workspace（environment_mode = "shared"，可写），agent 交
        的 report.md 和它装进 PYTHONUSERBASE 的包都在。和 docker 后端
        `_run_rerun` 对应的差别只有一处：那边 rerun 是独立 sandbox；这边脚本
        自己把 PTDBENCH_STAGE 顶成 rerun 再跑实验，跑完恢复成 verifier。

        report 门槛对**所有** solution 任务生效（含没有 run_spec 的
        aime25）：没交 report 就不重跑也不判分，直接 0 分。
        issue 层没有这段（level 由运行时的 PTDBENCH_EVALUATION_LEVEL 判断，
        wrapper 对两个层级是同一份）。
        """
        rerun_section = ""
        if spec is not None:
            log_roots = " ".join(shlex.quote(root) for root in spec.log_roots)
            expanded_env = spec.expand_env(cls._HARBOR_SPEC_PATHS)
            env_exports = "\n".join(
                f"export {key}={shlex.quote(value)}"
                for key, value in expanded_env.items()
            )
            command_line = f"bash -c {shlex.quote(spec.command)}"
            # 超时能落实就落实：timeout 不在（个别精简镜像）就裸跑，交给
            # Harbor 自己的 verifier 超时兜底。
            if spec.timeout_s is not None:
                command_line = (
                    f"command -v timeout >/dev/null 2>&1 "
                    f"&& exec timeout {spec.timeout_s}s {command_line}; "
                    f"exec {command_line}"
                )
            env_note = (
                "# run_spec 的 env（$PTDBENCH_* 已在 materialize 时展开）。它盖在"
                "任务 metadata env 之上：run_spec 是协议要求的生效值。"
                if env_exports
                else ""
            )
            rerun_section = f'''# --- PTDBench rerun: re-run the experiment on the agent's workspace ------
# command / env 来自任务的 run_spec.json（materialize 时展开内联）。
# 先清日志：agent 阶段就算违反协议自己跑过，旧日志也不算数，判分只认这一遍。
for _root in {log_roots}; do
  rm -rf "$PTDBENCH_RESULTS/$_root"
done
unset _root
{env_note}
{env_exports}

# activate hook 的 once-only guard（PTDBENCH_ACTIVATED）不会随 stage 重跑，
# 所以这里照 hook 的 rerun 分支显式给这三个变量，跑完清掉——判分不能依赖
# agent 装进 workspace user base 的包。
PTDBENCH_STAGE=rerun
PYTHONUSERBASE=/workspace/.ptdbench-user
PIP_USER=1
PIP_BREAK_SYSTEM_PACKAGES=1
export PTDBENCH_STAGE PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
( cd /workspace && {command_line} )
_rerun_status=$?
unset PTDBENCH_STAGE PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
export PTDBENCH_STAGE=verifier
echo "ptdbench: rerun exit=$_rerun_status" >&2
# rerun 失败也继续判分：判分脚本对着空日志/残缺日志自然会打 0，与 docker
# 后端"verifier 总是跑"保持一致。

'''
        return f'''#!/usr/bin/env bash
set +e

# Harbor 用 `bash -c` 而不是登录 shell 执行命令，BASH_ENV 之外再显式 source 一次，
# 保证本 wrapper 自身（rerun / verifier）也拿到 prepare.sh 写进 $PTDBENCH_PREPARED/env
# 的变量。hook 内容见 runtimes.base.PTDBENCH_ACTIVATE_BODY，由 prepare 写进 prepared。
if [ -f {ACTIVATE_HOOK} ]; then
  . {ACTIVATE_HOOK}
fi

# --- PTDBench report gate: no report.md, no rerun, no grading -------------
if [ "$PTDBENCH_EVALUATION_LEVEL" = "solution" ] && [ ! -s "$PTDBENCH_WORKSPACE/{REPORT_FILENAME}" ]; then
  echo "ptdbench: agent did not submit {REPORT_FILENAME} in the workspace; skipping rerun and verifier" >&2
  echo '{{"passed": false, "score": 0.0, "metrics": {{"failure": "agent did not submit {REPORT_FILENAME}"}}}}' > "$PTDBENCH_RESULTS/result.json"
  cp "$PTDBENCH_RESULTS/result.json" /logs/verifier/ptdbench-result.json
  printf '0\\n' > /logs/verifier/reward.txt
  exit 1
fi
if [ "$PTDBENCH_EVALUATION_LEVEL" = "solution" ]; then
  # report 归档进 results，与 docker 后端一致：workspace 跑完会被收走，
  # 复盘"agent 到底改了什么"要有据可查。
  cp "$PTDBENCH_WORKSPACE/{REPORT_FILENAME}" "$PTDBENCH_RESULTS/{REPORT_FILENAME}" 2>/dev/null || true
fi
{rerun_section}bash "$PTDBENCH_VERIFIER/test.sh"
verifier_status=$?

# 结构化结果放进 verifier 日志，确保远程 Harbor environment 也会下载回来。
result_file="$PTDBENCH_RESULTS/result.json"
if [ -f "$result_file" ]; then
  cp "$result_file" /logs/verifier/ptdbench-result.json
fi

# Harbor 只消费 reward 文件；PTDBench 的完整 metrics 仍由 result.json 承载。
score=""
if [ -f "$result_file" ]; then
  score="$(tr '\\n' ' ' < "$result_file" | sed -nE 's/.*"score"[[:space:]]*:[[:space:]]*(-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?).*/\\1/p')"
fi
if ! printf '%s' "$score" | grep -Eq '^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'; then
  if [ "$verifier_status" -eq 0 ]; then
    score="1"
  else
    score="0"
  fi
fi

printf '%s\\n' "$score" > /logs/verifier/reward.txt
exit "$verifier_status"
'''

    @staticmethod
    def _docker_image(package: TaskPackage) -> str:
        if package.environment.image is None:
            raise HarborAdapterError(
                f"task {package.task_id!r} declares no environment.image; Harbor needs "
                "a prebuilt image"
            )
        return EnvironmentImage(package.environment.image).tag

    # 离线开关：与 SandboxRuntime._network_policy_env 对 agent / verifier 给的一致。
    # 联网的只有 prepare，而它在 `harbor run` 之前就跑完了。
    _OFFLINE_ENV = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_API_KEY": "ptdbench-offline",
        "WANDB_SILENT": "true",
    }

    # /workspace 收回来只用于留档和 workspace_after_hash；判分在 shared 容器里读的
    # 是真实的 /workspace。权重和训练 checkpoint 动辄几十上百 GB，而带 exclude 的
    # 下载是容器里一次 `tar czf`（Harbor 给它 120s），所以按后缀挡掉。
    _WORKSPACE_ARTIFACT_EXCLUDE = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.distcp",
        "*.gguf",
        "__pycache__",
        # prepare.sh stages retrieval corpora and their arrow caches straight
        # into /workspace (search_r1 lands ~74GB of wiki-18 plus a 21M-row
        # arrow cache there). None of it is evidence about the run, and gzipping
        # it is what pushed the artifact download past its timeout -- which
        # discards the trial's score, not just its artifact.
        "*.arrow",
        ".hf_datasets_cache",
        "wiki-18",
        "*.index",
        "*.faiss",
    )

    def _write_task_config(
        self,
        unit: HarborTaskUnit,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        package, evaluation = unit.package, unit.evaluation
        oracle_path = "/tests/oracle.md" if evaluation.oracle.is_file() else "/tests/oracle"
        task_name = harbor_task_name(package.task_id, evaluation.level)
        title = str(package.metadata.get("title") or package.task_id)
        docker_image = self._docker_image(package)

        # 两个 stage 共用的部分：挂载路径、离线开关，以及让 `bash -c` 也加载
        # activate hook 的 BASH_ENV（Harbor 不起登录 shell）。
        # PTDBENCH_TASK_ID 只给 verifier：任务 id 往往直接点出缺陷
        # （verl_grpo_uid_after_repeat），agent 一跑 `env` 就能读到答案。
        shared_env = {
            "PTDBENCH_EVALUATION_LEVEL": evaluation.level,
            "PTDBENCH_WORKSPACE": "/workspace",
            "PTDBENCH_RESULTS": "/logs/artifacts",
            "PTDBENCH_VISIBLE_TASK": "/ptdbench",
            "PTDBENCH_PREPARED": PREPARED_DIR,
            "PTDBENCH_MODELS": MODELS_DIR,
            "PTDBENCH_DATASETS": DATASETS_DIR,
            "PTDBENCH_ENVIRONMENT": (
                PTDBENCH_ENVIRONMENT_DIR if package.paths.environment.is_dir() else ""
            ),
            "BASH_ENV": ACTIVATE_HOOK,
            **self._OFFLINE_ENV,
        }
        agent_env = {
            **shared_env,
            "PTDBENCH_TASK_ROOT": "/ptdbench",
            "PTDBENCH_STAGE": "agent",
        }
        verifier_env = {
            **shared_env,
            "PTDBENCH_TASK_ID": package.task_id,
            "PTDBENCH_TASK_ROOT": "/tests",
            "PTDBENCH_ORACLE": oracle_path,
            "PTDBENCH_VERIFIER": "/tests/verifier",
            "PTDBENCH_STAGE": "verifier",
            "PTDBENCH_JUDGE_PROVIDER": "openai-compatible",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL:-https://api.openai.com/v1}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
        }
        if config.judge_model:
            verifier_env["PTDBENCH_JUDGE_MODEL"] = config.judge_model

        # 任务声明的 env（CUDA_VISIBLE_DEVICES 之类）两个 stage 都要给。PTDBENCH_*
        # 进不来，EnvironmentPlan 已经挡过一道。
        for key, value in package.environment.env.items():
            agent_env[key] = value
            verifier_env[key] = value

        artifact_exclude = ", ".join(
            self._toml_string(pattern) for pattern in self._WORKSPACE_ARTIFACT_EXCLUDE
        )
        task_toml = f'''schema_version = "1.4"

artifacts = [{{ source = "/workspace", exclude = [{artifact_exclude}] }}]

[task]
name = {self._toml_string(task_name)}
description = {self._toml_string(title)}

[metadata]
ptdbench_task_id = {self._toml_string(package.task_id)}
ptdbench_evaluation_level = {self._toml_string(evaluation.level)}

[agent]
network_mode = {self._toml_string(config.agent_network)}

[verifier]
environment_mode = "shared"
network_mode = {self._toml_string(config.verifier_network)}

[environment]
docker_image = {self._toml_string(docker_image)}
workdir = "/workspace"
build_timeout_sec = {self._environment_start_timeout(task_dir / "environment")}
{self._toml_gpus(package, config)}
[environment.env]
{self._toml_env(agent_env)}

[verifier.env]
{self._toml_env(verifier_env)}
'''
        write_text(task_dir / "task.toml", task_toml)

    @staticmethod
    def _toml_gpus(package: TaskPackage, config: HarborEvaluationConfig) -> str:
        """`[environment].gpus` —— 远端后端唯一能拿到 GPU 数的地方。

        docker 后端拒收 task.toml 里的 `gpus`，它的设备预留走 compose overlay；但
        远端后端（cybertron）恰恰只读 task_env_config.gpus，devspace 的 `--gpu` 和
        CPU/内存配比都由它推出来，不写就会开出 0 卡的机器。

        取值优先级：显式 --gpus > 任务 metadata 里 CUDA_VISIBLE_DEVICES 的条目数。
        两者都没有就不写这一行，交给 Harbor 的默认值。
        """
        if not _is_remote_backend(config.harbor_args):
            return ""
        gpus = config.gpus
        if gpus is None:
            visible = (package.environment.env or {}).get("CUDA_VISIBLE_DEVICES")
            if visible:
                gpus = len([part for part in str(visible).split(",") if part.strip()])
        return "" if not gpus else f"gpus = {int(gpus)}\n"

    # Harbor 的默认值；预构建镜像不 build，这段时间几乎全花在上传 workspace 上。
    _ENVIRONMENT_START_BASE_SEC = 600
    # `docker compose cp` 的保守吞吐：每 GB 给 60s。
    _ENVIRONMENT_START_SEC_PER_GB = 60

    @classmethod
    def _environment_start_timeout(cls, environment_dir: Path) -> int:
        """build_timeout_sec 在 Harbor 里罩住整个环境启动，包括 environment/ 上传。

        prepare 把权重下进 workspace 的任务（slime_retool 约 16GB）按默认 600s
        会在 agent 开始前就超时，所以按实际上传量放宽。
        """
        size = sum(
            path.stat().st_size
            for path in environment_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        gigabytes = -(-size // (1 << 30))
        return cls._ENVIRONMENT_START_BASE_SEC + cls._ENVIRONMENT_START_SEC_PER_GB * gigabytes

    @classmethod
    def _toml_env(cls, env: Dict[str, str]) -> str:
        return "\n".join(
            f"{key} = {cls._toml_string(value)}" for key, value in env.items()
        )

    @staticmethod
    def _toml_string(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _write_manifest(
        package: TaskPackage,
        evaluation: EvaluationPaths,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        payload = {
            "adapter": "ptdbench-harbor",
            "task_id": package.task_id,
            "evaluation_level": evaluation.level,
            "verifier_mode": "shared",
            "docker_image": HarborTaskMaterializer._docker_image(package),
            "source_task": str(package.paths.root),
        }
        write_text(
            task_dir / "adapter-manifest.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class HarborRunner:
    """通过 CLI 启动 Harbor，避免与 Harbor 的 Python 版本和内部 API 耦合。"""

    _TASK_PATH_ARGS = {"-p", "--path"}
    # 挂载由 PTDBench 按 prepare 产物决定，而 Harbor 的 --mounts 是整个 job 共用的，
    # 所以每个 unit 单独一个 job；用户再传一份只会覆盖掉 prepared 的挂载。
    _MOUNT_ARGS = {"--mounts", "--mounts-json"}

    def run(
        self,
        prepared: PreparedHarborTask,
        config: HarborEvaluationConfig,
        job_name_suffix: Optional[str] = None,
    ) -> HarborRun:
        harbor_args = list(config.harbor_args)
        self._validate_harbor_args(harbor_args)
        mounts = self._mounts(prepared, config)

        job_name = self._option_value(harbor_args, "--job-name")
        jobs_dir_value = self._option_value(harbor_args, "-o", "--jobs-dir")
        config_path = self._option_value(harbor_args, "-c", "--config")
        injected_args = []
        # 没有 job config 时补全结果位置；使用 config 时完全尊重 Harbor 的配置。
        if job_name is not None and job_name_suffix:
            # 批量时用户给的 job 名要按 unit 区分，否则第二个 job 会撞上第一个。
            job_name = f"{job_name}-{job_name_suffix}"
            harbor_args = self._replace_option(harbor_args, "--job-name", job_name)
        if config_path is None:
            if job_name is None:
                job_name = self._job_name()
                injected_args.extend(["--job-name", job_name])
            if jobs_dir_value is None:
                jobs_dir_value = "jobs"
                injected_args.extend(["--jobs-dir", jobs_dir_value])

        injected_args.extend(self._remote_prepare_args(prepared, harbor_args))

        caller_cwd = Path.cwd().resolve()
        argv = [
            *config.harbor_command,
            "run",
            "--path",
            str(prepared.tasks_dir),
            *injected_args,
            "--mounts-json",
            json.dumps(mounts),
            *self._compose_overlay_args(prepared, harbor_args),
            *harbor_args,
        ]

        started = time.monotonic()
        display_argv = shlex.join(self._redact_argv(argv))
        print(f"[ptdbench] Starting Harbor: {display_argv}", file=sys.stderr, flush=True)
        try:
            completed = self._run_streaming(argv, caller_cwd)
            command = CommandResult(
                argv=self._redact_argv(argv),
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
                duration_s=time.monotonic() - started,
                cwd=str(caller_cwd),
            )
        except FileNotFoundError as exc:
            executable = config.harbor_command[0]
            raise HarborAdapterError(f"Harbor command not found: {executable}") from exc

        job_dir = self._reported_job_dir(completed.stdout, caller_cwd)
        if job_dir is None and jobs_dir_value is not None and job_name is not None:
            jobs_dir = Path(jobs_dir_value).expanduser()
            if not jobs_dir.is_absolute():
                jobs_dir = caller_cwd / jobs_dir
            job_dir = jobs_dir.resolve() / job_name
        if job_dir is None:
            raise HarborAdapterError(
                "Harbor did not report its result path; pass --jobs-dir and "
                "--job-name explicitly or use a Harbor version that prints it"
            )

        return HarborRun(command=command, job_dir=job_dir)

    @staticmethod
    def _run_streaming(
        argv: Sequence[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """实时转发 Harbor 输出，同时保留一份用于评测报告和结果定位。"""
        with subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def forward(source, target, chunks: list[str]) -> None:
                for line in iter(source.readline, ""):
                    chunks.append(line)
                    target.write(line)
                    target.flush()

            assert process.stdout is not None
            assert process.stderr is not None
            threads = (
                threading.Thread(
                    target=forward,
                    args=(process.stdout, sys.stdout, stdout_chunks),
                    daemon=True,
                ),
                threading.Thread(
                    target=forward,
                    args=(process.stderr, sys.stderr, stderr_chunks),
                    daemon=True,
                ),
            )
            for thread in threads:
                thread.start()
            returncode = process.wait()
            for thread in threads:
                thread.join()

        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    @classmethod
    def _validate_harbor_args(cls, args: Sequence[str]) -> None:
        """--path 必须指向在线生成的 task，其他参数保持 Harbor 原义。"""
        for arg in args:
            name = arg.split("=", 1)[0]
            if name in cls._TASK_PATH_ARGS:
                raise ValueError(
                    f"{name} is managed by the PTDBench task materializer; "
                    "pass the PTDBench task as task_root"
                )
            if name in cls._MOUNT_ARGS:
                raise ValueError(
                    f"{name} is managed by PTDBench (prepared / models / datasets "
                    "mounts); use --models-dir / --datasets-dir instead"
                )

    @staticmethod
    def _mounts(
        prepared: PreparedHarborTask,
        config: HarborEvaluationConfig,
    ) -> list[Dict[str, Any]]:
        """prepare 产物、任务 environment/ 和两个下载缓存，全部只读。

        与 docker 评测里 agent / verifier 看到的挂载一致；可写的只有 Harbor 上传
        的 /workspace。

        远端后端（cybertron）没有挂载这回事：它声明 capabilities.mounted=False，
        这几个路径由后端在 devspace 上自己建——/ptdbench/{models,datasets} 软链到
        集群共享缓存，prepared 由远端 prepare 写在持久化 workspace 里。这里的
        source 又都是本机绝对路径，送过去没有意义，所以整组略过。
        """
        if len(prepared.units) != 1:
            raise HarborAdapterError(
                "Harbor mounts apply to the whole job; run one PTDBench task per job"
            )
        if _is_remote_backend(config.harbor_args):
            return []
        unit = prepared.units[0]
        pairs = []
        if unit.prepared is not None:
            pairs.append((unit.prepared, PREPARED_DIR))
        if unit.package.paths.environment.is_dir():
            pairs.append((unit.package.paths.environment, PTDBENCH_ENVIRONMENT_DIR))
        pairs.append((config.models_dir, MODELS_DIR))
        pairs.append((config.datasets_dir, DATASETS_DIR))
        return [
            {
                "type": "bind",
                "source": str(Path(source).resolve()),
                "target": target,
                "read_only": True,
            }
            for source, target in pairs
        ]

    @staticmethod
    def _remote_prepare_args(
        prepared: PreparedHarborTask,
        harbor_args: Sequence[str],
    ) -> list[str]:
        """远端后端跑 prepare 所需的两个路径，作为 environment kwarg 注入。

        后端要自己把 prepare.sh 和 fetch/dataset 工具送上 devspace，但物化出来的
        Harbor task 里并没有任务原本的 environment/（那个目录名在 Harbor 语义下装
        的是要上传到 workdir 的 workspace 树），docker 路径是靠 bind mount 补的。
        远端没有挂载，所以这里把宿主机上的两个源目录显式告诉后端。

        用户显式给过同名 kwarg 时不覆盖——调试时可以指到别处。
        """
        if not _is_remote_backend(harbor_args):
            return []
        if len(prepared.units) != 1:
            raise HarborAdapterError(
                "remote prepare applies to the whole job; run one PTDBench task per job"
            )
        existing = {
            value.split("=", 1)[0]
            for value in HarborRunner._option_values(harbor_args, "--environment-kwarg")
            if "=" in value
        }
        package = prepared.units[0].package
        args: list[str] = []
        if "prepare_dir" not in existing and package.paths.environment.is_dir():
            args.extend(
                ["--environment-kwarg", f"prepare_dir={package.paths.environment.resolve()}"]
            )
        if "tooling_dir" not in existing and DOCKER_DIR.is_dir():
            args.extend(["--environment-kwarg", f"tooling_dir={DOCKER_DIR.resolve()}"])
        return args

    @staticmethod
    def _compose_overlay_args(
        prepared: PreparedHarborTask,
        harbor_args: Sequence[str] = (),
    ) -> list[str]:
        """任务的 environment/docker-compose.yaml 作为 compose overlay 交给 Harbor。

        GPU 任务靠它拿到设备预留和 shm_size：Harbor 的 docker 环境声明不支持
        GPU，task.toml 里写 `gpus` 会被拒，能生效的只有 `main` 服务上的 compose
        覆盖。它不能放进生成 task 的 environment/——那里有 compose 文件时 Harbor
        就不再上传 workspace——所以走 `--extra-docker-compose`，与 job 一一对应。

        远端后端没有 compose 这一层（devspace 本身就是容器，GPU 由 devspace 的
        `--gpu` 决定、读的是 task.toml 的 `gpus`），所以不注入；否则会把一个只有
        docker 才懂的参数塞给它。
        """
        if _is_remote_backend(harbor_args):
            return []
        overlays = []
        for unit in prepared.units:
            compose = unit.package.paths.environment / "docker-compose.yaml"
            if compose.is_file():
                overlays.extend(["--extra-docker-compose", str(compose.resolve())])
        return overlays

    @staticmethod
    def _replace_option(args: Sequence[str], name: str, value: str) -> list[str]:
        result = []
        skip = False
        for index, arg in enumerate(args):
            if skip:
                skip = False
                continue
            if arg == name:
                result.extend([name, value])
                skip = True
            elif arg.startswith(name + "="):
                result.append(f"{name}={value}")
            else:
                result.append(arg)
        return result

    @staticmethod
    def _option_value(args: Sequence[str], *names: str) -> Optional[str]:
        value = None
        for index, arg in enumerate(args):
            for name in names:
                if arg == name:
                    if index + 1 >= len(args):
                        raise ValueError(f"missing value for Harbor argument {name}")
                    value = args[index + 1]
                elif arg.startswith(name + "="):
                    value = arg.split("=", 1)[1]
        return value

    @staticmethod
    def _option_values(args: Sequence[str], *names: str) -> list[str]:
        """_option_value 的可重复版：收集某个选项出现过的全部取值。

        `--environment-kwarg` 这类选项会被给很多次，判断「用户是否已经自己指定过
        某个 key」要看全部取值，只取最后一个会漏。
        """
        values: list[str] = []
        for index, arg in enumerate(args):
            for name in names:
                if arg == name:
                    if index + 1 >= len(args):
                        raise ValueError(f"missing value for Harbor argument {name}")
                    values.append(args[index + 1])
                elif arg.startswith(name + "="):
                    values.append(arg.split("=", 1)[1])
        return values

    @staticmethod
    def _redact_argv(argv: Sequence[str]) -> list[str]:
        sensitive = re.compile(
            r"(?i)([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=([^\s]+)"
        )
        return [sensitive.sub(r"\1=<redacted>", arg) for arg in argv]

    @staticmethod
    def _reported_job_dir(stdout: str, cwd: Path) -> Optional[Path]:
        """从 Harbor 的稳定完成提示中读取实际结果目录。"""
        for line in reversed(stdout.splitlines()):
            match = re.search(r"Results written to (.+?result\.json)\s*$", line)
            if match is None:
                continue
            result_path = Path(match.group(1)).expanduser()
            if not result_path.is_absolute():
                result_path = cwd / result_path
            return result_path.resolve().parent
        return None

    @staticmethod
    def _job_name() -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        return f"ptdbench-{timestamp}-{uuid.uuid4().hex[:8]}"


class HarborResultMapper:
    """读取一个 Harbor job，并转换其中的所有 trial 结果。"""

    def map(
        self,
        prepared: PreparedHarborTask,
        run: HarborRun,
        config: HarborEvaluationConfig,
    ) -> HarborEvaluationReport:
        if not (run.job_dir / "result.json").is_file() and run.command.returncode != 0:
            raise HarborAdapterError(
                "Harbor command failed with exit code "
                f"{run.command.returncode}; no job result was written to {run.job_dir}"
            )
        job_result = self._read_json(run.job_dir / "result.json", "Harbor job result")
        trials = self._trial_results(run.job_dir, job_result)
        units = prepared.units_by_task_name

        results = []
        for trial in trials:
            if not isinstance(trial, dict):
                raise HarborAdapterError("Harbor trial result must be a JSON object")
            unit = self._unit_for_trial(trial, units, prepared.units)
            results.append(self._map_trial(unit, run, config, trial))

        report = HarborEvaluationReport(
            task_ids=[unit.package.task_id for unit in prepared.units],
            evaluation_level=config.evaluation_level,
            trials=results,
            report_dir=run.job_dir,
            command=run.command,
        )
        write_text(
            run.job_dir / "ptdbench-report.json",
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        )
        return report

    @staticmethod
    def _unit_for_trial(
        trial: Dict[str, Any],
        units: Dict[str, HarborTaskUnit],
        all_units: Sequence[HarborTaskUnit],
    ) -> HarborTaskUnit:
        """按 Harbor 的 task_name 把 trial 归回它的 PTDBench task。"""
        task_name = trial.get("task_name")
        if isinstance(task_name, str) and task_name in units:
            return units[task_name]
        # 单任务批次里 task_name 缺失仍可无歧义归属，历史 Harbor 版本依赖这一点。
        if task_name is None and len(all_units) == 1:
            return all_units[0]
        raise HarborAdapterError(
            f"Harbor trial task_name {task_name!r} does not match any "
            f"materialized task: {', '.join(sorted(units))}"
        )

    def _trial_results(
        self,
        job_dir: Path,
        job_result: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """当前 Harbor 将完整结果写在各 trial 目录，job 结果只保留汇总。"""
        trials = []
        try:
            children = sorted(job_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise HarborAdapterError(f"failed to inspect Harbor job: {job_dir}") from exc

        for child in children:
            result_path = child / "result.json"
            if not child.is_dir() or not result_path.is_file():
                continue
            trial = self._read_json(result_path, "Harbor trial result")
            if trial.get("trial_name") != child.name:
                raise HarborAdapterError(
                    f"Harbor trial_name does not match its directory: {child}"
                )
            trials.append(trial)
        if trials:
            return trials

        # 兼容仍把 trial_results 内嵌在 job/result.json 的旧版 Harbor。
        embedded = job_result.get("trial_results")
        if isinstance(embedded, list) and embedded:
            return embedded
        raise HarborAdapterError("Harbor job must contain at least one trial result")

    def _map_trial(
        self,
        unit: HarborTaskUnit,
        run: HarborRun,
        config: HarborEvaluationConfig,
        trial: Dict[str, Any],
    ) -> EvaluationResult:
        package = unit.package
        evaluation = unit.evaluation
        trial_dir = self._trial_dir(run.job_dir, trial.get("trial_name"))

        artifact_error = self._workspace_artifact_error(trial_dir)
        ptdbench_result = self._read_ptdbench_result(trial_dir)
        rewards = self._rewards(trial)
        metrics = self._metrics(ptdbench_result, rewards, trial, config)

        infrastructure_errors = []
        if run.command.returncode != 0 or run.command.timed_out:
            infrastructure_errors.append(
                f"Harbor command failed with exit code {run.command.returncode}"
            )
        if trial.get("exception_info"):
            exception = trial["exception_info"]
            message = (
                exception.get("exception_message")
                if isinstance(exception, dict)
                else exception
            )
            infrastructure_errors.append(f"Harbor trial exception: {message}")
        if artifact_error:
            infrastructure_errors.append(artifact_error)
        if infrastructure_errors:
            metrics["infrastructure_errors"] = infrastructure_errors

        score = self._score(metrics)
        passed = not infrastructure_errors and self._passed(metrics, score)
        workspace_dir = trial_dir / "artifacts" / "workspace"
        workspace_after_hash = tree_digest(workspace_dir) if workspace_dir.is_dir() else ""

        result = EvaluationResult(
            task_id=package.task_id,
            evaluation_level=evaluation.level,
            passed=passed,
            score=score if not infrastructure_errors else 0.0,
            metrics=metrics,
            report_dir=trial_dir,
            # Agent/model/backend 等信息以 Harbor 原生命令和 job config 为准。
            agent_spec=None,
            agent_setup=None,
            agent=None,
            verifier=None,
            workspace_before_hash=unit.workspace_before_hash,
            workspace_after_hash=workspace_after_hash,
            metadata=package.metadata,
        )
        self._write_adapter_outputs(trial_dir, run.command, result)
        return result

    @staticmethod
    def _trial_dir(job_dir: Path, trial_name: Any) -> Path:
        if not isinstance(trial_name, str) or not trial_name:
            raise HarborAdapterError("Harbor trial result is missing trial_name")
        trial_dir = (job_dir / trial_name).resolve()
        try:
            trial_dir.relative_to(job_dir.resolve())
        except ValueError as exc:
            raise HarborAdapterError("Harbor trial_name escapes its job directory") from exc
        if not trial_dir.is_dir():
            raise HarborAdapterError(f"Harbor trial directory not found: {trial_dir}")
        return trial_dir

    def _workspace_artifact_error(self, trial_dir: Path) -> Optional[str]:
        manifest_path = trial_dir / "artifacts" / "manifest.json"
        if not manifest_path.is_file():
            return "Harbor artifact manifest is missing"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Harbor artifact manifest is invalid: {exc}"
        entries = manifest.get("entries") if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list):
            return "Harbor artifact manifest must contain a list"
        workspace_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("source") == "/workspace"
        ]
        if not workspace_entries:
            return "Harbor did not record the /workspace artifact"
        failed = [
            entry.get("status")
            for entry in workspace_entries
            if entry.get("status") not in {"ok", "empty"}
        ]
        if failed:
            return f"Harbor failed to collect /workspace artifact: {', '.join(map(str, failed))}"
        if not (trial_dir / "artifacts" / "workspace").is_dir():
            return "Harbor /workspace artifact directory is missing"
        return None

    def _read_ptdbench_result(self, trial_dir: Path) -> Dict[str, Any]:
        candidates = (
            trial_dir / "verifier" / "ptdbench-result.json",
            trial_dir / "artifacts" / "logs" / "artifacts" / "result.json",
        )
        for path in candidates:
            if not path.is_file():
                continue
            payload = self._read_json(path, "PTDBench verifier result")
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _rewards(trial: Dict[str, Any]) -> Dict[str, float]:
        verifier_result = trial.get("verifier_result")
        rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
        if not isinstance(rewards, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in rewards.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    @staticmethod
    def _metrics(
        ptdbench_result: Dict[str, Any],
        rewards: Dict[str, float],
        trial: Dict[str, Any],
        config: HarborEvaluationConfig,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        nested = ptdbench_result.get("metrics")
        if isinstance(nested, dict):
            metrics.update(nested)
        for key in ("passed", "score"):
            if key in ptdbench_result:
                metrics[key] = ptdbench_result[key]
        for key, value in ptdbench_result.items():
            if key not in {"metrics", "passed", "score"} and key not in metrics:
                metrics[key] = value
        metrics["harbor_rewards"] = rewards
        metrics["harbor_verifier_mode"] = (
            trial.get("verifier_environment_mode") or "shared"
        )
        return metrics

    @staticmethod
    def _score(metrics: Dict[str, Any]) -> Optional[float]:
        score = metrics.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        rewards = metrics.get("harbor_rewards")
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
            if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                return float(reward)
        if metrics.get("passed") is True:
            return 1.0
        if metrics.get("passed") is False:
            return 0.0
        return None

    @staticmethod
    def _passed(metrics: Dict[str, Any], score: Optional[float]) -> bool:
        if isinstance(metrics.get("passed"), bool):
            return metrics["passed"]
        return score == 1.0

    @staticmethod
    def _read_json(path: Path, label: str) -> Dict[str, Any]:
        if not path.is_file():
            raise HarborAdapterError(f"{label} not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborAdapterError(f"failed to read {label}: {path}") from exc
        if not isinstance(payload, dict):
            raise HarborAdapterError(f"{label} must be a JSON object: {path}")
        return payload

    @staticmethod
    def _write_adapter_outputs(
        trial_dir: Path,
        command: CommandResult,
        result: EvaluationResult,
    ) -> None:
        write_text(
            trial_dir / "ptdbench-harbor-command.json",
            json.dumps(command.to_dict(), indent=2, ensure_ascii=False),
        )
        write_text(
            trial_dir / "report.json",
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        )


class HarborPreparer:
    """在 `harbor run` 之前，用同一个镜像跑任务的 prepare.sh。

    prepare 是唯一联网的阶段，下模型、下数据集、写 prepared/env。它不进 Harbor：
    Harbor 的 agent / verifier 都是离线的，而且一个 job 里没有"先跑一段准备"的
    位置。复用 TaskEvaluator 的 staging 和 prepare 逻辑，与 docker 评测完全一致。
    """

    def __init__(self, evaluator: Optional[TaskEvaluator] = None) -> None:
        self._evaluator = evaluator

    @property
    def evaluator(self) -> TaskEvaluator:
        if self._evaluator is None:
            self._evaluator = TaskEvaluator("docker")
        return self._evaluator

    def prepare(
        self,
        unit: HarborTaskUnit,
        work_dir: Path,
        config: HarborEvaluationConfig,
    ) -> HarborTaskUnit:
        package = unit.package
        image = package.environment.environment_image
        if image is None:
            raise HarborAdapterError(
                f"task {package.task_id!r} declares no environment.image; Harbor needs "
                "a prebuilt image"
            )
        evaluator = self.evaluator
        eval_config = EvaluationConfig(
            work_dir=work_dir,
            evaluation_level=unit.evaluation.level,
            huggingface_mirror=config.huggingface_mirror,
            models_dir=config.models_dir,
            datasets_dir=config.datasets_dir,
            keep_artifacts=True,
        )
        run = evaluator._prepare_run(package, unit.evaluation, eval_config)
        # 镜像按环境共享：已存在直接用，不存在才构建。Harbor 用的是同一个 tag。
        artifact = evaluator.runtime.ensure_environment(image)
        print(
            f"[ptdbench] Preparing {package.task_id} in {artifact.reference or image.tag}",
            file=sys.stderr,
            flush=True,
        )
        result = evaluator._run_prepare(package, unit.evaluation, run, eval_config, artifact)
        if result is not None and result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-20:])
            raise HarborAdapterError(
                f"task {package.task_id!r}: prepare failed with exit {result.returncode}"
                + (f"\n{tail}" if tail else "")
            )
        # activate hook 不再烤进镜像：把它和 prepare.sh 写的 env 放在一起，容器里按
        # PREPARED_DIR 挂进来，agent / verifier 的 `bash -c` 靠 BASH_ENV=ACTIVATE_HOOK
        # 自动 source（内容是 runtimes.base 的单一来源）。
        if run.prepared is not None:
            write_text(
                Path(run.prepared) / "ptdbench-activate.sh",
                "#!/usr/bin/env sh\n" + PTDBENCH_ACTIVATE_BODY,
            )
        return dataclasses.replace(unit, workspace=run.workspace, prepared=run.prepared)


class HarborShell:
    """prepare → docker run -it でインタラクティブシェルを起動する。

    harbor-evaluate の agent フェーズと完全に同じ環境を用意しながら、Harbor と
    agent を起動せずにインタラクティブシェルを渡す。HarborPreparer でステージング
    し、HarborRunner._mounts() や HarborTaskMaterializer._write_task_config() と
    同じマウント・env 構成を docker run に変換する。

    workspace のステージング結果は state_file に記録され、次回起動時に
    --force-prepare なしで再利用できる。16GB 超の prepare を毎回走らせないためのもの。
    """

    _STATE_FILE = "ptdbench-shell.json"

    def __init__(self, preparer: Optional[HarborPreparer] = None) -> None:
        self.preparer = preparer or HarborPreparer()

    def shell(
        self,
        task_root: str | Path,
        config: HarborEvaluationConfig,
        workspace_dir: Path,
        *,
        shell_cmd: str = "bash",
        force_prepare: bool = False,
        gpus: Optional[str] = None,
        shm_size: Optional[str] = None,
        extra_docker_args: Sequence[str] = (),
    ) -> None:
        """prepare して docker run -it に exec する。os.execvp で置き換わるため戻らない。"""
        package = load_task_package(Path(task_root))
        if not package.report.ok:
            raise SchemaError(f"task package failed validation: {package.paths.root}")
        try:
            evaluation = package.paths.evaluations[config.evaluation_level]
        except KeyError:
            available = ", ".join(package.paths.evaluations) or "none"
            raise SchemaError(
                f"evaluation level {config.evaluation_level!r} is unavailable; "
                f"available levels: {available}"
            )
        unit = HarborTaskUnit(
            package=package,
            evaluation=evaluation,
            workspace_before_hash="",
        )

        workspace_dir = workspace_dir.resolve()
        state_path = workspace_dir / self._STATE_FILE

        if not force_prepare and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            unit = dataclasses.replace(
                unit,
                workspace=Path(state["workspace"]),
                prepared=Path(state["prepared"]) if state.get("prepared") else None,
            )
            print(
                f"[ptdbench] Reusing staged workspace at {unit.workspace}",
                file=sys.stderr,
                flush=True,
            )
        else:
            ensure_dir(workspace_dir)
            unit = self.preparer.prepare(unit, workspace_dir, config)
            state_data: Dict[str, Any] = {
                "task_id": package.task_id,
                "workspace": str(unit.workspace),
                "prepared": str(unit.prepared) if unit.prepared else None,
            }
            write_text(state_path, json.dumps(state_data, indent=2, ensure_ascii=False))

        image = HarborTaskMaterializer._docker_image(package)
        argv = self._build_docker_argv(
            unit,
            package,
            config,
            image,
            shell_cmd,
            gpus=gpus,
            shm_size=shm_size,
            extra_docker_args=extra_docker_args,
        )
        print(f"[ptdbench] Entering container: {shlex.join(argv)}", file=sys.stderr, flush=True)
        os.execvp(argv[0], argv)

    def _build_docker_argv(
        self,
        unit: HarborTaskUnit,
        package: TaskPackage,
        config: HarborEvaluationConfig,
        image: str,
        shell_cmd: str,
        *,
        gpus: Optional[str],
        shm_size: Optional[str],
        extra_docker_args: Sequence[str],
    ) -> list[str]:
        argv = ["docker", "run", "--rm", "-it"]

        # GPU 予約と shm_size は docker-compose.yaml のオーバーレイから来る値を踏襲する。
        # 明示的に引数を渡した場合のみその値を使い、なければオーバーレイの有無で判定する。
        overlay = package.paths.environment / "docker-compose.yaml"
        has_overlay = overlay.is_file()
        effective_gpus = gpus if gpus is not None else ("all" if has_overlay else None)
        effective_shm = shm_size if shm_size is not None else ("16g" if has_overlay else None)
        if effective_gpus:
            argv.extend(["--gpus", effective_gpus])
        if effective_shm:
            argv.extend(["--shm-size", effective_shm])

        # workspace は書き込み可；その他はすべて読み取り専用 — harbor-evaluate と同じ方針。
        assert unit.workspace is not None
        argv.extend(["-v", f"{unit.workspace}:/workspace"])
        if unit.prepared is not None and Path(unit.prepared).is_dir():
            argv.extend(["-v", f"{unit.prepared}:{PREPARED_DIR}:ro"])
        if package.paths.environment.is_dir():
            argv.extend(
                ["-v", f"{package.paths.environment.resolve()}:{PTDBENCH_ENVIRONMENT_DIR}:ro"]
            )
        argv.extend(["-v", f"{config.models_dir}:{MODELS_DIR}:ro"])
        argv.extend(["-v", f"{config.datasets_dir}:{DATASETS_DIR}:ro"])

        argv.extend(["-w", "/workspace"])

        # _write_task_config() の agent_env と同じ構成。PTDBENCH_RESULTS は
        # /workspace 配下の書き込み可能な場所にする（/logs/artifacts は存在しない）。
        env: Dict[str, str] = {
            "PTDBENCH_TASK_ID": package.task_id,
            "PTDBENCH_EVALUATION_LEVEL": unit.evaluation.level,
            "PTDBENCH_WORKSPACE": "/workspace",
            "PTDBENCH_RESULTS": "/workspace/ptdbench-results",
            "PTDBENCH_TASK_ROOT": "/workspace",
            "PTDBENCH_STAGE": "agent",
            "PTDBENCH_PREPARED": PREPARED_DIR,
            "PTDBENCH_MODELS": MODELS_DIR,
            "PTDBENCH_DATASETS": DATASETS_DIR,
            "PTDBENCH_ENVIRONMENT": (
                PTDBENCH_ENVIRONMENT_DIR if package.paths.environment.is_dir() else ""
            ),
            "BASH_ENV": ACTIVATE_HOOK,
            **self._OFFLINE_ENV,
            # task 宣言の env（CUDA_VISIBLE_DEVICES 等）で上書きする。
            **package.environment.env,
        }
        for key, value in env.items():
            argv.extend(["-e", f"{key}={value}"])

        argv.extend(extra_docker_args)
        argv.append(image)
        argv.append(shell_cmd)
        return argv

    # harbor-evaluate の離線フラグと同一セット。
    _OFFLINE_ENV = HarborTaskMaterializer._OFFLINE_ENV


class HarborEvaluator:
    """PTDBench 的 Harbor 执行入口，与 TaskEvaluator 并行存在。"""

    def __init__(
        self,
        materializer: Optional[HarborTaskMaterializer] = None,
        runner: Optional[HarborRunner] = None,
        result_mapper: Optional[HarborResultMapper] = None,
        preparer: Optional[HarborPreparer] = None,
    ) -> None:
        self.preparer = preparer or HarborPreparer()
        self.materializer = materializer or HarborTaskMaterializer()
        self.runner = runner or HarborRunner()
        self.result_mapper = result_mapper or HarborResultMapper()

    def load(self, task_root: str | Path, strict: bool = True) -> TaskPackage:
        return load_task_package(Path(task_root), strict=strict)

    def validate(self, task_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(task_root, strict=strict).report.to_dict()

    def evaluate(
        self,
        task_roots: str | Path | Sequence[str | Path],
        config: Optional[HarborEvaluationConfig] = None,
        strict: bool = True,
    ) -> HarborEvaluationReport:
        """逐个 task：prepare → 物化 → 一个 Harbor job，最后合并成一份报告。

        每个 task 单独一个 job，是因为 Harbor 的挂载是 job 级的，而 prepared 目录
        是逐任务的。某个 task 的 prepare 失败只让它缺席报告，不拖累其余的。
        """
        config = config or HarborEvaluationConfig()
        units = [
            self._load_unit(task_root, config.evaluation_level, strict)
            for task_root in self._normalize_task_roots(task_roots)
        ]
        # 同名 task 放进各自的 job 也会互相覆盖报告，先整批拦下。
        self.materializer._reject_duplicate_task_names(units)
        batch = len(units) > 1
        reports: list[HarborEvaluationReport] = []
        errors: list[str] = []
        for unit in units:
            try:
                reports.append(self._evaluate_unit(unit, config, batch))
            except HarborAdapterError as exc:
                if not batch:
                    raise
                errors.append(f"{unit.package.task_id}: {exc}")
                print(f"[ptdbench] {unit.package.task_id} skipped: {exc}", file=sys.stderr)
        if not reports:
            raise HarborAdapterError("; ".join(errors) or "no task was evaluated")
        return self._merge_reports(units, reports, config)

    def _evaluate_unit(
        self,
        unit: HarborTaskUnit,
        config: HarborEvaluationConfig,
        batch: bool,
    ) -> HarborEvaluationReport:
        root = self._make_run_dir([unit], config.temp_dir)
        try:
            if _is_remote_backend(config.harbor_args):
                # 远端后端自己在 devspace 上跑 prepare（权重落在集群共享缓存 +
                # 该 trial 的持久 workspace，转换用 trial 自己的 GPU），所以这里
                # 不跑本地 prepare。unit.workspace / unit.prepared 保持 None：
                # 物化时 environment/ 退回任务自带的 workspace/ 树（git 里那份，
                # 小且正是要上传的内容），_mounts 也不会再发本机路径。
                print(
                    f"[ptdbench] Skipping local prepare for {unit.package.task_id}; "
                    "the remote backend runs prepare.sh on the devspace",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                unit = self.preparer.prepare(unit, ensure_dir(root / "prep"), config)
            print(
                f"[ptdbench] Materializing {unit.package.task_id}:{unit.evaluation.level}",
                file=sys.stderr,
                flush=True,
            )
            prepared = self.materializer.materialize([unit], root / "harbor", config)
            run = self.runner.run(
                prepared, config, job_name_suffix=unit.dir_name if batch else None
            )
            print(
                f"[ptdbench] Reading Harbor results from {run.job_dir}",
                file=sys.stderr,
                flush=True,
            )
            return self.result_mapper.map(prepared, run, config)
        finally:
            # Harbor job 保存在原生 --jobs-dir；这里只清理 prepare staging 和临时 task。
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def _merge_reports(
        units: Sequence[HarborTaskUnit],
        reports: Sequence[HarborEvaluationReport],
        config: HarborEvaluationConfig,
    ) -> HarborEvaluationReport:
        if len(units) == 1:
            return reports[0]
        last = reports[-1]
        return HarborEvaluationReport(
            task_ids=[unit.package.task_id for unit in units],
            evaluation_level=config.evaluation_level,
            trials=[trial for report in reports for trial in report.trials],
            report_dir=last.report_dir,
            command=last.command,
        )

    @staticmethod
    def _normalize_task_roots(
        task_roots: str | Path | Sequence[str | Path],
    ) -> list[Path]:
        if isinstance(task_roots, (str, Path)):
            return [Path(task_roots)]
        roots = [Path(task_root) for task_root in task_roots]
        if not roots:
            raise HarborAdapterError("at least one task root is required")
        return roots

    def _load_unit(
        self,
        task_root: str | Path,
        evaluation_level: str,
        strict: bool,
    ) -> HarborTaskUnit:
        package = self.load(task_root, strict=strict)
        if not package.report.ok:
            raise SchemaError(f"task package failed validation: {package.paths.root}")
        evaluation = self._select_evaluation(package, evaluation_level)
        return HarborTaskUnit(
            package=package,
            evaluation=evaluation,
            workspace_before_hash=tree_digest(package.paths.workspace),
        )

    @staticmethod
    def _select_evaluation(package: TaskPackage, level: str) -> EvaluationPaths:
        try:
            return package.paths.evaluations[level]
        except KeyError as exc:
            available = ", ".join(package.evaluation_levels) or "none"
            raise SchemaError(
                f"evaluation level {level!r} is unavailable; available levels: {available}"
            ) from exc

    @staticmethod
    def _make_run_dir(
        units: Sequence[HarborTaskUnit],
        temp_dir: Optional[Path] = None,
    ) -> Path:
        if len(units) == 1:
            unit = units[0]
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", unit.package.task_id).strip("-.")
            prefix = f"ptdbench-harbor-{safe_id or 'task'}-{unit.evaluation.level}-"
        else:
            prefix = f"ptdbench-harbor-batch-{len(units)}-"
        parent = ensure_dir(temp_dir).resolve() if temp_dir is not None else None
        return Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=None if parent is None else str(parent),
            )
        ).resolve()


__all__ = [
    "HarborAdapterError",
    "HarborEvaluationConfig",
    "HarborEvaluationReport",
    "HarborEvaluator",
    "HarborPreparer",
    "HarborResultMapper",
    "HarborRun",
    "HarborRunner",
    "HarborShell",
    "HarborTaskMaterializer",
    "HarborTaskUnit",
    "PreparedHarborTask",
]
