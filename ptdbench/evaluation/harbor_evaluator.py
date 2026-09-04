from __future__ import annotations

import json
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
    CommandResult,
    EvaluationPaths,
    EvaluationResult,
    TaskPackage,
)
from ..tasks import SchemaError, load_task_package
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


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
        self._write_environment(package, evaluation, task_dir, config)
        self._write_tests(evaluation, task_dir, config)
        self._write_task_config(package, evaluation, task_dir, config)
        self._write_manifest(package, evaluation, task_dir, config)

    @staticmethod
    def _write_instruction(evaluation: EvaluationPaths, task_dir: Path) -> None:
        instruction = evaluation.instruction_md.read_text(encoding="utf-8")
        write_text(task_dir / "instruction.md", instruction)

    def _write_environment(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        environment_dir = task_dir / "environment"
        copy_tree(package.paths.environment, environment_dir)

        build_context = ensure_dir(environment_dir / ".ptdbench")
        copy_tree(package.paths.workspace, build_context / "workspace")
        public_dir = ensure_dir(build_context / "public")
        shutil.copy2(evaluation.instruction_md, public_dir / "instruction.md")

        source_dockerfile = package.paths.environment / self._dockerfile(package)
        dockerfile = source_dockerfile.read_text(encoding="utf-8").rstrip()
        dockerfile += self._debian_mirror_layer(config.debian_mirror)
        dockerfile += """

# PTDBench Harbor adapter: workspace 和公开 instruction 会进入 agent 镜像；
# oracle/verifier 只会在 agent 结束后由 Harbor 上传到 /tests。
COPY --chmod=0777 .ptdbench/workspace/ /workspace/
COPY --chmod=0444 .ptdbench/public/ /ptdbench/
WORKDIR /workspace
"""
        write_text(environment_dir / "Dockerfile", dockerfile)
        self._allow_adapter_context(environment_dir / ".dockerignore")

    @staticmethod
    def _debian_mirror_layer(mirror: Optional[str]) -> str:
        if mirror is None:
            return ""
        mirror_value = shlex.quote(mirror)
        return f"""

# PTDBench Harbor adapter: Harbor 会在容器启动后安装 agent 依赖，提前切换 Debian 源。
RUN set -eu; \\
    mirror={mirror_value}; \\
    for sources in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \\
        [ -f "$sources" ] || continue; \\
        sed -i \\
            -e "s|http://deb.debian.org/debian-security|$mirror/debian-security|g" \\
            -e "s|https://deb.debian.org/debian-security|$mirror/debian-security|g" \\
            -e "s|http://security.debian.org/debian-security|$mirror/debian-security|g" \\
            -e "s|https://security.debian.org/debian-security|$mirror/debian-security|g" \\
            -e "s|http://deb.debian.org/debian|$mirror/debian|g" \\
            -e "s|https://deb.debian.org/debian|$mirror/debian|g" \\
            "$sources"; \\
    done
"""

    @staticmethod
    def _dockerfile(package: TaskPackage) -> Path:
        environment = package.metadata.get("environment")
        declared = environment.get("dockerfile") if isinstance(environment, dict) else None
        value = str(declared or "environment/Dockerfile")
        if value == "environment":
            value = "Dockerfile"
        elif value.startswith("environment/"):
            value = value[len("environment/") :]
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise SchemaError(
                "metadata.environment.dockerfile must stay under environment/: "
                f"{value}"
            )
        if not (package.paths.environment / path).is_file():
            raise SchemaError(f"Dockerfile not found: {package.paths.environment / path}")
        return path

    @staticmethod
    def _allow_adapter_context(dockerignore: Path) -> None:
        existing = dockerignore.read_text(encoding="utf-8") if dockerignore.exists() else ""
        suffix = "\n!.ptdbench/\n!.ptdbench/**\n"
        write_text(dockerignore, existing.rstrip() + suffix)

    def _write_tests(
        self,
        evaluation: EvaluationPaths,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        tests_dir = ensure_dir(task_dir / "tests")
        copy_tree(evaluation.verifier, tests_dir / "verifier")
        if evaluation.oracle.is_dir():
            copy_tree(evaluation.oracle, tests_dir / "oracle")
        else:
            shutil.copy2(evaluation.oracle, tests_dir / "oracle.md")

        test_script = tests_dir / "test.sh"
        write_text(test_script, self._verifier_wrapper())
        test_script.chmod(0o755)
        if config.verifier_mode == "separate":
            self._write_separate_verifier_context(task_dir, tests_dir)

    @staticmethod
    def _write_separate_verifier_context(
        task_dir: Path,
        tests_dir: Path,
    ) -> None:
        """Harbor separate verifier 从 tests/ 构建镜像，需复制原环境上下文。"""
        environment_dir = task_dir / "environment"
        for source in environment_dir.iterdir():
            if source.name == "Dockerfile":
                continue
            target = tests_dir / source.name
            if target.exists():
                continue
            if source.is_dir():
                copy_tree(source, target)
            else:
                shutil.copy2(source, target)

        base_dockerfile = (environment_dir / "Dockerfile").read_text(encoding="utf-8")
        verifier_dockerfile = base_dockerfile.rstrip() + """

# Harbor separate verifier context: tests are installed only in the verifier.
COPY --chmod=0777 verifier/ /tests/verifier/
COPY --chmod=0777 test.sh /tests/test.sh
"""
        oracle = tests_dir / "oracle"
        if oracle.is_dir():
            verifier_dockerfile += "COPY --chmod=0444 oracle/ /tests/oracle/\n"
        else:
            verifier_dockerfile += "COPY --chmod=0444 oracle.md /tests/oracle.md\n"
        write_text(tests_dir / "Dockerfile", verifier_dockerfile)
        dockerignore = tests_dir / ".dockerignore"
        existing = dockerignore.read_text(encoding="utf-8") if dockerignore.exists() else ""
        write_text(
            dockerignore,
            existing.rstrip()
            + "\n!.ptdbench/\n!.ptdbench/**\n!verifier/\n!oracle/\n!oracle.md\n!test.sh\n",
        )

    @staticmethod
    def _verifier_wrapper() -> str:
        return r'''#!/usr/bin/env bash
set +e

bash "$PTDBENCH_VERIFIER/test.sh"
verifier_status=$?

# 结构化结果放进 verifier 日志，确保远程 Harbor environment 也会下载回来。
result_file="$PTDBENCH_RESULTS/result.json"
if [ -f "$result_file" ]; then
  cp "$result_file" /logs/verifier/ptdbench-result.json
fi

# Harbor 只消费 reward 文件；PTDBench 的完整 metrics 仍由 result.json 承载。
score=""
if [ -f "$result_file" ]; then
  score="$(tr '\n' ' ' < "$result_file" | sed -nE 's/.*"score"[[:space:]]*:[[:space:]]*(-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?).*/\1/p')"
fi
if ! printf '%s' "$score" | grep -Eq '^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'; then
  if [ "$verifier_status" -eq 0 ]; then
    score="1"
  else
    score="0"
  fi
fi

printf '%s\n' "$score" > /logs/verifier/reward.txt
exit "$verifier_status"
'''

    def _write_task_config(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        task_dir: Path,
        config: HarborEvaluationConfig,
    ) -> None:
        oracle_path = "/tests/oracle.md" if evaluation.oracle.is_file() else "/tests/oracle"
        task_name = harbor_task_name(package.task_id, evaluation.level)
        title = str(package.metadata.get("title") or package.task_id)
        verifier_env = {
            "PTDBENCH_TASK_ID": package.task_id,
            "PTDBENCH_EVALUATION_LEVEL": evaluation.level,
            "PTDBENCH_WORKSPACE": "/workspace",
            "PTDBENCH_RESULTS": "/logs/artifacts",
            "PTDBENCH_VISIBLE_TASK": "/ptdbench",
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

        task_toml = f'''schema_version = "1.4"

artifacts = ["/workspace"]

[task]
name = {self._toml_string(task_name)}
description = {self._toml_string(title)}

[metadata]
ptdbench_task_id = {self._toml_string(package.task_id)}
ptdbench_evaluation_level = {self._toml_string(evaluation.level)}

[agent]
network_mode = {self._toml_string(config.agent_network)}

[verifier]
environment_mode = {self._toml_string(config.verifier_mode)}
network_mode = {self._toml_string(config.verifier_network)}

[environment]
workdir = "/workspace"

[environment.env]
PTDBENCH_TASK_ID = {self._toml_string(package.task_id)}
PTDBENCH_EVALUATION_LEVEL = {self._toml_string(evaluation.level)}
PTDBENCH_WORKSPACE = "/workspace"
PTDBENCH_RESULTS = "/logs/artifacts"
PTDBENCH_VISIBLE_TASK = "/ptdbench"
PTDBENCH_TASK_ROOT = "/ptdbench"
PTDBENCH_STAGE = "agent"

[verifier.env]
{self._toml_env(verifier_env)}
'''
        write_text(task_dir / "task.toml", task_toml)

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
            "verifier_mode": config.verifier_mode,
            "debian_mirror": config.debian_mirror,
            "source_task": str(package.paths.root),
        }
        write_text(
            task_dir / "adapter-manifest.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class HarborRunner:
    """通过 CLI 启动 Harbor，避免与 Harbor 的 Python 版本和内部 API 耦合。"""

    _TASK_PATH_ARGS = {"-p", "--path"}

    def run(
        self,
        prepared: PreparedHarborTask,
        config: HarborEvaluationConfig,
    ) -> HarborRun:
        harbor_args = list(config.harbor_args)
        self._validate_harbor_args(harbor_args)

        job_name = self._option_value(harbor_args, "--job-name")
        jobs_dir_value = self._option_value(harbor_args, "-o", "--jobs-dir")
        config_path = self._option_value(harbor_args, "-c", "--config")
        injected_args = []
        # 没有 job config 时补全结果位置；使用 config 时完全尊重 Harbor 的配置。
        if config_path is None:
            if job_name is None:
                job_name = self._job_name()
                injected_args.extend(["--job-name", job_name])
            if jobs_dir_value is None:
                jobs_dir_value = "jobs"
                injected_args.extend(["--jobs-dir", jobs_dir_value])

        caller_cwd = Path.cwd().resolve()
        argv = [
            *config.harbor_command,
            "run",
            "--path",
            str(prepared.tasks_dir),
            *injected_args,
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
            trial.get("verifier_environment_mode") or config.verifier_mode
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


class HarborEvaluator:
    """PTDBench 的 Harbor 执行入口，与 TaskEvaluator 并行存在。"""

    def __init__(
        self,
        materializer: Optional[HarborTaskMaterializer] = None,
        runner: Optional[HarborRunner] = None,
        result_mapper: Optional[HarborResultMapper] = None,
    ) -> None:
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
        """把一批 task 物化进同一个 Harbor job，并返回其中的所有 trial。"""
        config = config or HarborEvaluationConfig()
        units = [
            self._load_unit(task_root, config.evaluation_level, strict)
            for task_root in self._normalize_task_roots(task_roots)
        ]
        root = self._make_run_dir(units, config.temp_dir)
        try:
            print(
                "[ptdbench] Materializing "
                + ", ".join(
                    f"{unit.package.task_id}:{unit.evaluation.level}" for unit in units
                ),
                file=sys.stderr,
                flush=True,
            )
            prepared = self.materializer.materialize(units, root, config)
            run = self.runner.run(prepared, config)
            print(
                f"[ptdbench] Reading Harbor results from {run.job_dir}",
                file=sys.stderr,
                flush=True,
            )
            return self.result_mapper.map(prepared, run, config)
        finally:
            # Harbor job 保存在原生 --jobs-dir；这里只清理在线转换出的临时 task。
            shutil.rmtree(root, ignore_errors=True)

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
    "HarborResultMapper",
    "HarborRun",
    "HarborRunner",
    "HarborTaskMaterializer",
    "HarborTaskUnit",
    "PreparedHarborTask",
]
