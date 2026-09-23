"""environment 段的两种形态：统一镜像的 conda 环境，和 raw 任务的 legacy dockerfile。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdbench.core.types import known_conda_envs
from ptdbench.tasks import SchemaError, load_task_package
from ptdbench.tasks.validator import validate_metadata


METADATA_HEAD = """title: Conda task
submitter:
  name: test
license: MIT
created_date: 2026-09-13
domain: rl
model:
  name: Qwen2.5-1.5B
dataset:
  name: taco
training:
  framework: verl
  algo: grpo
  script: train.sh
"""

METADATA_TAIL = """outcome:
  failure_type: test
  failure_reason: test
"""


def create_task(
    root: Path,
    environment: str,
    *,
    requirements: str | None = None,
    prepare: str | None = None,
) -> Path:
    """按新形态铺一个最小任务，environment 段由调用方给。"""
    task = root / "conda-task"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    # metadata 里的 training.script 指向它；校验会检查这个文件真的在 workspace 下。
    (task / "workspace" / "train.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (task / "verifier" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (task / "metadata.yaml").write_text(
        METADATA_HEAD + environment + METADATA_TAIL, encoding="utf-8"
    )
    if requirements is not None:
        (task / "environment" / "requirements.txt").write_text(requirements, encoding="utf-8")
    if prepare is not None:
        (task / "environment" / "prepare.sh").write_text(prepare, encoding="utf-8")
    return task


class EnvironmentPlanTests(unittest.TestCase):
    def _load(self, environment: str, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment, **kwargs))

    # --- loader -------------------------------------------------------------

    def test_conda_task_picks_up_the_conventional_paths(self) -> None:
        # requirements.txt 和 prepare.sh 几乎每个任务都有；逼着每份 metadata.yaml
        # 再抄一遍路径只会增加抄错的机会。
        package = self._load(
            "environment:\n  conda: verl\n",
            requirements="torch==2.9.0\n",
            prepare="#!/usr/bin/env bash\n",
        )
        plan = package.environment
        self.assertTrue(package.report.ok, package.report.to_dict())
        self.assertEqual(plan.conda, "verl")
        self.assertFalse(plan.is_legacy)
        self.assertEqual(plan.requirements.name, "requirements.txt")
        self.assertEqual(plan.prepare.name, "prepare.sh")

    def test_explicit_paths_win_over_the_convention(self) -> None:
        package = self._load(
            "environment:\n"
            "  conda: verl\n"
            "  requirements: environment/pinned.txt\n"
            "  prepare: environment/fetch.sh\n",
        )
        self.assertEqual(package.environment.requirements.name, "pinned.txt")
        self.assertEqual(package.environment.prepare.name, "fetch.sh")

    def test_absent_files_leave_the_plan_empty_rather_than_dangling(self) -> None:
        # plan 里带一个不存在的路径，会把"任务没写 prepare.sh"和"写了但路径错了"
        # 混成同一种状态，而后者必须报错。
        plan = self._load("environment:\n  conda: slime-v0.3.1\n").environment
        self.assertIsNone(plan.requirements)
        self.assertIsNone(plan.prepare)

    def test_legacy_dockerfile_task_still_loads(self) -> None:
        package = self._load("environment:\n  dockerfile: environment/Dockerfile\n")
        self.assertTrue(package.report.ok, package.report.to_dict())
        self.assertTrue(package.environment.is_legacy)
        self.assertIsNone(package.environment.conda)

    def test_unknown_conda_env_is_a_schema_error(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            self._load("environment:\n  conda: pytorch\n")
        message = str(ctx.exception)
        self.assertIn("environment.conda must be one of", message)
        # 报错要说清楚怎么修：新环境不是改个字符串就有的。
        self.assertIn("docker/envs/pytorch/", message)

    # --- validator branches -------------------------------------------------

    def _environment_issues(self, environment: dict) -> list:
        report = validate_metadata({"environment": environment})
        return [i.path for i in report.issues if i.path.startswith("metadata.environment")]

    def test_validator_accepts_every_environment_the_image_builds(self) -> None:
        # 唯一事实来源是 docker/envs/ 下的目录名——镜像就是照它构建的。
        names = known_conda_envs()
        self.assertTrue(names, "docker/envs/ should exist in a source checkout")
        for name in sorted(names):
            self.assertEqual(self._environment_issues({"conda": name}), [], name)

    def test_validator_rejects_an_environment_the_image_does_not_have(self) -> None:
        self.assertEqual(
            self._environment_issues({"conda": "pytorch"}),
            ["metadata.environment.conda"],
        )

    def test_validator_rejects_conda_and_dockerfile_together(self) -> None:
        # 两个都写时运行时只有一个生效，读者不该靠猜。
        self.assertEqual(
            self._environment_issues(
                {"conda": "verl", "dockerfile": "environment/Dockerfile"}
            ),
            ["metadata.environment.dockerfile"],
        )

    def test_validator_rejects_non_path_requirements_and_prepare(self) -> None:
        self.assertEqual(
            self._environment_issues({"conda": "verl", "requirements": ["a"]}),
            ["metadata.environment.requirements"],
        )
        self.assertEqual(
            self._environment_issues({"conda": "verl", "prepare": 3}),
            ["metadata.environment.prepare"],
        )

    def test_validator_keeps_legacy_default_for_absent_dockerfile(self) -> None:
        # 迁移前的任务多数不写 dockerfile，靠约定路径兜底；这条行为不能因为新分支变严。
        self.assertEqual(self._environment_issues({"image": "x"}), [])
        self.assertEqual(
            self._environment_issues({"dockerfile": ""}),
            ["metadata.environment.dockerfile"],
        )


class EnvironmentEnvTests(unittest.TestCase):
    """metadata.environment.env：逐任务不同的变量（统一镜像只带共用的那批）。"""

    def _load(self, environment: str):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment))

    def test_env_is_parsed_into_the_plan(self):
        package = self._load(
            "environment:\n"
            "  conda: verl\n"
            "  env:\n"
            '    CUDA_VISIBLE_DEVICES: "0,1"\n'
            '    NVIDIA_VISIBLE_DEVICES: "0,1"\n'
        )
        self.assertEqual(
            package.environment.env,
            {"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "0,1"},
        )

    def test_absent_env_is_an_empty_mapping(self):
        self.assertEqual(self._load("environment:\n  conda: verl\n").environment.env, {})

    def test_tasks_cannot_hijack_the_harness_channel(self):
        # PTDBENCH_* 是 harness 告诉容器"你是哪个 stage、用哪套环境"的通道。
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                "environment:\n  conda: verl\n  env:\n    PTDBENCH_STAGE: verifier\n"
            )
        self.assertIn("PTDBENCH_STAGE", str(ctx.exception))

    def test_unquoted_number_is_rejected_rather_than_coerced(self):
        # `CUDA_VISIBLE_DEVICES: 0` 在 YAML 里是 int。悄悄 str() 的话，
        # "只用 0 号卡" 和 "写错了" 就再也分不出来。
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                "environment:\n  conda: verl\n  env:\n    CUDA_VISIBLE_DEVICES: 0\n"
            )
        self.assertIn("CUDA_VISIBLE_DEVICES", str(ctx.exception))

    def test_scalar_env_is_rejected(self):
        with self.assertRaises(SchemaError):
            self._load("environment:\n  conda: verl\n  env: nope\n")


class EnvironmentEnvRuntimeTests(unittest.TestCase):
    def _request(self, **kwargs):
        from ptdbench.runtimes import SandboxRequest

        root = Path("/host/run")
        defaults = dict(
            stage="agent",
            task_id="t",
            host_task_root=root / "task",
            host_visible_task=root / "visible",
            host_workspace=root / "workspace",
            host_results=root / "results",
        )
        defaults.update(kwargs)
        return SandboxRequest(**defaults)

    def test_task_env_reaches_the_command_but_cannot_shadow_the_harness(self):
        from ptdbench.runtimes.base import CommandSpec
        from ptdbench.runtimes.providers.docker import DockerRuntime

        request = self._request(conda_env="verl", task_env={"CUDA_VISIBLE_DEVICES": "0,1"})
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))

        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(merged.env["PTDBENCH_STAGE"], "agent")
        self.assertEqual(merged.env["PTDBENCH_CONDA_ENV"], "verl")

    def test_caller_env_still_wins_over_the_task_declaration(self):
        from ptdbench.runtimes.base import CommandSpec
        from ptdbench.runtimes.providers.docker import DockerRuntime

        request = self._request(
            task_env={"CUDA_VISIBLE_DEVICES": "0,1"},
            env={"CUDA_VISIBLE_DEVICES": "2,3"},
        )
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))
        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "2,3")


if __name__ == "__main__":
    unittest.main()
