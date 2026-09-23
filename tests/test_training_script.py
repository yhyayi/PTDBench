"""metadata.training.script：任务的训练入口，由任务自己声明而不是靠 glob 猜。

以前 docker/debug-env.sh 的 `run` 用 `workspace/examples/*/{*_rl,run_*}.sh` 这组
glob 去找训练脚本。那组 pattern 描述的是 slime 的目录布局：RLVE 的入口在
`scripts/training/<model>/rlve.sh`，一个都匹配不上，那几个任务的 `run` 因此根本
起不来。现在入口写在任务卡上，这里校验的就是那条路径本身站得住。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdbench.tasks import load_task_package
from ptdbench.tasks.validator import validate_metadata


METADATA = """title: Training task
submitter:
  name: test
license: MIT
created_date: 2026-09-23
domain: rl
model:
  name: Qwen2.5-0.5B-Instruct
dataset:
  name: test
training:
{training}
environment:
  conda: verl
outcome:
  failure_type: test
  failure_reason: test
"""


def create_task(root: Path, training: str, *, launch: str | None = "train.sh") -> Path:
    """铺一个最小任务，training 段由调用方给。"""
    task = root / "training-task"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    (task / "verifier" / "test.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    if launch is not None:
        path = task / "workspace" / launch
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (task / "metadata.yaml").write_text(
        METADATA.format(training=training), encoding="utf-8"
    )
    return task


class TrainingScriptPathTests(unittest.TestCase):
    """路径这半边要有 TaskPaths 才能判断，所以走完整的 load_task_package。"""

    def _issues(self, training: str, **kwargs) -> list:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        package = load_task_package(create_task(Path(temp.name), training, **kwargs))
        return [
            (issue.severity, issue.path)
            for issue in package.report.issues
            if issue.path.startswith("metadata.training")
        ]

    def test_declared_script_that_exists_is_clean(self) -> None:
        self._assert_clean(
            "  framework: verl\n  algo: grpo\n  script: examples/run.sh\n",
            launch="examples/run.sh",
        )

    def test_script_outside_examples_is_just_as_valid(self) -> None:
        # 这就是 glob 猜不到的那一类：RLVE 把入口放在 scripts/training/ 下面。
        self._assert_clean(
            "  framework: slime\n  algo: grpo\n"
            "  script: scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh\n",
            launch="scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh",
        )

    def test_args_ride_along_with_the_script(self) -> None:
        # rlve.sh 在 `$# -lt 3` 时直接退出：没有参数，那条路径是跑不起来的。
        self._assert_clean(
            "  framework: slime\n  algo: grpo\n  script: train.sh\n"
            "  script_args:\n    - ptdbench\n    - solution_rerun\n    - _\n"
        )

    def test_missing_file_is_an_error_not_a_surprise_at_run_time(self) -> None:
        # 路径写错要在校验时就炸；拖到 `run` 那一刻通常已经是几小时之后了。
        self.assertEqual(
            self._issues("  framework: verl\n  algo: grpo\n  script: nope.sh\n"),
            [("error", "metadata.training.script")],
        )

    def test_script_must_stay_inside_the_workspace(self) -> None:
        for script in ("/opt/train.sh", "../train.sh"):
            with self.subTest(script=script):
                self.assertEqual(
                    self._issues(f"  framework: verl\n  algo: grpo\n  script: {script}\n"),
                    [("error", "metadata.training.script")],
                )

    def test_missing_script_is_a_warning_when_the_task_trains(self) -> None:
        issues = self._issues("  framework: verl\n  algo: grpo\n")
        self.assertEqual(issues, [("warning", "metadata.training.script")])

    def test_a_task_without_training_needs_no_script(self) -> None:
        # 纯代码修复任务（framework: none）本来就没有训练脚本。
        self._assert_clean("  framework: none\n  algo: none\n", launch=None)

    def _assert_clean(self, training: str, **kwargs) -> None:
        self.assertEqual(self._issues(training, **kwargs), [])


class TrainingScriptArgsTests(unittest.TestCase):
    """script_args 的形状，不需要文件系统。"""

    def _issues(self, training: dict) -> list:
        report = validate_metadata({"training": training})
        return [
            (issue.severity, issue.path)
            for issue in report.issues
            if issue.path.startswith("metadata.training.script")
        ]

    def test_a_string_is_rejected_rather_than_split(self) -> None:
        # "a b c" 是一个参数还是三个，取决于谁来分词。只认列表，读者就不用猜。
        self.assertEqual(
            self._issues({"framework": "verl", "algo": "grpo", "script": "t.sh",
                          "script_args": "ptdbench solution_rerun _"}),
            [("error", "metadata.training.script_args")],
        )

    def test_unquoted_number_is_rejected_rather_than_coerced(self) -> None:
        # YAML 里 `- 0` 是 int；悄悄 str() 的话就看不出任务卡写错了。
        self.assertEqual(
            self._issues({"framework": "verl", "algo": "grpo", "script": "t.sh",
                          "script_args": ["ptdbench", 0]}),
            [("error", "metadata.training.script_args[1]")],
        )

    def test_args_without_a_script_have_nothing_to_apply_to(self) -> None:
        self.assertEqual(
            self._issues({"framework": "none", "algo": "none", "script_args": ["x"]}),
            [("error", "metadata.training.script_args")],
        )


if __name__ == "__main__":
    unittest.main()
