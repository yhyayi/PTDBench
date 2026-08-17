from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ptdbench.cli import build_parser
from ptdbench.core.harbor_types import (
    HarborEvaluationConfig,
    HarborRun,
    PreparedHarborTask,
)
from ptdbench.core.types import CommandResult
from ptdbench.evaluation.harbor_evaluator import (
    HarborAdapterError,
    HarborEvaluator,
    HarborResultMapper,
    HarborRunner,
    HarborTaskMaterializer,
)
from ptdbench.tasks import load_task_package


ROOT = Path(__file__).resolve().parents[1]
SWE_TASK = ROOT / "example" / "swe_slugfix"


class HarborTaskMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = load_task_package(SWE_TASK)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def materialize(self, level: str, mode: str = "separate") -> PreparedHarborTask:
        return HarborTaskMaterializer().materialize(
            self.package,
            self.package.paths.evaluations[level],
            self.root / f"{level}-{mode}",
            HarborEvaluationConfig(evaluation_level=level, verifier_mode=mode),
        )

    def test_only_selected_evaluation_is_materialized(self):
        prepared = self.materialize("issue")

        self.assertTrue((prepared.task_dir / "tests" / "oracle.md").is_file())
        self.assertFalse((prepared.task_dir / "tests" / "oracle").exists())
        self.assertEqual(
            (prepared.task_dir / "instruction.md").read_text(encoding="utf-8"),
            self.package.paths.evaluations["issue"].instruction_md.read_text(
                encoding="utf-8"
            ),
        )

    def test_hidden_files_do_not_enter_agent_environment(self):
        prepared = self.materialize("solution")
        environment = prepared.task_dir / "environment"

        self.assertFalse((environment / "oracle").exists())
        self.assertFalse((environment / "verifier").exists())
        self.assertTrue((prepared.task_dir / "tests" / "oracle").is_dir())
        self.assertTrue((prepared.task_dir / "tests" / "verifier" / "test.sh").is_file())
        self.assertTrue(
            (environment / ".ptdbench" / "workspace" / "toy_swe" / "slugify.py").is_file()
        )

    def test_shared_and_separate_are_harbor_configuration_only(self):
        shared = self.materialize("solution", "shared")
        separate = self.materialize("solution", "separate")
        shared_toml = (shared.task_dir / "task.toml").read_text(encoding="utf-8")
        separate_toml = (separate.task_dir / "task.toml").read_text(encoding="utf-8")

        self.assertIn('environment_mode = "shared"', shared_toml)
        self.assertIn('environment_mode = "separate"', separate_toml)
        self.assertIn('artifacts = ["/workspace"]', shared_toml)
        self.assertIn('artifacts = ["/workspace"]', separate_toml)
        self.assertNotIn("PTDBENCH_ORACLE", shared_toml.split("[verifier.env]")[0])
        self.assertTrue((separate.task_dir / "tests" / "Dockerfile").is_file())
        self.assertFalse((shared.task_dir / "tests" / "Dockerfile").exists())

    def test_wrapper_preserves_structured_result_and_writes_harbor_reward(self):
        prepared = self.materialize("solution")
        wrapper = (prepared.task_dir / "tests" / "test.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("ptdbench-result.json", wrapper)
        self.assertIn("/logs/verifier/reward.txt", wrapper)
        self.assertIn('bash "$PTDBENCH_VERIFIER/test.sh"', wrapper)

    def test_common_agent_and_verifier_settings_enter_task_config(self):
        config = HarborEvaluationConfig(
            judge_model="judge-model",
            agent_network="no-network",
            verifier_network="public",
        )
        prepared = HarborTaskMaterializer().materialize(
            self.package,
            self.package.paths.evaluations["solution"],
            self.root / "common-config",
            config,
        )
        task_toml = (prepared.task_dir / "task.toml").read_text(encoding="utf-8")

        self.assertIn('PTDBENCH_JUDGE_MODEL = "judge-model"', task_toml)
        self.assertIn("${OPENAI_BASE_URL:-https://api.openai.com/v1}", task_toml)
        self.assertIn("${OPENAI_API_KEY:-}", task_toml)
        self.assertIn('network_mode = "no-network"', task_toml)
        self.assertIn('network_mode = "public"', task_toml)

    def test_debian_mirror_is_written_before_agent_setup(self):
        prepared = HarborTaskMaterializer().materialize(
            self.package,
            self.package.paths.evaluations["solution"],
            self.root / "debian-mirror",
            HarborEvaluationConfig(
                debian_mirror="https://mirrors.tuna.tsinghua.edu.cn/"
            ),
        )
        dockerfile = (prepared.task_dir / "environment" / "Dockerfile").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "mirror=https://mirrors.tuna.tsinghua.edu.cn", dockerfile
        )
        self.assertIn("$mirror/debian-security", dockerfile)
        self.assertLess(
            dockerfile.index("mirrors.tuna.tsinghua.edu.cn"),
            dockerfile.index("COPY --chmod=0777 .ptdbench/workspace/"),
        )


class HarborRunnerTests(unittest.TestCase):
    @patch.object(HarborRunner, "_run_streaming")
    def test_runner_builds_a_secret_free_harbor_command(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "done"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = PreparedHarborTask(
                root=root,
                task_dir=root / "task",
            )
            prepared.task_dir.mkdir()
            config = HarborEvaluationConfig(
                harbor_command=("uv", "run", "harbor"),
                harbor_args=(
                    "--job-name",
                    "native-job",
                    "--jobs-dir",
                    str(root / "jobs"),
                    "--agent",
                    "codex",
                    "--model",
                    "gpt-test",
                    "--env",
                    "docker",
                    "--agent-kwarg",
                    "reasoning_effort=high",
                    "--n-concurrent",
                    "2",
                    "--agent-env",
                    "CUSTOM_TOKEN=raw-secret",
                ),
            )

            harbor_run = HarborRunner().run(prepared, config)

        argv = run_mock.call_args.args[0]
        self.assertEqual(argv[:4], ["uv", "run", "harbor", "run"])
        self.assertIn("--path", argv)
        self.assertIn("--jobs-dir", argv)
        self.assertIn("--model", argv)
        self.assertIn("reasoning_effort=high", argv)
        self.assertIn("--n-concurrent", argv)
        self.assertIn("CUSTOM_TOKEN=raw-secret", argv)
        self.assertNotIn("raw-secret", " ".join(harbor_run.command.argv))
        self.assertEqual(run_mock.call_args.args[1], Path.cwd())
        self.assertEqual(
            harbor_run.job_dir,
            (root / "jobs" / "native-job").resolve(),
        )

    def test_only_materialized_task_path_is_reserved(self):
        config = HarborEvaluationConfig(harbor_args=("--path", "other-task"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = PreparedHarborTask(root, root / "task")
            with self.assertRaisesRegex(ValueError, "task materializer"):
                HarborRunner().run(prepared, config)

    @patch.object(HarborRunner, "_run_streaming")
    def test_job_config_is_forwarded_without_overriding_its_output(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "Results written to configured/job/result.json\n"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = PreparedHarborTask(root, root / "task")
            prepared.task_dir.mkdir()

            harbor_run = HarborRunner().run(
                prepared,
                HarborEvaluationConfig(harbor_args=("--config", "job.yaml")),
            )

        argv = run_mock.call_args.args[0]
        self.assertEqual(argv[-2:], ["--config", "job.yaml"])
        self.assertNotIn("--jobs-dir", argv)
        self.assertNotIn("--job-name", argv)
        self.assertEqual(
            harbor_run.job_dir,
            (Path.cwd() / "configured" / "job").resolve(),
        )

    def test_cli_forwards_native_harbor_arguments_unchanged(self):
        args, harbor_args = build_parser().parse_known_args(
            [
                "harbor-evaluate",
                "task",
                "--evaluation-level",
                "issue",
                "--judge-model",
                "judge-model",
                "--harbor-temp-dir",
                "custom-temp",
                "--agent",
                "codex",
                "--model",
                "gpt-test",
                "--env",
                "daytona",
                "--n-concurrent",
                "8",
            ]
        )

        self.assertEqual(args.evaluation_level, "issue")
        self.assertEqual(args.judge_model, "judge-model")
        self.assertEqual(args.harbor_temp_dir, Path("custom-temp"))
        self.assertEqual(
            harbor_args,
            [
                "--agent",
                "codex",
                "--model",
                "gpt-test",
                "--env",
                "daytona",
                "--n-concurrent",
                "8",
            ],
        )

    def test_temp_run_directory_uses_configured_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "harbor-tmp"
            run_dir = HarborEvaluator._make_run_dir("task/id", "solution", parent)
            try:
                self.assertEqual(run_dir.parent, parent.resolve())
                self.assertTrue(run_dir.name.startswith("ptdbench-harbor-task-id-solution-"))
            finally:
                shutil.rmtree(run_dir)


class HarborResultMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = load_task_package(SWE_TASK)
        self.evaluation = self.package.paths.evaluations["solution"]
        self.job_dir = self.root / "job"
        self.trial_dir = self.job_dir / "trial-1"
        (self.trial_dir / "artifacts").mkdir(parents=True)
        (self.trial_dir / "verifier").mkdir()
        shutil.copytree(
            self.package.paths.workspace,
            self.trial_dir / "artifacts" / "workspace",
        )
        self._write_manifest("ok")
        (self.trial_dir / "verifier" / "ptdbench-result.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "score": 0.75,
                    "metrics": {"tests_run": 4},
                }
            ),
            encoding="utf-8",
        )
        (self.trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_name": "trial-1",
                    "verifier_environment_mode": "separate",
                    "verifier_result": {"rewards": {"reward": 0.75}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        (self.job_dir / "result.json").write_text(
            json.dumps({"n_total_trials": 1, "stats": {"n_completed_trials": 1}}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_manifest(self, status: str) -> None:
        (self.trial_dir / "artifacts" / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "source": "/workspace",
                        "destination": "artifacts/workspace",
                        "type": "directory",
                        "status": status,
                    }
                ]
            ),
            encoding="utf-8",
        )

    def _map(self):
        command = CommandResult(
            argv=["harbor", "run"],
            returncode=0,
            stdout="",
            stderr="",
            timed_out=False,
            duration_s=1.0,
            cwd=str(self.root),
        )
        report = HarborResultMapper().map(
            self.package,
            self.evaluation,
            HarborRun(command=command, job_dir=self.job_dir),
            HarborEvaluationConfig(),
            workspace_before_hash="before",
        )
        return report.trials[0]

    def test_maps_harbor_trial_and_ptdbench_metrics(self):
        result = self._map()

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.metrics["tests_run"], 4)
        self.assertEqual(result.metrics["harbor_rewards"], {"reward": 0.75})
        self.assertTrue((self.trial_dir / "report.json").is_file())
        self.assertTrue(
            (self.trial_dir / "ptdbench-harbor-command.json").is_file()
        )

    def test_workspace_artifact_failure_fails_closed(self):
        self._write_manifest("failed")

        result = self._map()

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertIn("/workspace", result.metrics["infrastructure_errors"][0])

    def test_maps_all_trials_into_a_job_report(self):
        second_trial = self.job_dir / "trial-2"
        shutil.copytree(self.trial_dir, second_trial)
        second_result = json.loads(
            (second_trial / "result.json").read_text(encoding="utf-8")
        )
        second_result["trial_name"] = "trial-2"
        (second_trial / "result.json").write_text(
            json.dumps(second_result), encoding="utf-8"
        )
        command = CommandResult(
            argv=["harbor", "run"],
            returncode=0,
            stdout="",
            stderr="",
            timed_out=False,
            duration_s=1.0,
            cwd=str(self.root),
        )

        report = HarborResultMapper().map(
            self.package,
            self.evaluation,
            HarborRun(command=command, job_dir=self.job_dir),
            HarborEvaluationConfig(),
            workspace_before_hash="before",
        )

        self.assertEqual(len(report.trials), 2)
        self.assertTrue(report.passed)
        self.assertEqual(report.score, 0.75)
        self.assertTrue((self.job_dir / "ptdbench-report.json").is_file())

    def test_reads_legacy_embedded_trial_results(self):
        (self.trial_dir / "result.json").unlink()
        (self.job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "trial_name": "trial-1",
                            "verifier_environment_mode": "separate",
                            "verifier_result": {"rewards": {"reward": 0.75}},
                            "exception_info": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = self._map()

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)

    def test_failed_harbor_command_without_result_has_a_clear_error(self):
        (self.job_dir / "result.json").unlink()
        command = CommandResult(
            argv=["harbor", "run"],
            returncode=2,
            stdout="",
            stderr="invalid option",
            timed_out=False,
            duration_s=1.0,
            cwd=str(self.root),
        )

        with self.assertRaisesRegex(HarborAdapterError, "exit code 2"):
            HarborResultMapper().map(
                self.package,
                self.evaluation,
                HarborRun(command=command, job_dir=self.job_dir),
                HarborEvaluationConfig(),
                workspace_before_hash="before",
            )


if __name__ == "__main__":
    unittest.main()
