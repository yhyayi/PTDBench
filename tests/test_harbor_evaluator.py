from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ptdbench.cli import build_parser, expand_task_roots, main as cli_main
from ptdbench.core.harbor_types import (
    HarborEvaluationConfig,
    HarborRun,
    HarborTaskUnit,
    PreparedHarborTask,
)
from ptdbench.core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    CommandResult,
)
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


def make_unit(package, level: str, workspace_before_hash: str = "before") -> HarborTaskUnit:
    return HarborTaskUnit(
        package=package,
        evaluation=package.paths.evaluations[level],
        workspace_before_hash=workspace_before_hash,
    )


class HarborTaskMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = load_task_package(SWE_TASK)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def materialize(self, level: str, mode: str = "separate") -> Path:
        unit = make_unit(self.package, level)
        prepared = HarborTaskMaterializer().materialize(
            [unit],
            self.root / f"{level}-{mode}",
            HarborEvaluationConfig(evaluation_level=level, verifier_mode=mode),
        )
        return prepared.task_dir(unit)

    def test_only_selected_evaluation_is_materialized(self):
        task_dir = self.materialize("issue")

        self.assertTrue((task_dir / "tests" / "oracle.md").is_file())
        self.assertFalse((task_dir / "tests" / "oracle").exists())
        self.assertEqual(
            (task_dir / "instruction.md").read_text(encoding="utf-8"),
            self.package.paths.evaluations["issue"].instruction_md.read_text(
                encoding="utf-8"
            ),
        )

    def test_hidden_files_do_not_enter_agent_environment(self):
        task_dir = self.materialize("solution")
        environment = task_dir / "environment"

        self.assertFalse((environment / "oracle").exists())
        self.assertFalse((environment / "verifier").exists())
        self.assertTrue((task_dir / "tests" / "oracle").is_dir())
        self.assertTrue((task_dir / "tests" / "verifier" / "test.sh").is_file())
        self.assertTrue(
            (environment / ".ptdbench" / "workspace" / "toy_swe" / "slugify.py").is_file()
        )

    def test_shared_and_separate_are_harbor_configuration_only(self):
        shared = self.materialize("solution", "shared")
        separate = self.materialize("solution", "separate")
        shared_toml = (shared / "task.toml").read_text(encoding="utf-8")
        separate_toml = (separate / "task.toml").read_text(encoding="utf-8")

        self.assertIn('environment_mode = "shared"', shared_toml)
        self.assertIn('environment_mode = "separate"', separate_toml)
        self.assertIn('artifacts = ["/workspace"]', shared_toml)
        self.assertIn('artifacts = ["/workspace"]', separate_toml)
        self.assertNotIn("PTDBENCH_ORACLE", shared_toml.split("[verifier.env]")[0])
        self.assertTrue((separate / "tests" / "Dockerfile").is_file())
        self.assertFalse((shared / "tests" / "Dockerfile").exists())

    def test_wrapper_preserves_structured_result_and_writes_harbor_reward(self):
        task_dir = self.materialize("solution")
        wrapper = (task_dir / "tests" / "test.sh").read_text(encoding="utf-8")

        self.assertIn("ptdbench-result.json", wrapper)
        self.assertIn("/logs/verifier/reward.txt", wrapper)
        self.assertIn('bash "$PTDBENCH_VERIFIER/test.sh"', wrapper)

    def test_each_task_gets_its_own_directory_under_one_batch(self):
        other_root = self.root / "swe_slugfix_copy"
        shutil.copytree(SWE_TASK, other_root)
        units = [
            make_unit(self.package, "solution"),
            make_unit(load_task_package(other_root), "solution"),
        ]

        prepared = HarborTaskMaterializer().materialize(
            units,
            self.root / "batch",
            HarborEvaluationConfig(),
        )

        self.assertEqual(prepared.tasks_dir, (self.root / "batch" / "tasks").resolve())
        for unit in units:
            self.assertTrue((prepared.task_dir(unit) / "task.toml").is_file())
        self.assertEqual(
            {path.name for path in prepared.tasks_dir.iterdir()},
            {unit.dir_name for unit in units},
        )

    def test_duplicate_task_names_are_rejected(self):
        units = [make_unit(self.package, "solution")] * 2

        with self.assertRaisesRegex(HarborAdapterError, "duplicate Harbor task name"):
            HarborTaskMaterializer().materialize(
                units,
                self.root / "duplicate",
                HarborEvaluationConfig(),
            )

    def test_common_agent_and_verifier_settings_enter_task_config(self):
        config = HarborEvaluationConfig(
            judge_model="judge-model",
            agent_network="no-network",
            verifier_network="public",
        )
        unit = make_unit(self.package, "solution")
        prepared = HarborTaskMaterializer().materialize(
            [unit],
            self.root / "common-config",
            config,
        )
        task_toml = (prepared.task_dir(unit) / "task.toml").read_text(encoding="utf-8")

        self.assertIn('PTDBENCH_JUDGE_MODEL = "judge-model"', task_toml)
        self.assertIn("${OPENAI_BASE_URL:-https://api.openai.com/v1}", task_toml)
        self.assertIn("${OPENAI_API_KEY:-}", task_toml)
        self.assertIn('network_mode = "no-network"', task_toml)
        self.assertIn('network_mode = "public"', task_toml)

    def test_debian_mirror_is_written_before_agent_setup(self):
        unit = make_unit(self.package, "solution")
        prepared = HarborTaskMaterializer().materialize(
            [unit],
            self.root / "debian-mirror",
            HarborEvaluationConfig(
                debian_mirror="https://mirrors.tuna.tsinghua.edu.cn/"
            ),
        )
        dockerfile = (
            prepared.task_dir(unit) / "environment" / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "mirror=https://mirrors.tuna.tsinghua.edu.cn", dockerfile
        )
        self.assertIn("$mirror/debian-security", dockerfile)
        self.assertLess(
            dockerfile.index("mirrors.tuna.tsinghua.edu.cn"),
            dockerfile.index("COPY --chmod=0777 .ptdbench/workspace/"),
        )


class HarborRunnerTests(unittest.TestCase):
    @staticmethod
    def _prepared(root: Path, create: bool = True) -> PreparedHarborTask:
        prepared = PreparedHarborTask(root=root, tasks_dir=root / "tasks")
        if create:
            prepared.tasks_dir.mkdir(parents=True)
        return prepared

    @patch.object(HarborRunner, "_run_streaming")
    def test_runner_builds_a_secret_free_harbor_command(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "done"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = self._prepared(root)
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
        self.assertEqual(argv[argv.index("--path") + 1], str(prepared.tasks_dir))
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
            prepared = self._prepared(Path(temp_dir), create=False)
            with self.assertRaisesRegex(ValueError, "task materializer"):
                HarborRunner().run(prepared, config)

    @patch.object(HarborRunner, "_run_streaming")
    def test_job_config_is_forwarded_without_overriding_its_output(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "Results written to configured/job/result.json\n"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = self._prepared(Path(temp_dir))

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

    def test_cli_accepts_several_task_roots_and_forwards_harbor_arguments(self):
        args, harbor_args = build_parser().parse_known_args(
            [
                "harbor-evaluate",
                "task-a",
                "task-b",
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

        self.assertEqual(args.task_root, ["task-a", "task-b"])
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
        package = load_task_package(SWE_TASK)
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "harbor-tmp"
            run_dir = HarborEvaluator._make_run_dir(
                [make_unit(package, "solution")],
                parent,
            )
            try:
                self.assertEqual(run_dir.parent, parent.resolve())
                self.assertTrue(
                    run_dir.name.startswith("ptdbench-harbor-swe_slugfix-solution-")
                )
            finally:
                shutil.rmtree(run_dir)

    def test_temp_run_directory_is_named_for_the_batch(self):
        package = load_task_package(SWE_TASK)
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "harbor-tmp"
            run_dir = HarborEvaluator._make_run_dir(
                [make_unit(package, "solution"), make_unit(package, "issue")],
                parent,
            )
            try:
                self.assertTrue(run_dir.name.startswith("ptdbench-harbor-batch-2-"))
            finally:
                shutil.rmtree(run_dir)


class ExpandTaskRootsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _task(self, path: Path) -> Path:
        path.mkdir(parents=True)
        (path / "metadata.yaml").write_text("id: x\n", encoding="utf-8")
        return path

    def test_task_directory_is_used_as_is(self):
        task = self._task(self.root / "task-a")

        self.assertEqual(expand_task_roots([str(task)]), [task])

    def test_parent_directory_expands_to_its_tasks(self):
        family = self.root / "family"
        second = self._task(family / "task-b")
        first = self._task(family / "task-a")
        (family / "not-a-task").mkdir()

        self.assertEqual(expand_task_roots([str(family)]), [first, second])

    def test_repeated_paths_are_submitted_once(self):
        family = self.root / "family"
        task = self._task(family / "task-a")

        roots = expand_task_roots([str(family), str(task)])

        self.assertEqual(roots, [task])

    def test_directory_without_tasks_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()

        with self.assertRaises(SystemExit):
            expand_task_roots([str(empty)])

    def test_missing_path_is_rejected(self):
        with self.assertRaises(SystemExit):
            expand_task_roots([str(self.root / "absent")])


class HarborResultMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = load_task_package(SWE_TASK)
        self.unit = make_unit(self.package, "solution")
        self.prepared = PreparedHarborTask(
            root=self.root / "prepared",
            tasks_dir=self.root / "prepared" / "tasks",
            units=[self.unit],
        )
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
                    "task_name": self.unit.harbor_task_name,
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

    def _command(self, returncode: int = 0, stderr: str = "") -> CommandResult:
        return CommandResult(
            argv=["harbor", "run"],
            returncode=returncode,
            stdout="",
            stderr=stderr,
            timed_out=False,
            duration_s=1.0,
            cwd=str(self.root),
        )

    def _report(self, prepared: PreparedHarborTask | None = None):
        return HarborResultMapper().map(
            prepared or self.prepared,
            HarborRun(command=self._command(), job_dir=self.job_dir),
            HarborEvaluationConfig(),
        )

    def _map(self):
        return self._report().trials[0]

    def test_maps_harbor_trial_and_ptdbench_metrics(self):
        result = self._map()

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.task_id, self.package.task_id)
        self.assertEqual(result.workspace_before_hash, "before")
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

        report = self._report()

        self.assertEqual(len(report.trials), 2)
        self.assertTrue(report.passed)
        self.assertEqual(report.score, 0.75)
        self.assertEqual(report.to_dict()["task_ids"], [self.package.task_id])
        self.assertTrue((self.job_dir / "ptdbench-report.json").is_file())

    def test_each_trial_is_mapped_back_to_its_own_task(self):
        other_root = self.root / "swe_slugfix_copy"
        shutil.copytree(SWE_TASK, other_root)
        other_unit = make_unit(load_task_package(other_root), "solution", "other-before")
        second_trial = self.job_dir / "trial-2"
        shutil.copytree(self.trial_dir, second_trial)
        (second_trial / "result.json").write_text(
            json.dumps(
                {
                    "trial_name": "trial-2",
                    "task_name": other_unit.harbor_task_name,
                    "verifier_environment_mode": "separate",
                    "verifier_result": {"rewards": {"reward": 0.75}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )

        report = self._report(prepared)

        by_task = {trial.task_id: trial for trial in report.trials}
        self.assertEqual(
            set(by_task),
            {self.package.task_id, other_unit.package.task_id},
        )
        self.assertEqual(by_task[self.package.task_id].workspace_before_hash, "before")
        self.assertEqual(
            by_task[other_unit.package.task_id].workspace_before_hash,
            "other-before",
        )
        self.assertEqual(
            report.to_dict()["task_ids"],
            [self.package.task_id, other_unit.package.task_id],
        )
        self.assertEqual(report.to_dict()["n_completed_tasks"], 2)
        self.assertEqual(report.to_dict()["missing_task_ids"], [])

    def test_report_fails_when_a_materialized_task_has_no_trial(self):
        other_root = self.root / "swe_slugfix_copy"
        shutil.copytree(SWE_TASK, other_root)
        other_unit = make_unit(load_task_package(other_root), "solution")
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )

        report = self._report(prepared)

        self.assertFalse(report.passed)
        self.assertEqual(report.completed_task_ids, (self.package.task_id,))
        self.assertEqual(report.missing_task_ids, (other_unit.package.task_id,))
        self.assertEqual(report.to_dict()["n_completed_tasks"], 1)
        self.assertEqual(
            report.to_dict()["missing_task_ids"],
            [other_unit.package.task_id],
        )

    @patch("ptdbench.cli.HarborEvaluator.evaluate")
    def test_cli_keeps_batch_report_when_only_one_task_has_a_trial(self, evaluate):
        other_root = self.root / "swe_slugfix_copy"
        shutil.copytree(SWE_TASK, other_root)
        other_unit = make_unit(load_task_package(other_root), "solution")
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )
        evaluate.return_value = self._report(prepared)
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            returncode = cli_main(
                ["harbor-evaluate", str(SWE_TASK), str(other_root)]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(returncode, 1)
        self.assertEqual(
            payload["task_ids"],
            [self.package.task_id, other_unit.package.task_id],
        )
        self.assertEqual(payload["missing_task_ids"], [other_unit.package.task_id])

    def test_unknown_task_name_in_a_batch_is_an_error(self):
        other_root = self.root / "swe_slugfix_copy"
        shutil.copytree(SWE_TASK, other_root)
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, make_unit(load_task_package(other_root), "solution")],
        )
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        trial["task_name"] = "ptdbench/unrelated-solution"
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        with self.assertRaisesRegex(HarborAdapterError, "does not match any"):
            self._report(prepared)

    def test_reads_legacy_embedded_trial_results(self):
        (self.trial_dir / "result.json").unlink()
        (self.job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "trial_name": "trial-1",
                            "task_name": self.unit.harbor_task_name,
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

    def test_single_task_trial_without_task_name_still_maps(self):
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        del trial["task_name"]
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        result = self._map()

        self.assertEqual(result.task_id, self.package.task_id)

    def test_unknown_task_name_in_a_single_task_job_is_an_error(self):
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        trial["task_name"] = "ptdbench/unrelated-solution"
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        with self.assertRaisesRegex(HarborAdapterError, "does not match any"):
            self._report()

    def test_failed_harbor_command_without_result_has_a_clear_error(self):
        (self.job_dir / "result.json").unlink()

        with self.assertRaisesRegex(HarborAdapterError, "exit code 2"):
            HarborResultMapper().map(
                self.prepared,
                HarborRun(
                    command=self._command(returncode=2, stderr="invalid option"),
                    job_dir=self.job_dir,
                ),
                HarborEvaluationConfig(),
            )


class HarborCondaTaskTests(unittest.TestCase):
    """迁移后的任务在 Harbor 下没有自己的 Dockerfile，适配器要现生成一个。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _conda_task(self, environment_block: str, *, prepare: str | None = None) -> Path:
        """把 example 任务改成 conda 形态：删掉 Dockerfile，换掉 environment 段。"""
        task = self.root / "conda_task"
        shutil.copytree(SWE_TASK, task)
        (task / "environment" / "Dockerfile").unlink()
        metadata = (task / "metadata.yaml").read_text(encoding="utf-8")
        metadata = metadata.replace(
            "environment:\n  dockerfile: environment/Dockerfile\n",
            environment_block,
        )
        (task / "metadata.yaml").write_text(metadata, encoding="utf-8")
        if prepare is not None:
            (task / "environment" / "prepare.sh").write_text(prepare, encoding="utf-8")
        return task

    def _materialize(self, task: Path, **kwargs) -> Path:
        package = load_task_package(task)
        unit = make_unit(package, "solution")
        prepared = HarborTaskMaterializer().materialize(
            [unit],
            self.root / "out",
            HarborEvaluationConfig(evaluation_level="solution", **kwargs),
        )
        return prepared.task_dir(unit)

    def test_dockerfile_is_generated_from_the_unified_image(self):
        task = self._conda_task("environment:\n  conda: verl\n")
        task_dir = self._materialize(task, image="ptdbench/env:test")
        dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM ptdbench/env:test", dockerfile)
        self.assertIn("COPY --chmod=0777 .ptdbench/workspace/ /workspace/", dockerfile)
        self.assertIn("WORKDIR /workspace", dockerfile)

    def test_legacy_task_keeps_its_own_base_image(self):
        # raw 任务有自己的 Dockerfile，不该被改写成统一镜像。
        package = load_task_package(SWE_TASK)
        unit = make_unit(package, "solution")
        prepared = HarborTaskMaterializer().materialize(
            [unit], self.root / "legacy", HarborEvaluationConfig()
        )
        dockerfile = (
            prepared.task_dir(unit) / "environment" / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertNotIn("FROM ptdbench/env", dockerfile)

    def test_conda_env_reaches_both_stages(self):
        task = self._conda_task("environment:\n  conda: slime-v0.3.1\n")
        task_dir = self._materialize(task)
        toml = (task_dir / "task.toml").read_text(encoding="utf-8")
        agent_block, verifier_block = toml.split("[verifier.env]")

        self.assertIn('PTDBENCH_CONDA_ENV = "slime-v0.3.1"', agent_block)
        self.assertIn('PTDBENCH_CONDA_ENV = "slime-v0.3.1"', verifier_block)

    def test_both_stages_are_told_where_prepare_left_its_output(self):
        # prepare.sh 在构建期就跑完了；两个 stage 只有靠这个变量才找得到它写的东西。
        task = self._conda_task("environment:\n  conda: verl\n")
        toml = (self._materialize(task) / "task.toml").read_text(encoding="utf-8")
        agent_block, verifier_block = toml.split("[verifier.env]")
        for block in (agent_block, verifier_block):
            self.assertIn(f'PTDBENCH_PREPARED = "{PREPARED_DIR}"', block)
            self.assertIn(f'PTDBENCH_ENVIRONMENT = "{PTDBENCH_ENVIRONMENT_DIR}"', block)

    def test_no_workspace_source_variable_reaches_harbor(self):
        # 这个变量已经随 image 模式一起删掉了；再出现说明有地方回退了旧接线。
        task = self._conda_task("environment:\n  conda: verl\n")
        toml = (self._materialize(task) / "task.toml").read_text(encoding="utf-8")
        self.assertNotIn("PTDBENCH_WORKSPACE_SOURCE", toml)

    def test_legacy_task_still_has_no_conda_variable(self):
        package = load_task_package(SWE_TASK)
        unit = make_unit(package, "solution")
        prepared = HarborTaskMaterializer().materialize(
            [unit], self.root / "legacy2", HarborEvaluationConfig()
        )
        toml = (prepared.task_dir(unit) / "task.toml").read_text(encoding="utf-8")
        self.assertNotIn("PTDBENCH_CONDA_ENV", toml)

    def test_prepare_is_baked_as_a_build_layer(self):
        # Harbor 的 task.toml 没有 pre-agent 阶段，构建期是这条路上唯一还有网络、
        # 还能往镜像里写东西的时刻。同一份 prepare.sh，一个字都不用改。
        task = self._conda_task(
            "environment:\n  conda: verl\n",
            prepare="#!/usr/bin/env bash\necho fetching\n",
        )
        task_dir = self._materialize(task)
        dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(f"bash {PTDBENCH_ENVIRONMENT_DIR}/prepare.sh", dockerfile)
        self.assertIn("PTDBENCH_STAGE=prepare", dockerfile)
        self.assertIn("PTDBENCH_CONDA_ENV=verl", dockerfile)
        self.assertEqual(
            (
                task_dir / "environment" / ".ptdbench" / "environment" / "prepare.sh"
            ).read_text(encoding="utf-8"),
            "#!/usr/bin/env bash\necho fetching\n",
        )

    def test_workspace_is_copied_in_before_prepare_runs(self):
        # 反过来的话，COPY workspace 会把 prepare.sh 下好的权重盖掉。
        task = self._conda_task(
            "environment:\n  conda: verl\n",
            prepare="#!/usr/bin/env bash\n",
        )
        dockerfile = (
            self._materialize(task) / "environment" / "Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertLess(
            dockerfile.index("COPY --chmod=0777 .ptdbench/workspace/"),
            dockerfile.index("bash /ptdbench/environment/prepare.sh"),
        )

    def test_the_download_cache_does_not_ship_in_the_image(self):
        # cache 是"为了不重下"，不是产物。Harbor 那边每个任务一个镜像，留着等于
        # 把每个模型存两份，这笔账翻倍得很快。
        task = self._conda_task(
            "environment:\n  conda: verl\n",
            prepare="#!/usr/bin/env bash\n",
        )
        dockerfile = (
            self._materialize(task) / "environment" / "Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertIn(f"rm -rf {MODELS_DIR} {DATASETS_DIR}", dockerfile)

    def test_no_prepare_script_means_no_prepare_layer(self):
        task = self._conda_task("environment:\n  conda: verl\n")
        dockerfile = (
            self._materialize(task) / "environment" / "Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertNotIn("PTDBENCH_STAGE=prepare", dockerfile)

    def test_missing_prepare_file_is_reported_with_its_path(self):
        task = self._conda_task(
            "environment:\n  conda: verl\n  prepare: environment/fetch.sh\n"
        )
        with self.assertRaises(Exception) as ctx:
            self._materialize(task)
        self.assertIn("fetch.sh", str(ctx.exception))

    def test_prepare_outside_the_environment_directory_is_rejected(self):
        # 只有 environment/ 被 COPY 进镜像；声明一个在外面的路径会变成构建期的
        # "文件不存在"，报错点离原因很远。
        task = self._conda_task("environment:\n  conda: verl\n  prepare: fetch.sh\n")
        (task / "fetch.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            self._materialize(task)
        self.assertIn("must stay", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
