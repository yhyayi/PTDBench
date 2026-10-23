"""solution 层 rerun 协议的机器可读部分：run_spec.json 的加载与展开。

协议见 ptdbench/evaluation/run_spec.py 的模块注释。这里盯两件事：

- load_run_spec 的校验：spec 写坏了要在加载时就炸，报 RunSpecError 指到
  具体字段，而不是训练跑了一半才发现；
- expand_env 的展开：$PTDBENCH_* 引用换成当前后端的真实路径，引用了
  不存在的变量时原样保留。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ptdbench.evaluation.run_spec import (
    REPORT_FILENAME,
    RUN_SPEC_FILENAME,
    RunSpec,
    RunSpecError,
    load_run_spec,
)


def write_spec(root: Path, payload) -> Path:
    path = root / "verifier" / "resources" / RUN_SPEC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


VALID = {
    "schema_version": 1,
    "command": 'bash run.sh > "$PTDBENCH_RESULTS/experiment_logs/train.log"',
    "env": {"LOG_DIR": "$PTDBENCH_RESULTS/experiment_logs", "STEPS": "20"},
    "timeout_s": 3600,
    "log_roots": ["experiment_logs"],
}


class LoadRunSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_missing_spec_means_no_rerun(self) -> None:
        # 没有 run_spec.json 是合法状态：critic/aime25 这类任务不启用 rerun。
        self.assertIsNone(load_run_spec(self.root, "task_001"))

    def test_valid_spec_round_trips(self) -> None:
        write_spec(self.root, VALID)
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.task_id, "task_001")
        self.assertEqual(spec.command, VALID["command"])
        self.assertEqual(spec.env, VALID["env"])
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.log_roots, ["experiment_logs"])

    def test_env_and_log_roots_have_defaults(self) -> None:
        write_spec(self.root, {"schema_version": 1, "command": "bash run.sh"})
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.env, {})
        self.assertEqual(spec.log_roots, ["experiment_logs"])
        self.assertIsNone(spec.timeout_s)

    def test_env_scalars_are_stringified(self) -> None:
        # YAML/JSON 里数字写裸值很常见；spec 侧统一成 str。
        write_spec(
            self.root,
            {"schema_version": 1, "command": "bash run.sh", "env": {"STEPS": 20}},
        )
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.env, {"STEPS": "20"})

    def test_bad_payload_raises_run_spec_error(self) -> None:
        cases = {
            "bad schema_version": {"schema_version": 2, "command": "bash run.sh"},
            "empty command": {"schema_version": 1, "command": "  "},
            "command not a string": {"schema_version": 1, "command": ["bash"]},
            "env not an object": {"schema_version": 1, "command": "x", "env": []},
            "env value not a scalar": {
                "schema_version": 1,
                "command": "x",
                "env": {"STEPS": {"n": 1}},
            },
            "env key empty": {"schema_version": 1, "command": "x", "env": {"": "1"}},
            "env declares a harness variable": {
                "schema_version": 1,
                "command": "x",
                "env": {"PTDBENCH_RESULTS": "/results"},
            },
            "log_roots not a list": {
                "schema_version": 1,
                "command": "x",
                "log_roots": "experiment_logs",
            },
            "log_roots has an empty entry": {
                "schema_version": 1,
                "command": "x",
                "log_roots": ["experiment_logs", ""],
            },
            "timeout_s not positive": {
                "schema_version": 1,
                "command": "x",
                "timeout_s": 0,
            },
            "timeout_s is a bool": {
                "schema_version": 1,
                "command": "x",
                "timeout_s": True,
            },
            "top level not an object": ["not", "an", "object"],
        }
        for name, payload in cases.items():
            with self.subTest(name):
                write_spec(self.root, payload)
                with self.assertRaises(RunSpecError):
                    load_run_spec(self.root, "task_001")

    def test_broken_json_raises_run_spec_error(self) -> None:
        path = write_spec(self.root, VALID)
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(RunSpecError):
            load_run_spec(self.root, "task_001")

    def test_report_filename_is_the_protocol_contract(self) -> None:
        # instruction.md 和 evaluator 都指向同一份约定，写死在这里防漂移。
        self.assertEqual(REPORT_FILENAME, "report.md")


class ExpandEnvTests(unittest.TestCase):
    def _spec(self, env) -> RunSpec:
        return RunSpec(
            task_id="task_001",
            command="bash run.sh",
            env=env,
            timeout_s=None,
            log_roots=["experiment_logs"],
        )

    def test_harness_references_are_replaced(self) -> None:
        spec = self._spec({"LOG_DIR": "$PTDBENCH_RESULTS/experiment_logs"})
        self.assertEqual(
            spec.expand_env({"PTDBENCH_RESULTS": "/results"}),
            {"LOG_DIR": "/results/experiment_logs"},
        )

    def test_plain_values_pass_through_untouched(self) -> None:
        spec = self._spec({"STEPS": "20", "MODE": "offline"})
        self.assertEqual(
            spec.expand_env({"PTDBENCH_RESULTS": "/results"}),
            {"STEPS": "20", "MODE": "offline"},
        )

    def test_unresolvable_reference_is_left_alone(self) -> None:
        # 一个小笔误不该把整个评测炸成 orchestration failure。
        spec = self._spec({"A": "$PTDBENCH_MISSING/x", "B": "$OTHER/x"})
        self.assertEqual(
            spec.expand_env({"PTDBENCH_RESULTS": "/results"}),
            {"A": "$PTDBENCH_MISSING/x", "B": "$OTHER/x"},
        )

    def test_empty_resolved_value_keeps_the_reference(self) -> None:
        spec = self._spec({"LOG_DIR": "$PTDBENCH_RESULTS/experiment_logs"})
        self.assertEqual(
            spec.expand_env({"PTDBENCH_RESULTS": ""}),
            {"LOG_DIR": "$PTDBENCH_RESULTS/experiment_logs"},
        )


if __name__ == "__main__":
    unittest.main()
