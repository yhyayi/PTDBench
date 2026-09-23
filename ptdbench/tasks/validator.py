from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    EvaluationPaths,
    TaskPaths,
    ValidationReport,
    known_conda_envs,
)


def validate_task_paths(paths: TaskPaths, strict: bool = True) -> ValidationReport:
    """检查 task 的目录和入口文件。"""
    report = ValidationReport()
    _expect_file(report, paths.metadata_yaml, "metadata.yaml")
    _expect_dir(report, paths.workspace, "workspace")
    _expect_dir(report, paths.environment, "environment")
    if not paths.evaluations:
        report.add("error", "evaluations", "at least one evaluation is required")
    for evaluation in paths.evaluations.values():
        _validate_evaluation_paths(
            report,
            evaluation,
            strict=strict,
            legacy=evaluation.root == paths.root,
        )
    return report


def _validate_evaluation_paths(
    report: ValidationReport,
    evaluation: EvaluationPaths,
    *,
    strict: bool,
    legacy: bool,
) -> None:
    prefix = "" if legacy else f"evaluations/{evaluation.level}/"
    if not legacy:
        _expect_dir(report, evaluation.root, prefix.rstrip("/"))
    _expect_file(report, evaluation.instruction_md, f"{prefix}instruction.md")
    if evaluation.level == "issue":
        _expect_file(report, evaluation.oracle, f"{prefix}oracle.md")
    else:
        _expect_dir(report, evaluation.oracle, f"{prefix}oracle")
    _expect_dir(report, evaluation.verifier, f"{prefix}verifier")
    _expect_file(report, evaluation.verifier_test, f"{prefix}verifier/test.sh")
    if strict:
        _expect_optional_dir(
            report,
            evaluation.verifier_tests,
            f"{prefix}verifier/tests",
        )
        _expect_optional_dir(
            report,
            evaluation.verifier_resources,
            f"{prefix}verifier/resources",
        )


def validate_metadata(
    metadata: Dict[str, Any], paths: Optional[TaskPaths] = None
) -> ValidationReport:
    """检查 metadata 中当前 schema 强制要求的字段。

    `paths` 可选：给了就顺便校验 metadata 里指向任务目录的路径真的存在（现在只有
    `training.script`）。不给也能用——这个函数本身只需要一份 metadata。
    """
    report = ValidationReport()
    required_fields = [
        "title",
        "submitter",
        "license",
        "created_date",
        "domain",
        "model",
        "dataset",
        "training",
        "environment",
        "outcome",
    ]
    for key in required_fields:
        if key not in metadata or metadata[key] in (None, ""):
            report.add("error", f"metadata.{key}", "missing required field")

    for key in ("submitter", "model", "dataset", "training", "environment", "outcome"):
        _expect_mapping(report, metadata.get(key), f"metadata.{key}")

    submitter = metadata.get("submitter") or {}
    if isinstance(submitter, dict):
        _expect_nonempty(report, submitter.get("name"), "metadata.submitter.name")

    model = metadata.get("model") or {}
    if isinstance(model, dict):
        _expect_nonempty(report, model.get("name"), "metadata.model.name")

    dataset = metadata.get("dataset") or {}
    if isinstance(dataset, dict):
        _expect_nonempty(report, dataset.get("name"), "metadata.dataset.name")

    training = metadata.get("training") or {}
    if isinstance(training, dict):
        for key in ("framework", "algo"):
            _expect_nonempty(report, training.get(key), f"metadata.training.{key}")
        _validate_training(report, training, paths)

    environment = metadata.get("environment") or {}
    if isinstance(environment, dict):
        _validate_environment(report, environment)

    outcome = metadata.get("outcome") or {}
    if isinstance(outcome, dict):
        for key in ("failure_type", "failure_reason"):
            _expect_nonempty(report, outcome.get(key), f"metadata.outcome.{key}")
    return report


def dump_report(report: ValidationReport) -> str:
    """把校验结果转为 CLI 可直接打印的 JSON。"""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def _expect_file(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing file")
    elif not path.is_file():
        report.add("error", label, "expected a file")


def _expect_dir(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing directory")
    elif not path.is_dir():
        report.add("error", label, "expected a directory")


def _expect_optional_dir(report: ValidationReport, path, label: str) -> None:
    if path.exists() and not path.is_dir():
        report.add("error", label, "expected a directory")


def _validate_training(
    report: ValidationReport, training: Dict[str, Any], paths: Optional[TaskPaths]
) -> None:
    """校验 training 段里的启动脚本。

    `script` 是"这个任务怎么跑起来"的唯一出处：debug-env.sh 的 `run` 读它，
    instruction.md 里写给 agent 看的也是同一条路径。以前靠 glob 猜（examples 下
    的 *_rl.sh / run_*.sh），rlve 那种把脚本放在 scripts/training/ 下的任务就一个
    都猜不到，所以现在由任务自己声明。

    路径相对 workspace/，不是相对任务根目录：容器里 workspace 被挂到 /workspace，
    训练脚本是在那儿跑的，任务卡上记的也该是同一条路径。

    不是必填项——`framework: none` 的任务（例如纯代码修复）本来就没有训练脚本。
    但凡声明了 framework 又没写 script，就是漏了，给 warning。
    """
    script = training.get("script")
    args = training.get("script_args")

    if script in (None, ""):
        framework = str(training.get("framework") or "").strip().lower()
        if framework not in ("", "none"):
            report.add(
                "warning",
                "metadata.training.script",
                "missing: declare the launch script (relative to workspace/) so "
                "that docker/debug-env.sh's `run` and the task card agree on it",
            )
        if args not in (None, ""):
            report.add(
                "error",
                "metadata.training.script_args",
                "cannot be set without metadata.training.script",
            )
        return

    if not isinstance(script, str):
        report.add(
            "error",
            "metadata.training.script",
            f"expected a path relative to workspace/, got {script!r}",
        )
    elif script.startswith("/") or ".." in Path(script).parts:
        # 绝对路径在别人的机器上没有意义，`..` 则指到 workspace 外面去了——
        # 而容器里只有 workspace 那一份是任务自己的。
        report.add(
            "error",
            "metadata.training.script",
            f"must stay inside workspace/: {script}",
        )
    elif paths is not None and not (paths.workspace / script).is_file():
        # 路径写错和压根没写，症状完全不同：前者要到 `run` 那一刻才炸，而那通常
        # 已经是几个小时之后了。
        report.add(
            "error",
            "metadata.training.script",
            f"workspace/{script} does not exist",
        )

    if args in (None, ""):
        return
    if not isinstance(args, list):
        # 一整行 "a b c" 和三个参数的区别，在带空格的参数上就不一样了；这里只认
        # 列表，免得读者去猜是谁来做分词。
        report.add(
            "error",
            "metadata.training.script_args",
            "expected a list of strings, one argument per item",
        )
        return
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            # YAML 里 `- 0` 是 int，`- "0"` 才是参数本身的样子。
            report.add(
                "error",
                f"metadata.training.script_args[{index}]",
                "expected a string; quote the value in YAML",
            )


def _validate_environment(report: ValidationReport, environment: Dict[str, Any]) -> None:
    """校验 environment 段，两种形态二选一。

    统一镜像的任务给 `conda`（再可选 `requirements` / `prepare` / `env`）；raw 任务
    还没迁移，继续走 `dockerfile` 这条 legacy 路径。这里看不到任务是不是 raw，所以按
    字段分支：有 `conda` 就按新形态校验，没有就按 legacy 要求 `dockerfile`。
    """
    conda = environment.get("conda")
    dockerfile = environment.get("dockerfile")

    if conda in (None, ""):
        # legacy：保持原来的语义——字段缺省时按约定路径算数，只有显式写空才报错。
        _expect_nonempty(
            report,
            environment.get("dockerfile", "environment/Dockerfile"),
            "metadata.environment.dockerfile",
        )
        return

    known = known_conda_envs()
    if not isinstance(conda, str):
        report.add("error", "metadata.environment.conda", f"expected a string, got {conda!r}")
    elif known and conda not in known:
        # 拿不到注册表（比如装成 wheel 之后没有 docker/）时不报错：那种情况下这里
        # 只能瞎猜，真正的判据在镜像里，activate hook 找不到环境会明确说出来。
        report.add(
            "error",
            "metadata.environment.conda",
            f"must be one of {', '.join(sorted(known))}, got {conda!r}; "
            f"a new environment needs a docker/envs/<name>/ directory and a "
            f"matching stage in docker/Dockerfile",
        )

    if dockerfile not in (None, ""):
        # 两个都写了的话，运行时只有一个会生效，而哪个生效不该由读者猜。
        report.add(
            "error",
            "metadata.environment.dockerfile",
            "cannot be combined with environment.conda; drop one",
        )

    for key in ("requirements", "prepare"):
        value = environment.get(key)
        if value not in (None, "") and not isinstance(value, str):
            report.add(
                "error",
                f"metadata.environment.{key}",
                "expected a path relative to the task root",
            )

    env = environment.get("env")
    if env not in (None, ""):
        if not isinstance(env, dict):
            report.add("error", "metadata.environment.env", "expected a mapping")
        else:
            for key, value in env.items():
                if str(key).startswith("PTDBENCH_"):
                    # 这些是 harness 自己的通道；任务改写它们等于改写评测对自己的判断。
                    report.add(
                        "error",
                        f"metadata.environment.env.{key}",
                        "tasks must not set PTDBENCH_* variables",
                    )
                elif not isinstance(value, str):
                    # YAML 里 `CUDA_VISIBLE_DEVICES: 0,1` 是字符串，但 `: 0` 是 int。
                    report.add(
                        "error",
                        f"metadata.environment.env.{key}",
                        "expected a string; quote the value in YAML",
                    )


def _expect_mapping(report: ValidationReport, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        report.add("error", label, "expected a mapping")


def _expect_nonempty(report: ValidationReport, value: Any, label: str) -> None:
    if value in (None, "", [], {}):
        report.add("error", label, "missing required value")
