from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.types import (
    EvaluationPaths,
    TaskPaths,
    ValidationReport,
    known_images,
)


def validate_task_paths(paths: TaskPaths, strict: bool = True) -> ValidationReport:
    """检查 task 的目录和入口文件。"""
    report = ValidationReport()
    _expect_file(report, paths.metadata_yaml, "metadata.yaml")
    _expect_dir(report, paths.workspace, "workspace")
    # environment/ 里只剩可选内容（prepare.sh 等），镜像本身由 metadata 的
    # environment.image 指向 docker/images/，所以目录可以不存在，但存在就得是目录。
    if paths.environment.exists() and not paths.environment.is_dir():
        report.add("error", "environment", "expected a directory")
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
        "type",
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

    _validate_type(report, metadata.get("type"), paths)

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

    `script` 是"这个任务怎么跑起来"的唯一出处：任务卡上记的、instruction.md
    里写给 agent 看的都是同一条路径。以前靠 glob 猜（examples 下
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
                "that the task card records how the task is run",
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


#: The canonical task categories. `type` is one of these and names the
#: directory the task lives under (tasks/<type>/<id>); `domain` stays a
#: free-form descriptive label (e.g. `post_train`, `math_tool_use`).
TASK_TYPES = frozenset(
    {
        "configuration",
        "reward",
        "data",
        "environment",
        "algorithm",
        "infrastructure",
    }
)


def _validate_type(
    report: ValidationReport, task_type: Any, paths: Optional[TaskPaths]
) -> None:
    """校验 `type`：必须是六个规范类目之一，且与任务所在目录一致。

    `type` 回答"这是哪一类任务"，和目录 tasks/<type>/<id> 是同一个答案，所以能机器
    校验；`domain` 是自由描述（post_train / math_tool_use…），不受这六个约束。缺失由
    required_fields 那轮报 "missing required field"，这里只校验取值。
    """
    if task_type in (None, ""):
        return
    if not isinstance(task_type, str):
        report.add("error", "metadata.type", f"expected a string, got {task_type!r}")
        return
    if task_type not in TASK_TYPES:
        report.add(
            "error",
            "metadata.type",
            f"must be one of {', '.join(sorted(TASK_TYPES))}, got {task_type!r}",
        )
        return
    # 和目录对不上，任务卡记的类目就和它实际所在的位置矛盾了。拿不到 paths（只校验
    # 一份孤立 metadata）时跳过这一条。
    if paths is not None and paths.root.parent.name != task_type:
        report.add(
            "error",
            "metadata.type",
            f"must match the task's category directory "
            f"({paths.root.parent.name!r}), got {task_type!r}",
        )


def _validate_environment(report: ValidationReport, environment: Dict[str, Any]) -> None:
    """校验 environment 段：`image` 必填，`prepare` / `env` 可选。

    `image` 是 docker/images/ 下的环境名，镜像照 docker/images/<image>.Dockerfile
    构建。旧 schema 的 `conda` / `dockerfile` / `requirements` 直接报错而不是忽略：
    忽略的话任务作者会以为它们还在起作用。
    """
    for legacy in ("conda", "dockerfile", "requirements"):
        if legacy in environment:
            report.add(
                "error",
                f"metadata.environment.{legacy}",
                "no longer supported; name a docker/images/<name>.Dockerfile "
                "with environment.image instead",
            )

    image = environment.get("image")
    known = known_images()
    if image in (None, ""):
        report.add("error", "metadata.environment.image", "missing required value")
    elif not isinstance(image, str):
        report.add("error", "metadata.environment.image", f"expected a string, got {image!r}")
    elif known and image not in known:
        # 拿不到 docker/images/（比如装成 wheel 之后）时不报错：那种情况下这里只能
        # 瞎猜，真正的判据是构建镜像时找不找得到 Dockerfile。
        report.add(
            "error",
            "metadata.environment.image",
            f"must be one of {', '.join(sorted(known))}, got {image!r}; "
            f"a new environment needs a docker/images/<name>.Dockerfile",
        )

    prepare = environment.get("prepare")
    if prepare not in (None, "") and not isinstance(prepare, str):
        report.add(
            "error",
            "metadata.environment.prepare",
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
