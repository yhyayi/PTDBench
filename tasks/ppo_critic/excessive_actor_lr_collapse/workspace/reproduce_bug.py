#!/usr/bin/env python3

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "run_buggy.sh"
RECIPE = ROOT / "recipe/vimpo/run_vimpo_critic.sh"
LOG = ROOT / "ptdbench_case/logs/observed_buggy_run.log"


def active_lines(text):
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield raw


def clean_shell_value(value):
    value = value.strip()
    if value.endswith(chr(92)):
        value = value[:-1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()


def extract_launcher_lr(text):
    values = []
    for raw in active_lines(text):
        stripped = raw.strip()
        if stripped.startswith("export LR="):
            values.append(clean_shell_value(stripped.split("=", 1)[1]))
    return values[-1] if values else None


def extract_recipe_lr_spec(text):
    marker = "actor_rollout_ref.actor.optim.lr="
    values = []
    for raw in active_lines(text):
        stripped = raw.strip()
        if marker in stripped:
            rhs = stripped.split(marker, 1)[1]
            values.append(clean_shell_value(rhs))
    if not values:
        raise RuntimeError("actor optimizer LR override not found in VIMPO recipe")
    return values[-1]


def resolve_effective_lr(launcher_lr, recipe_spec):
    prefix = "${LR:-"
    if recipe_spec.startswith(prefix) and recipe_spec.endswith("}"):
        default = recipe_spec[len(prefix):-1]
        raw = launcher_lr if launcher_lr is not None else default
    else:
        raw = recipe_spec
    return float(clean_shell_value(raw))


def collect_step_lines(text):
    steps = {}
    for raw in text.splitlines():
        m = re.search(r"\bstep:(\d+)\s+-", raw)
        if m:
            steps[int(m.group(1))] = raw.replace(chr(92) + ":", ":")
    return steps


def metric(line, key):
    pattern = re.escape(key) + r":(?:np\.float64\()?([-+0-9.eE]+)\)?"
    m = re.search(pattern, line)
    if not m:
        raise RuntimeError(f"metric {key!r} not found in preserved log line")
    return float(m.group(1))


def optional_metric(line, key):
    try:
        return metric(line, key)
    except RuntimeError:
        return None


launcher_text = LAUNCHER.read_text()
recipe_text = RECIPE.read_text()
log_text = LOG.read_text(errors="replace")

launcher_lr = extract_launcher_lr(launcher_text)
recipe_lr_spec = extract_recipe_lr_spec(recipe_text)
effective_lr = resolve_effective_lr(launcher_lr, recipe_lr_spec)

steps = collect_step_lines(log_text)
if 0 not in steps:
    raise RuntimeError("preserved log is missing validation step 0")

completed_step = max(steps)
if completed_step < 8:
    raise RuntimeError(f"expected completed 8-step run, got step {completed_step}")

initial_aime = metric(steps[0], "val-core/math__aime_repeated_8x/acc/mean@1")
initial_math = metric(steps[0], "val-core/math__math/acc/mean@1")
final_aime = metric(steps[8], "val-core/math__aime_repeated_8x/acc/mean@1")
final_math = metric(steps[8], "val-core/math__math/acc/mean@1")

entropy_values = []
clip_values = []
actor_lr_values = []

for step, line in sorted(steps.items()):
    x = optional_metric(line, "actor/old_entropy")
    if x is not None:
        entropy_values.append(x)

    x = optional_metric(line, "response_length/clip_ratio")
    if x is not None:
        clip_values.append((step, x))

    x = optional_metric(line, "actor/lr")
    if x is not None:
        actor_lr_values.append((step, x))

if not entropy_values:
    raise RuntimeError("no actor entropy observations found")

entropy_min = min(entropy_values)
entropy_max = max(entropy_values)
max_length_saturation = any(
    step >= 4 and ratio >= 0.999
    for step, ratio in clip_values
)

bug_present = all(
    [
        effective_lr >= 1e-3,
        completed_step >= 8,
        initial_aime > 0.0,
        initial_math > 0.0,
        final_aime == 0.0,
        final_math == 0.0,
        max_length_saturation,
        entropy_max > 5.0,
        entropy_min < 0.01,
    ]
)

report = {
    "launcher_lr": launcher_lr,
    "recipe_lr_spec": recipe_lr_spec,
    "effective_target_lr": effective_lr,
    "completed_rl_steps": completed_step,
    "initial_validation": {"aime": initial_aime, "math": initial_math},
    "final_validation": {"aime": final_aime, "math": final_math},
    "entropy_min": entropy_min,
    "entropy_max": entropy_max,
    "max_length_saturation_observed": max_length_saturation,
    "actor_lr_schedule_observed": actor_lr_values,
    "bug_present": bug_present,
}

print(json.dumps(report, indent=2, sort_keys=True))

if not bug_present:
    raise SystemExit(
        "BUG_NOT_REPRODUCED: launcher/log evidence does not match "
        "the controlled actor-LR collapse"
    )

print("PASS_BUG")
