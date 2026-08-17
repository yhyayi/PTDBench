#!/usr/bin/env python3
"""Self-contained grading primitives aligned with evaluation/script/grade_cases.py."""

from __future__ import annotations

import json
import os
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


STAGE_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
DIRECTION_WORDS = {
    "increase": ["increase", "raise", "larger", "bigger", "higher", "up"],
    "decrease": ["decrease", "lower", "reduce", "smaller", "down", "disable", "set to 0"],
    "set": ["set", "change to", "set to"],
    "remove": ["remove", "delete", "drop", "unset"],
    "restore": ["restore", "revert", "back to"],
    "rewrite_reward": ["rewrite", "redesign", "partial credit", "scorer"],
}
RUBRIC_DIRECTIONS = {"rewrite_reward"}
ROOT_CAUSE_KEYS = {"name", "where", "observed", "expected"}
FIX_KEYS = {"target", "change"}


class GradingError(RuntimeError):
    """A malformed or unsafe verifier input."""


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise GradingError(f"{label} must not be a symlink")
    if not path.is_file():
        raise GradingError(f"missing {label}: {path}")
    return path


def load_submission(path: Path) -> Dict[str, Any]:
    regular_file(path, "submission.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise GradingError("submission.json must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise GradingError(f"submission.json is invalid JSON: {exc}") from exc
    validate_submission(value)
    return value


def validate_submission(value: Any) -> None:
    if not isinstance(value, dict):
        raise GradingError("submission must be a JSON object")
    expected = {"verdict", "root_cause", "evidence", "fix"}
    if set(value) != expected:
        raise GradingError(
            "submission keys must be exactly verdict, root_cause, evidence, fix"
        )
    if value["verdict"] not in {"healthy", "unhealthy"}:
        raise GradingError("verdict must be healthy or unhealthy")
    if not isinstance(value["root_cause"], list):
        raise GradingError("root_cause must be an array")
    for index, item in enumerate(value["root_cause"]):
        if not isinstance(item, dict):
            raise GradingError(f"root_cause[{index}] must be an object")
        if "name" not in item or set(item) - ROOT_CAUSE_KEYS:
            raise GradingError(
                f"root_cause[{index}] requires name and permits only schema fields"
            )
        if any(not isinstance(field, str) for field in item.values()):
            raise GradingError(f"root_cause[{index}] fields must be strings")
    if not isinstance(value["evidence"], list) or any(
        not isinstance(item, str) for item in value["evidence"]
    ):
        raise GradingError("evidence must be an array of strings")
    if not isinstance(value["fix"], list):
        raise GradingError("fix must be an array")
    for index, item in enumerate(value["fix"]):
        if not isinstance(item, dict) or set(item) != FIX_KEYS:
            raise GradingError(
                f"fix[{index}] must contain exactly target and change"
            )
        if any(not isinstance(field, str) for field in item.values()):
            raise GradingError(f"fix[{index}] fields must be strings")


def load_gold(path: Path) -> Dict[str, Any]:
    regular_file(path, "gold.yaml")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise GradingError(f"gold.yaml could not be parsed: {exc}") from exc
    if not isinstance(value, dict):
        raise GradingError("gold.yaml must contain a mapping")
    return value


def norm_name(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^--", "", text)
    text = text.split("::")[-1] if "::" in text else text
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def norm_full(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^--", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def name_variants(name: str) -> set[str]:
    base = norm_name(name)
    out = {base, norm_full(name)}
    parts = base.split("_")
    for prefix in (
        "data",
        "actor_rollout_ref",
        "rollout",
        "algorithm",
        "trainer",
        "verl",
    ):
        if parts and parts[0] == prefix and len(parts) > 1:
            out.add("_".join(parts[1:]))
    head = name.split("::")[0]
    if "::" in name:
        out.add(norm_name(name.split("::")[-1]))
        out.add(norm_full(head))
    if "/" in head or head.endswith(".py"):
        segments = [segment for segment in head.split("/") if segment]
        if segments:
            out.add(norm_full(segments[-1]))
        if len(segments) >= 2:
            out.add(norm_full(f"{segments[-2]}/{segments[-1]}"))
    return {item for item in out if item}


def name_matches(submission_name: str, gold_variants: set[str]) -> bool:
    submission_forms = name_variants(submission_name)
    haystack = norm_full(submission_name)
    for gold in gold_variants:
        if not gold or len(gold) < 4:
            continue
        if gold in submission_forms:
            return True
        if re.search(rf"(?<![a-z0-9]){re.escape(gold)}(?![a-z0-9])", haystack):
            return True
    return False


def submission_names(submission: Dict[str, Any]) -> list[str]:
    return [
        (item or {}).get("name") or ""
        for item in (submission.get("root_cause") or [])
        if isinstance(item, dict)
    ]


def text_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def rule_issue(gold: Dict[str, Any], submission: Dict[str, Any]) -> Dict[str, Any]:
    gold_verdict = gold.get("verdict")
    submission_verdict = (submission.get("verdict") or "").strip().lower()
    result: Dict[str, Any] = {
        "verdict_match": submission_verdict == gold_verdict,
        "decided_by": "rule",
        "matched_gold_names": [],
    }
    if submission_verdict not in ("healthy", "unhealthy"):
        result.update(
            {"root_cause_verdict": "wrong", "why": "missing or invalid verdict"}
        )
        return result
    if not result["verdict_match"]:
        result.update(
            {
                "root_cause_verdict": "wrong",
                "why": (
                    f"verdict mismatch: gold={gold_verdict} "
                    f"got={submission_verdict}"
                ),
            }
        )
        return result
    if gold_verdict == "healthy":
        result.update(
            {"root_cause_verdict": "correct", "why": "healthy verdict matches"}
        )
        return result

    names = submission_names(submission)
    if not names:
        result.update(
            {
                "root_cause_verdict": "wrong",
                "why": "unhealthy but empty root_cause",
            }
        )
        return result

    variants_by_name: Dict[str, set[str]] = {}
    for item in gold.get("root_cause") or []:
        gold_name = (item or {}).get("name") or ""
        variants_by_name[gold_name] = name_variants(gold_name)

    hits = []
    for gold_name, variants in variants_by_name.items():
        for index, name in enumerate(names):
            if name_matches(name, variants):
                hits.append((gold_name, index))
                break
    if hits:
        result["matched_gold_names"] = [name for name, _ in hits]
        first_hit_index = min(index for _, index in hits)
        verdict = "correct" if first_hit_index == 0 else "partial"
        result.update(
            {
                "root_cause_verdict": verdict,
                "why": (
                    f"matched gold {result['matched_gold_names']} "
                    f"(first hit at {first_hit_index})"
                ),
            }
        )
        return result

    result.update(
        {
            "root_cause_verdict": "needs_judge",
            "why": "no rule match; defer to judge",
        }
    )
    return result


def rule_solution(
    gold: Dict[str, Any], submission: Dict[str, Any]
) -> Dict[str, Any]:
    must = ((gold.get("fix") or {}).get("must")) or []
    submitted_fix = submission.get("fix") or []
    result: Dict[str, Any] = {
        "decided_by": "rule",
        "covered_must": [],
        "missing_must": [],
    }
    if not submitted_fix:
        result.update({"fix_verdict": "wrong", "why": "empty fix"})
        return result
    if not must:
        result.update(
            {
                "fix_verdict": "needs_judge",
                "why": "no fix.must; defer to judge/rubric",
            }
        )
        return result
    if any(
        ((item or {}).get("direction") or "") in RUBRIC_DIRECTIONS
        for item in must
    ) or gold.get("rubric"):
        result.update(
            {
                "fix_verdict": "needs_judge",
                "why": "qualitative fix / rubric present",
            }
        )
        return result

    blob = text_blob(submitted_fix)
    for item in must:
        target = (item or {}).get("target") or ""
        direction = ((item or {}).get("direction") or "").strip()
        variants = name_variants(target)
        target_hit = False
        for submitted_item in submitted_fix:
            haystack = norm_name(
                f"{(submitted_item or {}).get('target', '')} "
                f"{(submitted_item or {}).get('change', '')}"
            )
            if any(
                variant
                and re.search(
                    rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])",
                    haystack,
                )
                for variant in variants
            ):
                target_hit = True
                break
        if not target_hit:
            result["missing_must"].append(target)
            continue
        words = list(DIRECTION_WORDS.get(direction, []))
        target_value = str((item or {}).get("to") or "").strip().lower()
        if target_value and re.search(
            rf"(?<![0-9a-z.]){re.escape(target_value)}(?![0-9a-z.])",
            blob,
        ):
            words += DIRECTION_WORDS["set"]
        if words and not any(word in blob for word in words):
            result.update(
                {
                    "fix_verdict": "needs_judge",
                    "why": (
                        f"matched target {target} but direction unclear "
                        f"(want {direction})"
                    ),
                }
            )
            return result
        result["covered_must"].append(target)

    if result["covered_must"] and not result["missing_must"]:
        result.update(
            {
                "fix_verdict": "correct",
                "why": f"covered {result['covered_must']}",
            }
        )
    elif result["covered_must"]:
        result.update(
            {
                "fix_verdict": "partial",
                "why": (
                    f"covered {result['covered_must']}; "
                    f"missing {result['missing_must']}"
                ),
            }
        )
    else:
        result.update(
            {"fix_verdict": "needs_judge", "why": "no must target matched"}
        )
    return result


def _endpoint(provider: str, base_url: str, model: str, api_key: str) -> str:
    base = base_url.rstrip("/")
    if provider == "openai-compatible":
        return base if base.endswith("/chat/completions") else base + "/chat/completions"
    if provider == "anthropic":
        return base if base.endswith("/v1/messages") else base + "/v1/messages"
    if "{model}" in base:
        base = base.replace("{model}", urllib.parse.quote(model, safe=""))
    elif ":generateContent" not in base:
        base += f"/models/{urllib.parse.quote(model, safe='')}:generateContent"
    delimiter = "&" if "?" in base else "?"
    return base + delimiter + "key=" + urllib.parse.quote(api_key, safe="")


def _request_payload(provider: str, model: str, prompt: str) -> Dict[str, Any]:
    if provider == "openai-compatible":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    if provider == "anthropic":
        return {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }


def _response_text(provider: str, payload: Dict[str, Any]) -> str:
    if provider == "openai-compatible":
        return str(payload["choices"][0]["message"]["content"])
    if provider == "anthropic":
        return "".join(
            str(item.get("text") or "")
            for item in payload.get("content") or []
            if isinstance(item, dict)
        )
    candidates = payload.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    return "".join(
        str(item.get("text") or "") for item in parts if isinstance(item, dict)
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise GradingError("judge response did not contain a JSON object")


def call_judge(
    kind: str,
    payload: Dict[str, Any],
    prompt_path: Path,
    *,
    no_judge: bool = False,
) -> Dict[str, Any]:
    prompt = regular_file(prompt_path, f"judge_{kind}.md").read_text(
        encoding="utf-8"
    )
    prompt = prompt.replace(
        "{GOLD_KEY}", json.dumps(payload["gold"], ensure_ascii=False)
    )
    if kind == "issue":
        prompt = prompt.replace(
            "{SUBMISSION}",
            json.dumps(payload["submission"], ensure_ascii=False),
        )
    else:
        prompt = prompt.replace(
            "{SUBMISSION_FIX}",
            json.dumps(payload.get("submission_fix"), ensure_ascii=False),
        ).replace(
            "{MATCHED_GOLD_NAMES}",
            json.dumps(payload.get("confirmed"), ensure_ascii=False),
        )
    if no_judge or os.environ.get("PTDBENCH_NO_JUDGE") == "1":
        return {"_judge": "skipped"}

    provider = os.environ.get("PTDBENCH_JUDGE_PROVIDER", "openai-compatible")
    model = os.environ.get("PTDBENCH_JUDGE_MODEL", "").strip()
    base_url = os.environ.get("PTDBENCH_JUDGE_BASE_URL", "").strip()
    key_name = os.environ.get("PTDBENCH_JUDGE_API_KEY_ENV", "").strip()
    if provider not in {"openai-compatible", "anthropic", "gemini"}:
        return {"_judge": "error", "error": f"unsupported provider: {provider}"}
    if not model or not base_url or not key_name:
        return {"_judge": "error", "error": "judge configuration is incomplete"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        return {"_judge": "error", "error": "invalid judge API key variable name"}
    api_key = os.environ.get(key_name, "")
    if not api_key:
        return {"_judge": "error", "error": f"missing environment variable {key_name}"}

    headers = {"Content-Type": "application/json"}
    if provider == "openai-compatible":
        headers["Authorization"] = "Bearer " + api_key
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    request = urllib.request.Request(
        _endpoint(provider, base_url, model, api_key),
        data=json.dumps(_request_payload(provider, model, prompt)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout_raw = os.environ.get("PTDBENCH_JUDGE_TIMEOUT", "0").strip() or "0"
    try:
        timeout_value = int(timeout_raw)
        timeout: Optional[float] = None if timeout_value <= 0 else float(timeout_value)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        judged = _extract_json_object(_response_text(provider, response_payload))
        judged["_judge"] = "ok"
        judged["_judge_model"] = model
        return judged
    except (
        ValueError,
        KeyError,
        IndexError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        urllib.error.URLError,
        GradingError,
    ) as exc:
        return {"_judge": "error", "error": str(exc)}


def grade_issue(
    gold: Dict[str, Any],
    submission: Dict[str, Any],
    judge_issue_path: Path,
    *,
    no_judge: bool = False,
) -> Dict[str, Any]:
    result = rule_issue(gold, submission)
    if result["root_cause_verdict"] == "needs_judge":
        judged = call_judge(
            "issue",
            {"gold": gold, "submission": submission},
            judge_issue_path,
            no_judge=no_judge,
        )
        if judged.get("_judge") == "ok":
            result.update(
                {
                    "root_cause_verdict": judged.get(
                        "root_cause_verdict", "wrong"
                    ),
                    "decided_by": "judge",
                    "matched_gold_names": (
                        judged.get("matched_gold_names")
                        or result.get("matched_gold_names")
                    ),
                    "judge": judged,
                }
            )
        else:
            result.update(
                {
                    "root_cause_verdict": "unavailable",
                    "decided_by": "judge_failed",
                    "judge": judged,
                }
            )
    return result


def grade_solution(
    gold: Dict[str, Any],
    submission: Dict[str, Any],
    issue_result: Dict[str, Any],
    judge_solution_path: Path,
    *,
    no_judge: bool = False,
) -> Dict[str, Any]:
    result = rule_solution(gold, submission)
    if result["fix_verdict"] == "needs_judge":
        judged = call_judge(
            "solution",
            {
                "gold": gold,
                "submission_fix": submission.get("fix"),
                "confirmed": issue_result.get("matched_gold_names"),
            },
            judge_solution_path,
            no_judge=no_judge,
        )
        if judged.get("_judge") == "ok":
            result.update(
                {
                    "fix_verdict": judged.get("fix_verdict", "wrong"),
                    "decided_by": "judge",
                    "covered_must": (
                        judged.get("covered_must") or result.get("covered_must")
                    ),
                    "missing_must": (
                        judged.get("missing_must") or result.get("missing_must")
                    ),
                    "judge": judged,
                }
            )
        else:
            result.update(
                {
                    "fix_verdict": "unavailable",
                    "decided_by": "judge_failed",
                    "judge": judged,
                }
            )
    return result
