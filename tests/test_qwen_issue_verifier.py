from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ISSUE = (
    ROOT
    / "example/qwen_dapo_hparam/task_hparam_seqlen_micro_014/evaluations/issue"
)
JUDGE_PATH = ISSUE / "verifier/judge.py"


def load_judge_module():
    spec = importlib.util.spec_from_file_location("qwen_issue_judge", JUDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {JUDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class QwenIssueVerifierTests(unittest.TestCase):
    def test_issue_package_has_only_the_minimal_files(self):
        files = {
            str(path.relative_to(ISSUE))
            for path in ISSUE.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            files,
            {
                "instruction.md",
                "oracle.md",
                "verifier/test.sh",
                "verifier/judge.py",
                "verifier/judge_prompt.md",
            },
        )
        instruction = (ISSUE / "instruction.md").read_text(encoding="utf-8")
        self.assertIn("$PTDBENCH_RESULTS/agent_answer.md", instruction)
        self.assertNotIn("submission.json", instruction)

    def test_judge_uses_openai_compatible_json_request(self):
        judge = load_judge_module()
        response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict_score": 1,
                                    "root_cause_score": 0.5,
                                    "evidence_score": 0.75,
                                    "reason": "The verdict is right but one cause is missing.",
                                }
                            )
                        }
                    }
                ]
            }
        )
        with patch.object(judge.urllib.request, "urlopen", return_value=response) as call:
            result = judge.call_judge(
                prompt="rubric",
                oracle="canonical diagnosis",
                answer="agent diagnosis",
                model="judge-model",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                timeout_s=10,
            )

        request = call.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(payload["model"], "judge-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("canonical diagnosis", payload["messages"][1]["content"])
        self.assertIn("agent diagnosis", payload["messages"][1]["content"])
        self.assertEqual(result["root_cause_score"], 0.5)

    def test_judge_rejects_an_invalid_score(self):
        judge = load_judge_module()
        with self.assertRaisesRegex(judge.JudgeError, "between 0 and 1"):
            judge.validate_judgment(
                {
                    "verdict_score": 1.0,
                    "root_cause_score": 1.2,
                    "evidence_score": 1.0,
                    "reason": "invalid",
                }
            )

    def test_judge_main_writes_ptdbench_result(self):
        judge = load_judge_module()
        response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict_score": 1,
                                    "root_cause_score": 1,
                                    "evidence_score": 1,
                                    "reason": "The diagnosis agrees with the oracle.",
                                }
                            )
                        }
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            answer = root / "agent_answer.md"
            oracle = root / "oracle.md"
            prompt = root / "judge_prompt.md"
            result = root / "result.json"
            answer.write_text("agent diagnosis", encoding="utf-8")
            oracle.write_text("canonical diagnosis", encoding="utf-8")
            prompt.write_text("rubric", encoding="utf-8")
            argv = [
                "judge.py",
                "--answer",
                str(answer),
                "--oracle",
                str(oracle),
                "--prompt",
                str(prompt),
                "--result",
                str(result),
            ]
            env = {
                "PTDBENCH_JUDGE_MODEL": "judge-model",
                "OPENAI_BASE_URL": "https://api.example.test/v1",
                "OPENAI_API_KEY": "secret-key",
            }
            with patch.object(
                judge.urllib.request, "urlopen", return_value=response
            ):
                with patch.object(sys, "argv", argv), patch.dict(
                    os.environ, env, clear=False
                ):
                    returncode = judge.main()
            payload = json.loads(result.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["metrics"]["status"], "judged")


if __name__ == "__main__":
    unittest.main()
