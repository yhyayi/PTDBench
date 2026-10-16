"""Offline contract and evidence tests; these do NOT validate LLM grading quality."""
import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TASK = Path(__file__).resolve().parents[4]
VERIFIER = TASK / 'evaluations/issue/verifier'


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = module_from(VERIFIER / 'judge.py', 'shared_issue_judge')


class JudgeContract(unittest.TestCase):
    def run_case(self, answer, judgment=None, configured=True):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            if answer is not None:
                (results / 'agent_answer.md').write_text(answer)
            env = {'PTDBENCH_RESULTS': str(results), 'PTDBENCH_VERIFIER': str(VERIFIER),
                   'PTDBENCH_ORACLE': str(TASK / 'evaluations/issue/oracle.md')}
            if configured:
                env.update(PTDBENCH_JUDGE_MODEL='contract-fixture', OPENAI_BASE_URL='https://invalid.example/v1',
                           OPENAI_API_KEY='not-a-real-key')
            with patch.dict(os.environ, env, clear=True), patch('sys.argv', ['judge.py']), \
                    patch.object(judge, 'call_judge', return_value=judgment) as call:
                code = judge.main()
                result = json.loads((results / 'result.json').read_text())
            return code, result, call.call_count

    def test_positive_judgment_serialization(self):
        code, result, calls = self.run_case('fixture answer', {
            'verdict_score': 1.0, 'root_cause_score': 1.0, 'evidence_score': 1.0, 'reason': 'contract fixture'})
        self.assertEqual(code, 0)
        self.assertTrue(result['passed'])
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(calls, 1)

    def test_partial_root_cause_cannot_pass_threshold(self):
        code, result, _ = self.run_case('fixture answer', {
            'verdict_score': 1.0, 'root_cause_score': 0.5, 'evidence_score': 1.0, 'reason': 'partial fixture'})
        self.assertEqual(code, 1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['score'], 0.7)

    def test_empty_and_missing_answers_do_not_call_judge(self):
        for answer in [None, '', '  \n']:
            with self.subTest(answer=answer):
                code, result, calls = self.run_case(answer)
                self.assertEqual(code, 2)
                self.assertFalse(result['passed'])
                self.assertIsNone(result['score'])
                self.assertEqual(calls, 0)

    def test_absent_configuration_is_unavailable_not_pass(self):
        code, result, calls = self.run_case('answer', configured=False)
        self.assertEqual(code, 2)
        self.assertEqual(result['metrics']['status'], 'unavailable')
        self.assertEqual(calls, 0)

    def test_invalid_judgment_values_fail(self):
        for invalid in [-0.1, 1.1, float('nan'), float('inf'), True, '1']:
            with self.subTest(invalid=invalid), self.assertRaises(judge.JudgeError):
                judge.validate_judgment({'verdict_score': invalid, 'root_cause_score': 1.0,
                                         'evidence_score': 1.0, 'reason': 'test'})

    def test_extra_keys_fail(self):
        with self.assertRaises(judge.JudgeError):
            judge.validate_judgment({'verdict_score': 1, 'root_cause_score': 1,
                                     'evidence_score': 1, 'reason': 'ok', 'passed': True})

    def test_json_extraction(self):
        self.assertEqual(judge.extract_json_object('```json\n{"score": 0}\n```'), {'score': 0})


class MathematicalContract(unittest.TestCase):
    def test_candidate_shift_keeps_loss_but_changes_mass(self):
        import torch
        loss_module = module_from(TASK / 'workspace/verl/verl/trainer/opd/order_only.py', 'ranking_objective')
        z = torch.tensor([[8., 6., 4., 2., 0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 1, 2, 3]])
        base = loss_module.direct_topk_loss(z.log_softmax(-1).gather(-1, ids), 'adjacent_logistic')
        shifted = z + torch.tensor([[-30., -30., -30., -30., 0., 0., 0., 0.]], dtype=z.dtype)
        moved = loss_module.direct_topk_loss(shifted.log_softmax(-1).gather(-1, ids), 'adjacent_logistic')
        torch.testing.assert_close(base, moved)
        self.assertGreater(z.softmax(-1)[:, :4].sum().item(), 0.99)
        self.assertLess(shifted.softmax(-1)[:, :4].sum().item(), 1e-8)
        grad = torch.autograd.grad(base.sum(), z)[0]
        torch.testing.assert_close(grad[:, 4:], torch.zeros_like(grad[:, 4:]), atol=1e-14, rtol=0)

    def test_full_vocabulary_shift_does_not_change_mass(self):
        import torch
        z = torch.tensor([[8., 6., 4., 2., 0., 0., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(z.softmax(-1), (z - 30).softmax(-1))


class EvidenceAndPackaging(unittest.TestCase):
    def test_observed_run_trajectories(self):
        with (TASK / 'workspace/logs/metrics.csv').open() as handle:
            rows = {(r['run'], int(r['step'])): r for r in csv.DictReader(handle)}
        self.assertEqual(len(rows), 401)
        for run in ['A', 'B']:
            self.assertGreater(float(rows[run, 1]['actor/opd_student_mass']), 0.9)
            self.assertLess(float(rows[run, 200]['actor/opd_student_mass']), 0.001)
            self.assertGreater(float(rows[run, 200]['actor/entropy']), 11.7)
            self.assertEqual(float(rows[run, 100]['val-core/AIME25/acc/mean@16']), 0.0)

    def test_fixed_prefix_tradeoff(self):
        metrics = json.loads((TASK / 'workspace/evidence/fixed_prefix_metrics.json').read_text())['metrics']
        self.assertLess(metrics['200']['adjacent_loss'], metrics['0']['adjacent_loss'])
        self.assertLess(metrics['200']['teacher_top8_mass'], metrics['0']['teacher_top8_mass'])
        self.assertGreater(metrics['200']['entropy'], metrics['0']['entropy'])

    def test_asset_manifest_is_immutable_and_checked(self):
        manifest = json.loads((TASK / 'environment/assets.json').read_text())
        for asset in manifest['assets']:
            self.assertNotIn(asset['revision'], ['', 'main', 'master', 'latest'])
            self.assertIn(asset['revision'], asset['url'])
            self.assertRegex(asset['sha256'], r'^[a-f0-9]{64}$')
            self.assertGreater(asset['size'], 0)
            self.assertFalse(Path(asset['dest']).is_absolute())
            self.assertNotIn('..', Path(asset['dest']).parts)

    def test_cache_corruption_is_hard_failure(self):
        import hashlib
        fetch = module_from(TASK / 'environment/fetch_assets.py', 'task_fetch')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            data = b'validated artifact'
            asset = {'name': 'fixture', 'dest': 'data.bin', 'destination': 'workspace',
                     'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            (cache / asset['sha256']).write_bytes(b'x' * len(data))
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                fetch.deliver(asset, cache, {'workspace': root / 'workspace'}, offline=True)
            self.assertFalse((root / 'workspace/data.bin').exists())

    def test_valid_cache_is_copied_and_prepare_is_repeatable(self):
        import hashlib
        fetch = module_from(TASK / 'environment/fetch_assets.py', 'task_fetch_repeat')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            data = b'validated artifact'
            asset = {'name': 'fixture', 'dest': 'data.bin', 'destination': 'workspace',
                     'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            (cache / asset['sha256']).write_bytes(data)
            for _ in range(2):
                fetch.deliver(asset, cache, {'workspace': root / 'workspace'}, offline=True)
            destination = root / 'workspace/data.bin'
            self.assertEqual(destination.read_bytes(), data)
            self.assertFalse(destination.is_symlink())
            self.assertNotEqual(destination.stat().st_ino, (cache / asset['sha256']).stat().st_ino)


if __name__ == '__main__':
    unittest.main()
