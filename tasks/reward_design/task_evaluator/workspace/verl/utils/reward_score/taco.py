from __future__ import annotations

import faulthandler
import json
import multiprocessing
import os
import platform
import re
import signal
import sys
import types
from enum import Enum
from io import StringIO
from unittest.mock import mock_open, patch

import numpy as np

try:
    from pyext import RuntimeModule
except ImportError:
    RuntimeModule = None

class _FallbackRuntimeModule:
    @staticmethod
    def from_string(name, _doc, code):
        module = types.ModuleType(name)
        exec(compile(code, name, "exec"), module.__dict__)
        return module

def _runtime_module():
    return RuntimeModule if RuntimeModule is not None else _FallbackRuntimeModule

TIMEOUT = 10

def post_process(text):
    if "```" in text:
        blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split("```")[1]
        else:
            text = blocks[0]
            if not text.startswith("\n"):
                text = text[max(text.find("\n") + 1, 0):]
    return text

def _parse_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        return ground_truth
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    return json.loads(ground_truth)

def _to_bool_results(raw):
    fixed = []
    for e in raw:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return fixed

def _run_all_tests(solution_str, ground_truth, extra_info=None):

    code = post_process(solution_str)
    in_outs = _parse_ground_truth(ground_truth)
    sample = {"input_output": json.dumps(in_outs)}
    n_cases = len(in_outs.get("inputs") or [])
    timeout = min(max(TIMEOUT, 4 * max(n_cases, 1) + 1), 120)
    raw = check_correctness(sample, code, timeout=timeout)
    return _to_bool_results(raw)

def _summarize(results):
    n = len(results) if results else 0
    if n == 0:
        return 0.0, 0.0
    n_pass = sum(1 for r in results if r is True)
    pass_rate = n_pass / n
    binary = 1.0 if n_pass == n else 0.0
    return float(pass_rate), float(binary)

def compute_score_pass_rate(solution_str, ground_truth, extra_info=None, **kwargs):

    results = _run_all_tests(solution_str, ground_truth, extra_info)
    pass_rate, _ = _summarize(results)
    return pass_rate

def compute_score_binary(solution_str, ground_truth, extra_info=None, **kwargs):

    results = _run_all_tests(solution_str, ground_truth, extra_info)
    _, binary = _summarize(results)
    return binary

def _is_validate(extra_info):
    if not isinstance(extra_info, dict):
        return False
    if extra_info.get("validate"):
        return True
    return extra_info.get("split") == "test"

def compute_score(solution_str=None, ground_truth=None, extra_info=None, data_source=None, **kwargs):

    if solution_str is None:
        solution_str = kwargs.get("completion") or kwargs.get("generation") or ""
    results = _run_all_tests(solution_str, ground_truth, extra_info)
    pass_rate, binary = _summarize(results)
    score = binary
    return {
        "score": float(score),
        "acc": float(binary),
        "pass_rate": float(pass_rate),
    }

def _temp_run(sample, generation, result):

    try:
        results = run_test(sample, test=generation)

        result.append(list(results))
    except Exception:
        result.append([-1])
    finally:
        os._exit(0)

def check_correctness(sample, generation, timeout):

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        n = len(in_outs.get("inputs") or [])
        return [-1 for _ in range(n)]
    return result[0]

class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def _install_sigalrm():

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
    except (AttributeError, ValueError):
        pass

timeout = 4

class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()

        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def run_test(sample, test=None):

    reliability_guard()
    _install_sigalrm()

    sys.stdout = StringIO()

    try:
        in_outs = json.loads(sample['input_output'])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get('fn_name') is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs['fn_name']

    if test is None:
        return in_outs
    elif test is not None:
        results = []
        sol = 'import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n'

        if which_type == CODE_TYPE.call_based:
            sol += test
            signal.alarm(timeout)
            try:
                tmp_sol = _runtime_module().from_string('tmp_sol', '', sol)
                if 'class Solution' not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
                results.append(-2)
                return results
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:

            tmp_test = test.split('\n')

            new_test = []
            for x in tmp_test:
                if (not x.startswith('from ')) and (
                        not x.startswith('import ')):
                    new_test.append('\t' + x + '\n')
                else:
                    new_test.append(x + '\n')
            tmp_test = new_test

            new_test = ''
            started = False
            for i in tmp_test:
                if i.startswith('\t') and not started:
                    new_test += 'stdin = sys.stdin\nstdout = sys.stdout\n'
                    new_test += 'def code():\n'
                    new_test += i
                    started = True
                elif started and ((i.startswith('from ')) or
                                  (i.startswith('import '))):
                    new_test += '\t' + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            method_name = 'code'
            signal.alarm(timeout)
            try:
                tmp_sol = _runtime_module().from_string('tmp_sol', '', sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
                results.append(-2)
                return results
            signal.alarm(0)

        try:
            method = getattr(tmp,
                             method_name)
        except:
            signal.alarm(0)
            results.append(-2)
            return results

        for index, inputs in enumerate(in_outs['inputs']):

            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs['outputs'][index], dict):
                    in_outs['outputs'][index] = {
                        int(k): v
                        for k, v in in_outs['outputs'][index].items()
                    }
            except:
                True

            if which_type == CODE_TYPE.call_based:
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)
                    tmp_result = output == in_outs['outputs'][index]
                    results.append(tmp_result)

                    signal.alarm(0)
                except Exception:
                    signal.alarm(0)
                    faulthandler.disable()
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
            elif which_type == CODE_TYPE.standard_input:
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = '\n'.join(inputs)

                with Capturing() as output:
                    try:
                        call_method(method, inputs)

                        signal.alarm(0)
                        passed = True
                    except Exception:

                        signal.alarm(0)
                        results.append(-1)
                    signal.alarm(0)

                if not passed:
                    continue

                expected = in_outs['outputs'][index]
                if isinstance(expected, list):
                    tmp_result = list(output) == expected
                else:
                    tmp_result = '\n'.join(output) == expected
                results.append(tmp_result)

    return results

def custom_compare_(output, ground_truth):

    if isinstance(output, list):
        output_1 = '\n'.join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = '\n'.join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2

def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)

    inputs_line_iterator = iter(inputs.split('\n'))

    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split('\n'))
    @patch('sys.stdin.read', lambda *args: inputs)

    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)

def reliability_guard(maximum_memory_bytes=None):

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS,
                           (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA,
                           (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK,
                               (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None

    __builtins__['help'] = None

    import sys

    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
