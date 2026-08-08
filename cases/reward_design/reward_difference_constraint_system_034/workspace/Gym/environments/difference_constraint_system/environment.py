# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Any
from Gym.environment import VerifiableEnvironment


class DifferenceConstraintSystem_Environment(VerifiableEnvironment):
    prompt_template = (
        'There are {N} integers x[0], x[1], ..., x[{N_minus_1}]. '
        'They satisfy the following {M} inequations:\n'
        '{inequations}\n\n'
        'Please find any solution x[0], x[1], ..., x[{N_minus_1}] '
        'that satisfies the inequations.\n\n'
        'Output Format: Your final answer should be a single line '
        'containing x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._constraints: List[tuple] = []
        # Each constraint is (i, j, c, op) where op is '<=' or '>='
        # Semantics: x[i] - x[j] <= c   OR   x[i] - x[j] >= c
        self._ground_truth: List[int] = []

    def _generate(self) -> None:
        """Generate a solvable difference constraint system.

        Uses self.parameter dict keys: N (int), M (int).
        """
        N = self.parameter.get('N', 3)
        M = self.parameter.get('M', max(N, 5))

        # Ground-truth solution in a moderate range
        gt = [random.randint(-50, 50) for _ in range(N)]

        constraints = []
        seen_pairs = set()

        max_attempts = M * 20
        attempts = 0

        while len(constraints) < M and attempts < max_attempts:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                attempts += 1
                continue
            pair = (i, j)
            if pair in seen_pairs:
                attempts += 1
                continue

            base = gt[i] - gt[j]

            # Randomly choose <= or >=
            if random.random() < 0.5:
                # x[i] - x[j] <= c   with c >= base
                slack = random.choices(
                    [0, 1, 2, 3, 4, 5, 10],
                    weights=[3, 2, 2, 1, 1, 1, 1],
                    k=1
                )[0]
                c = base + slack
                constraints.append((i, j, c, '<='))
            else:
                # x[i] - x[j] >= c   with c <= base
                slack = random.choices(
                    [0, 1, 2, 3, 4, 5, 10],
                    weights=[3, 2, 2, 1, 1, 1, 1],
                    k=1
                )[0]
                c = base - slack
                constraints.append((i, j, c, '>='))

            seen_pairs.add(pair)
            attempts += 1

        # If still short, allow duplicate pairs
        while len(constraints) < M:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                continue
            base = gt[i] - gt[j]
            if random.random() < 0.5:
                c = base + random.randint(0, 10)
                constraints.append((i, j, c, '<='))
            else:
                c = base - random.randint(0, 10)
                constraints.append((i, j, c, '>='))

        random.shuffle(constraints)
        self._constraints = constraints[:M]
        self._ground_truth = gt
        self.parameter['constraints'] = self._constraints
        self.parameter['ground_truth'] = self._ground_truth

    def _prompt_generate(self) -> str:
        """Format the prompt with constraints."""
        M = len(self._constraints)

        lines = []
        for i, j, c, op in self._constraints:
            lines.append(f"  x[{i}] - x[{j}] {op} {c}")
        inequations = '\n'.join(lines)

        return self.prompt_template.format(
            N=len(self._ground_truth),
            N_minus_1=len(self._ground_truth) - 1,
            M=M,
            inequations=inequations,
        )

    # ------------------------------------------------------------------
    # Answer parsing
    # ------------------------------------------------------------------
    def _try_parse_clean_tokens(self, line: str) -> Optional[List[int]]:
        """Parse whitespace/comma/semicolon-separated integer tokens."""
        cleaned = re.sub(r'[,;]', ' ', line)
        tokens = cleaned.split()
        if not tokens:
            return None
        try:
            return [int(t) for t in tokens]
        except ValueError:
            return None

    def _try_parse_x_equals(self, text: str) -> Optional[List[int]]:
        """Extract values from patterns like x[0]=5, x0=5, x[1]=-3, etc."""
        N = len(self._ground_truth)
        # Match: x [ 0 ] = 5   or   x0 = 5   (optional brackets)
        pattern = re.findall(r'x\s*\[?\s*(\d+)\s*\]?\s*=\s*(-?\d+)', text)
        if not pattern:
            return None
        d = {}
        for idx_str, val_str in pattern:
            idx = int(idx_str)
            if idx < N:
                d[idx] = int(val_str)
        if len(d) < N:
            return None
        try:
            return [d[i] for i in range(N)]
        except KeyError:
            return None

    def _try_parse_all_ints_fallback(self, text: str) -> Optional[List[int]]:
        """Last-resort: find all integer substrings, take first N."""
        N = len(self._ground_truth)
        all_nums = [int(x) for x in re.findall(r'-?\d+', text)]
        if len(all_nums) >= N:
            return all_nums[:N]
        return None

    def _parse_answer_text(self, text: str) -> Optional[List[int]]:
        """Try to extract N integers from free-form text."""
        if not text:
            return None
        N = len(self._ground_truth)

        # 1. Clean-token lines (prefer last matching line)
        for line in reversed(text.split('\n')):
            nums = self._try_parse_clean_tokens(line.strip())
            if nums is not None and len(nums) >= N:
                return nums[:N]

        # 2. x[i]=value pattern
        nums = self._try_parse_x_equals(text)
        if nums is not None:
            return nums

        # 3. Greedy fallback
        return self._try_parse_all_ints_fallback(text)

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer extracted from <answer> tags."""
        if answer is None:
            return None
        return self._parse_answer_text(answer)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """Score the model output.

        Returns:
            1.0  – every constraint satisfied
            -1.0 – no valid answer could be parsed
            Linear interpolation between based on satisfied fraction.
        """
        if not self._constraints:
            return 0.0

        # Primary path: base-class processor (handles <answer> tags, prefixes)
        parsed = self.processor(output)

        # Fallback: parse raw output directly
        if parsed is None:
            text = output
            for prefix in ["Assistant:", "<|im_start|>assistant"]:
                if prefix in text:
                    text = text.split(prefix, 1)[1]
                    break
            parsed = self._parse_answer_text(text)

        if parsed is None:
            return -1.0

        x = parsed
        N = len(self._ground_truth)
        M = len(self._constraints)

        if len(x) != N:
            return -1.0

        satisfied = 0
        for i, j, c, op in self._constraints:
            diff = x[i] - x[j]
            if op == '<=':
                if diff <= c:
                    satisfied += 1
            else:  # '>='
                if diff >= c:
                    satisfied += 1

        fraction = satisfied / M
        score = 2.0 * fraction - 1.0
        return score
