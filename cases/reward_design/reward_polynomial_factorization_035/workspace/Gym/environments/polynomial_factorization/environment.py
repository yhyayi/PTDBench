# PolynomialFactorization environment: given a monic polynomial that factors as
# (x - a_1)...(x - a_N), recover the integer roots a_1,...,a_N.
#
# scorer(output) -> float in [-1.0, +1.0] with partial credit based on the
# fraction of correctly identified roots (respecting multiplicities).

import random
import re
from typing import Optional, List
from collections import Counter

from Gym.environment import VerifiableEnvironment


def _poly_to_str(coeffs: List[int]) -> str:
    """Convert coefficient list [c0, c1, ..., cN] (cN=1, monic) to a readable string.

    coeffs[i] is the coefficient of x^i.
    coeffs[N] = 1 (leading coefficient).
    Returns something like 'x^3 - 6x^2 + 11x - 6'.
    """
    N = len(coeffs) - 1
    terms = []
    for i in range(N, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == N:
            # leading term
            if c == 1:
                term = f"x^{i}" if i > 1 else ("x" if i == 1 else "1")
            elif c == -1:
                term = f"-x^{i}" if i > 1 else ("-x" if i == 1 else "-1")
            else:
                term = f"{c}x^{i}" if i > 1 else (f"{c}x" if i == 1 else f"{c}")
        else:
            abs_c = abs(c)
            sign = "+" if c > 0 else "-"
            if i == 0:
                term = f"{sign} {abs_c}"
            elif i == 1:
                term = f"{sign} {abs_c}x" if abs_c != 1 else f"{sign} x"
            else:
                term = f"{sign} {abs_c}x^{i}" if abs_c != 1 else f"{sign} x^{i}"
        terms.append(term)

    if not terms:
        return "0"

    result = " ".join(terms)
    # Fix leading "+ " if present
    if result.startswith("+ "):
        result = result[2:]
    return result


def _expand_roots(roots: List[int]) -> List[int]:
    """Expand (x - r0)*(x - r1)*...*(x - r_{N-1}) into coefficients.

    Returns list [c0, c1, ..., cN] where c_i is coefficient of x^i.
    cN = 1 (monic).
    """
    # Start with polynomial = 1 (constant 1)
    coeffs = [1]  # represents 1 * x^0
    for r in roots:
        # Multiply by (x - r)
        # new_coeffs[k] = old_coeffs[k-1] - r * old_coeffs[k]
        new_coeffs = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new_coeffs[i] += -r * c       # constant term from -r
            new_coeffs[i + 1] += c        # x term
        coeffs = new_coeffs
    return coeffs  # [c0, c1, ..., cN], cN=1


class PolynomialFactorization_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given a degree-{N} polynomial: (x - a_1)...(x - a_{N}) = {polynomial}\n\n'
        'Your task is to find any valid set of integers `a_1, ..., a_{N}` '
        '(not necessarily distinct) such that the product of the linear factors '
        'on the left expands to match the given polynomial.\n\n'
        '**Output Format:** Your final answer should be a single line containing '
        '`a_1, ..., a_{N}`, separated by **spaces**.\n\n'
        'Wrap your final answer in <answer>...</answer> tags.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers that read from self.parameter (works after both
    # _generate and set_config).
    # ------------------------------------------------------------------
    @property
    def _N(self) -> int:
        return self.parameter['N']

    @property
    def _roots(self) -> List[int]:
        return self.parameter['roots']

    @property
    def _coeffs(self) -> List[int]:
        return self.parameter['coeffs']

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate a random monic polynomial with integer roots."""
        # N between 2 and 8 for reasonable difficulty
        N = random.randint(2, 8)
        # Generate N random integer roots in [-10, 10]
        roots = [random.randint(-10, 10) for _ in range(N)]

        # Expand to get polynomial coefficients
        coeffs = _expand_roots(roots)

        # Store in parameter dict (sorted roots for consistent comparison)
        self.parameter['N'] = N
        self.parameter['roots'] = sorted(roots)
        self.parameter['coeffs'] = list(coeffs)

    def _prompt_generate(self) -> str:
        """Format the prompt with the polynomial."""
        poly_str = _poly_to_str(self._coeffs)
        return self.prompt_template.format(N=self._N, polynomial=poly_str)

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string to extract a list of N integers.

        Returns None if parsing fails.
        """
        if answer is None:
            return None

        # Find all integers in the answer
        cleaned = answer.strip()

        # Try to extract all integer tokens (handle negative numbers)
        tokens = re.findall(r'-?\d+', cleaned)
        if not tokens:
            return None

        try:
            values = [int(t) for t in tokens]
        except ValueError:
            return None

        # We expect exactly N values
        if len(values) != self._N:
            return None

        return values

    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].

        Uses partial credit: for each correctly identified root (respecting
        multiplicities), the model earns 1/N toward the score.

        Score = 2 * (num_correct / N) - 1, which maps:
          - all correct → +1.0
          - none correct → -1.0
          - half correct → 0.0
        """
        # Use processor to extract parsed answer
        parsed = self.processor(output)

        if parsed is None:
            return -1.0

        N = self._N
        expected_roots = self._roots

        # Compare multisets: count how many values match (respecting multiplicities)
        expected_counter = Counter(expected_roots)
        actual_counter = Counter(parsed)

        # Count matches: for each value, the number matched is
        # min(count in expected, count in actual)
        num_correct = 0
        for val, count in expected_counter.items():
            num_correct += min(count, actual_counter.get(val, 0))

        # Partial credit score in [-1.0, +1.0]
        score = 2.0 * (num_correct / N) - 1.0

        # Clamp to [-1.0, 1.0] for safety (should already be in range)
        score = max(-1.0, min(1.0, score))
        return score
