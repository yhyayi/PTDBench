# Environment for PrefixProductMODDistinctPermutation
# Task: Find a permutation of [1..N] such that all N prefix products are distinct modulo N.
#
# Solutions exist only for N ∈ {1, 2, 3, 4} ∪ {odd primes}.
# For composite N > 4, (N-1)! ≡ 0 mod N, making it impossible to avoid duplicate zeros.
#
# scorer(output) -> float in [-1.0, +1.0]:
#   +1.0 : perfect valid permutation (all N prefix products distinct mod N)
#   partial: linear interpolation based on number of distinct prefix products
#   -1.0 : unparseable or completely wrong

import random
import re
from typing import Optional, List

from Gym.environment import VerifiableEnvironment


class PrefixProductMODDistinctPermutation_Environment(VerifiableEnvironment):
    prompt_template = (
        'Please find a permutation of the numbers from 1 to {N} such that '
        'all {N} prefix products (i.e., the product of the first i numbers '
        'for all i from 1 to {N}) are **distinct modulo {N}**. '
        'Output the permutation as {N} integers (in order) in one line, '
        'separated by spaces.'
    )

    @staticmethod
    def _is_solvable(n: int) -> bool:
        """Return True if a solution exists for the given N.
        
        Solvable N: 1, 2, 4, and all odd primes.
        For composite N > 4, (N-1)! ≡ 0 (mod N), so the product of the first
        N-1 numbers is always 0 mod N, making distinct prefix products impossible.
        """
        if n <= 0:
            return False
        if n == 1 or n == 2 or n == 4:
            return True
        if n % 2 == 0:
            return False  # even numbers > 2 are composite
        # Check primality for odd n > 1
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(self) -> None:
        """Generate problem: pick a random solvable N ≤ MAX_N."""
        max_n = self.parameter.get('MAX_N', 8)
        # Collect all solvable N in [1, max_n]
        candidates = [n for n in range(1, max_n + 1) if self._is_solvable(n)]
        if not candidates:
            candidates = [1]
        self.parameter['N'] = random.choice(candidates)

    def _prompt_generate(self) -> str:
        """Fill the prompt template with the chosen N."""
        n = self.parameter['N']
        return self.prompt_template.format(N=n)

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string into a list of integers.
        
        Returns None if parsing fails.
        """
        if answer is None:
            return None
        # Extract all integers from the string
        tokens = re.findall(r'-?\d+', answer)
        if not tokens:
            return None
        try:
            nums = [int(t) for t in tokens]
        except ValueError:
            return None
        return nums

    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].
        
        Scoring rubric:
        - If output cannot be parsed into N integers: -1.0
        - If parsed but not a valid permutation of 1..N:
            score based on fraction of required numbers that appear at least once
            (ranges from -1.0 to near 0.0)
        - If valid permutation:
            score based on fraction of distinct prefix products modulo N
            linearly mapped from [-1.0 (all same) to +1.0 (all distinct)]
        """
        n = self.parameter['N']
        nums = self._process(output)

        # --- unparseable ---
        if nums is None:
            return -1.0

        # --- Check if valid permutation ---
        required = set(range(1, n + 1))
        present = set(nums) & required
        coverage = len(present)

        is_valid_permutation = (set(nums) == required and len(nums) == n)

        if not is_valid_permutation:
            # coverage ∈ [0, N]. Map to [-1, 0).
            if n <= 1:
                return -1.0
            return -1.0 + coverage / n

        # --- valid permutation: compute prefix products ---
        prod = 1
        prefix_prods = []
        for x in nums:
            prod = (prod * x) % n
            prefix_prods.append(prod)

        distinct_count = len(set(prefix_prods))

        if n == 1:
            return 1.0 if distinct_count == 1 else -1.0

        # Map distinct_count ∈ [1, N] to [-1, +1]
        score = -1.0 + 2.0 * (distinct_count - 1) / (n - 1)
        return max(-1.0, min(1.0, score))
