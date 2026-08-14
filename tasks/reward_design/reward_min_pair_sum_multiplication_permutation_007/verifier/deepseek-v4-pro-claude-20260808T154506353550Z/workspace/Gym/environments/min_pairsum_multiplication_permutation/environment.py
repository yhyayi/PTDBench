# MinPairSumMultiplicationPermutation environment.
# scorer(output) -> float in [-1.0, +1.0].

import random
import itertools
import re
from typing import Optional, List, Tuple

from Gym.environment import VerifiableEnvironment


class MinPairSumMultiplicationPermutation_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given two matrices `A` and `B`, each of size {N} × {N}:\n'
        '{matrix_A}\n'
        '{matrix_B}\n\n'
        'You need to find a permutation P of indices from 0 to {N_minus_1} '
        'such that the value (sum of A[0][P[0]], A[1][P[1]], ..., '
        'A[{N_minus_1}][P[{N_minus_1}]]) multiplied by '
        '(sum of B[0][P[0]], B[1][P[1]], ..., '
        'B[{N_minus_1}][P[{N_minus_1}]]) is minimized.\n\n'
        '**Output Format:** A single line containing P[0], P[1], ..., '
        'P[{N_minus_1}], separated by spaces.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Internal state set during _generate:
        self._N: int = 0
        self._A: List[List[int]] = []
        self._B: List[List[int]] = []
        self._optimal_value: int = 0
        self._optimal_perm: Tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Problem generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate two N×N matrices and compute the optimal permutation."""
        self._N = self.parameter.get("N", 3)
        rng = random.Random(self.seed)  # use seeded rng for reproducibility

        # Generate matrices with values in [1, 100]
        self._A = [
            [rng.randint(1, 100) for _ in range(self._N)]
            for _ in range(self._N)
        ]
        self._B = [
            [rng.randint(1, 100) for _ in range(self._N)]
            for _ in range(self._N)
        ]

        # Brute-force optimal
        best_val = float("inf")
        best_perm = tuple(range(self._N))
        for perm in itertools.permutations(range(self._N)):
            sumA = sum(self._A[i][perm[i]] for i in range(self._N))
            sumB = sum(self._B[i][perm[i]] for i in range(self._N))
            val = sumA * sumB
            if val < best_val:
                best_val = val
                best_perm = perm
        self._optimal_value = best_val
        self._optimal_perm = best_perm

        # Store everything needed for prompt / scoring into self.parameter
        # so that config serialisation works correctly.
        self.parameter["N"] = self._N
        self.parameter["A"] = self._A
        self.parameter["B"] = self._B
        self.parameter["optimal_value"] = self._optimal_value
        self.parameter["optimal_perm"] = list(self._optimal_perm)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        """Return the formatted prompt string."""
        N = self._N
        # Format matrix A row by row
        matrix_A_lines = []
        for i, row in enumerate(self._A):
            row_str = "[" + ", ".join(str(x) for x in row) + "]"
            matrix_A_lines.append(f"A[{i}] = {row_str}")
        matrix_A_str = "\n".join(matrix_A_lines)

        # Format matrix B row by row
        matrix_B_lines = []
        for i, row in enumerate(self._B):
            row_str = "[" + ", ".join(str(x) for x in row) + "]"
            matrix_B_lines.append(f"B[{i}] = {row_str}")
        matrix_B_str = "\n".join(matrix_B_lines)

        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            matrix_A=matrix_A_str,
            matrix_B=matrix_B_str,
        )

    # ------------------------------------------------------------------
    # Answer processing
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string into a permutation list, or None if invalid."""
        if answer is None:
            return None

        # Strip whitespace and try to extract numbers
        answer = answer.strip()
        if not answer:
            return None

        # Try to split by whitespace, commas, or both
        # Replace commas with spaces, then split
        cleaned = answer.replace(",", " ")
        parts = cleaned.split()
        try:
            perm = [int(p) for p in parts]
        except ValueError:
            return None

        N = self._N
        if len(perm) != N:
            return None
        if sorted(perm) != list(range(N)):
            return None

        return perm

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0]."""
        # Use the processor to extract and validate the answer
        parsed = self.processor(output)

        if parsed is None:
            return -1.0

        perm = parsed  # List[int]

        # Compute the product for the submitted permutation
        sumA = sum(self._A[i][perm[i]] for i in range(self._N))
        sumB = sum(self._B[i][perm[i]] for i in range(self._N))
        got = sumA * sumB
        opt = self._optimal_value

        if got == opt:
            return 1.0

        # Continuous score for suboptimal answers.
        # Relative error = (got - opt) / opt.
        # Score = 1.0 - 2.0 * relative_error, but capped below at -0.99
        # so that only the true optimal receives +1.0.
        relative_error = (got - opt) / opt
        score = 1.0 - 2.0 * relative_error
        if score < -0.99:
            score = -0.99
        return float(score)
