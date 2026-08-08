# Protecting Flowers (USACO 2007 January Silver)
# Minimize sum_i S[i] * D[p[i]] where S[i] = sum_{j < i} T[p[j]].
# Optimal: sort by T[i]/D[i] ascending (Smith's rule).
#
# scorer(output) -> float in [-1.0, +1.0]
# reward = score; accuracy = 1 iff score >= passing_reward_threshold (default 1.0)

import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ProtectingFlowers_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given two arrays `T` and `D`, each containing {N} integers:\n'
        '{T_and_D}\n'
        '\n'
        'Please output **a permutation of 1 to {N}**, denoted as p[1], p[2], ..., p[{N}] '
        '({N} integers in one line) with adjacent numbers separated by spaces:\n'
        '- Define S[i] as the sum of T[p[j]] for all 1 ≤ j < i (so S[1] = 0).\n'
        '- The objective is to minimize the total sum of S[i] * D[p[i]] for i from 1 to {N}.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  _generate
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate random T and D arrays and precompute optimal/worst costs."""
        N = self.parameter.get("N", 3)

        # Generate T and D values
        # T: 1..1000, D: 1..100  (positive integers, prevent division by zero)
        T = [random.randint(1, 1000) for _ in range(N)]
        D = [random.randint(1, 100) for _ in range(N)]

        self.parameter["T"] = T
        self.parameter["D"] = D

        # Compute optimal cost: sort indices by T[i]/D[i] ascending
        # Comparator: i before j iff T[i]*D[j] < T[j]*D[i]
        indices = list(range(N))
        # Use stable sort with key = T[i]/D[i] (float)
        # For exactness, we use the cross-product comparator
        from functools import cmp_to_key

        def cmp(i, j):
            # i before j if T[i]*D[j] < T[j]*D[i]
            lhs = T[i] * D[j]
            rhs = T[j] * D[i]
            if lhs < rhs:
                return -1
            elif lhs > rhs:
                return 1
            else:
                return 0

        opt_order = sorted(indices, key=cmp_to_key(cmp))
        opt_cost = self._compute_cost(opt_order, T, D)
        self.parameter["reference_answer"] = opt_cost
        self.parameter["opt_order"] = opt_order

        # Compute worst cost: reverse of optimal order
        worst_order = list(reversed(opt_order))
        worst_cost = self._compute_cost(worst_order, T, D)
        self.parameter["worst_cost"] = worst_cost

    # ------------------------------------------------------------------
    #  _compute_cost
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_cost(order: List[int], T: List[int], D: List[int]) -> int:
        """Compute sum_i S[i] * D[p[i]] for the given permutation order."""
        total = 0
        prefix_sum = 0
        for idx in order:
            total += prefix_sum * D[idx]
            prefix_sum += T[idx]
        return total

    # ------------------------------------------------------------------
    #  _prompt_generate
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        """Format the prompt with the generated T and D arrays."""
        N = self.parameter["N"]
        T = self.parameter["T"]
        D = self.parameter["D"]

        # Format T and D as two lines
        T_str = "T = [" + ", ".join(str(v) for v in T) + "]"
        D_str = "D = [" + ", ".join(str(v) for v in D) + "]"
        T_and_D = T_str + "\n" + D_str

        return self.prompt_template.format(N=N, T_and_D=T_and_D)

    # ------------------------------------------------------------------
    #  _process
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the model's answer into a permutation of 1..N (1-indexed)."""
        if answer is None:
            return None

        # Split by whitespace and commas
        import re
        tokens = re.findall(r'-?\d+', answer)
        try:
            perm = [int(t) for t in tokens]
        except ValueError:
            return None

        N = self.parameter.get("N", 0)
        if len(perm) != N:
            return None

        # Check that it's a valid permutation of 1..N
        if sorted(perm) != list(range(1, N + 1)):
            return None

        return perm

    # ------------------------------------------------------------------
    #  scorer
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].

        - +1.0: optimal permutation
        - -1.0: worst possible permutation (reverse optimal)
        - Linear interpolation in between
        - If output is unparseable: -1.0
        """
        perm = self.processor(output)
        if perm is None:
            return -1.0

        N = self.parameter["N"]
        T = self.parameter["T"]
        D = self.parameter["D"]

        model_cost = self._compute_cost([p - 1 for p in perm], T, D)  # convert 1-indexed -> 0-indexed

        opt_cost = self.parameter["reference_answer"]
        worst_cost = self.parameter["worst_cost"]

        if worst_cost == opt_cost:
            # All permutations have the same cost
            return 1.0

        # Normalized: 0 (worst) to 1 (optimal)
        normalized = (worst_cost - model_cost) / (worst_cost - opt_cost)
        # Clamp to [0, 1] in case of numerical issues
        normalized = max(0.0, min(1.0, normalized))

        # Map to [-1.0, +1.0]
        score = 2.0 * normalized - 1.0
        return float(score)

    # ------------------------------------------------------------------
    #  verifier (override is not needed; base class calls scorer)
    # ------------------------------------------------------------------
