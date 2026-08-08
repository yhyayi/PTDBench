# LandAcquisition: Partition items into sets to minimize sum of max(W)*max(L) per set.
# scorer(output) -> float in [-1.0, +1.0].

import random
import math
from typing import Optional, List, Tuple

from Gym.environment import VerifiableEnvironment


class LandAcquisition_Environment(VerifiableEnvironment):
    prompt_template = (
        'There are {N} items, and the i-th item has two attributes W[i] and L[i]. '
        'The arrays W and L are given as follows:\n'
        '{W_and_L}\n\n'
        'Partition all items into an arbitrary number of **disjoint non-empty sets**. '
        'For each set S, its cost is defined as: cost(S) = max(W[i] for i in S) × max(L[i] for i in S)\n'
        'Can you make the total cost, which is the sum of costs of all sets, as small as possible? '
        'Output M lines, where M is the number of sets in your partition - each line should contain '
        'the indices of the items in one set (separated by spaces).'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N: int = 0
        self.W: List[int] = []
        self.L: List[int] = []
        self.optimal_cost: float = 0.0

    # ------------------------------------------------------------------
    #  Problem generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        N = self.parameter.get("N", 4)

        # Generate random W and L in [1, 100]
        W = [random.randint(1, 100) for _ in range(N)]
        L = [random.randint(1, 100) for _ in range(N)]

        self.N = N
        self.W = W
        self.L = L
        self.optimal_cost = self._compute_optimal_cost()

    def _prompt_generate(self) -> str:
        # Format W and L as Python-like lists
        W_str = "W = " + str(self.W)
        L_str = "L = " + str(self.L)
        W_and_L = W_str + "\n" + L_str

        return self.prompt_template.format(
            N=self.N,
            W_and_L=W_and_L,
        )

    # ------------------------------------------------------------------
    #  Optimal cost computation (DP with domination filter)
    # ------------------------------------------------------------------
    def _compute_optimal_cost(self) -> float:
        """
        Compute the minimum possible total cost for partitioning the items.

        Algorithm:
        1. Sort items by W descending, then L descending.
        2. Remove dominated items: if an item has both W and L <= some other
           item, it can always be grouped with the dominating item for free.
        3. After filtering, W is strictly decreasing and L is strictly increasing.
        4. DP: dp[i] = min_{j>i} (W[i] * L[j-1] + dp[j])
           O(K^2) where K <= N is the number of non-dominated items.
        """
        N = self.N
        W = self.W
        L = self.L

        # Build list of (W, L) pairs with original indices
        items = [(W[i], L[i]) for i in range(N)]

        # Sort by W descending, then L descending
        items.sort(key=lambda x: (-x[0], -x[1]))

        # Filter dominated items: keep only those with strictly increasing L
        filtered = []
        max_L_so_far = -1
        for w, l in items:
            if l > max_L_so_far:
                filtered.append((w, l))
                max_L_so_far = l

        K = len(filtered)
        if K == 0:
            return 0.0

        Wf = [w for w, _ in filtered]
        Lf = [l for _, l in filtered]

        # DP: dp[i] = min cost for items i..K-1
        dp = [float('inf')] * (K + 1)
        dp[K] = 0.0

        for i in range(K - 1, -1, -1):
            max_L = 0
            for j in range(i + 1, K + 1):
                # max L in set [i, j-1] is Lf[j-1] (since Lf is increasing)
                cost = Wf[i] * Lf[j - 1]
                total = cost + dp[j]
                if total < dp[i]:
                    dp[i] = total

        return dp[0]

    # ------------------------------------------------------------------
    #  Answer processing / parsing
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[List[int]]]:
        """
        Parse the model's answer into a partition (list of lists of int indices).

        Returns None if parsing fails.
        """
        if answer is None:
            return None

        try:
            lines = answer.strip().splitlines()
            # Filter out empty lines
            lines = [line.strip() for line in lines if line.strip()]

            if not lines:
                return None

            partition = []
            seen = set()

            for line in lines:
                # Split by whitespace and/or commas
                tokens = line.replace(',', ' ').split()
                indices = []
                for tok in tokens:
                    try:
                        idx = int(tok)
                    except ValueError:
                        return None
                    if idx < 0 or idx >= self.N:
                        return None
                    if idx in seen:
                        return None  # duplicate index
                    indices.append(idx)
                    seen.add(idx)
                if not indices:
                    return None  # empty set not allowed
                partition.append(indices)

            # Check that all indices are covered exactly once
            if len(seen) != self.N:
                return None

            return partition

        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Cost of a given partition
    # ------------------------------------------------------------------
    def _partition_cost(self, partition: List[List[int]]) -> float:
        """Compute the total cost of a partition."""
        total = 0.0
        for group in partition:
            max_W = max(self.W[i] for i in group)
            max_L = max(self.L[i] for i in group)
            total += max_W * max_L
        return total

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """
        Score the model output.

        Returns:
            +1.0  if the partition achieves the optimal cost (or better).
            <1.0  otherwise, scaled based on optimal/actual ratio.
            -1.0  if the output cannot be parsed into a valid partition.
        """
        # Use processor to extract and parse the answer
        processed = self.processor(output)

        if processed is None:
            return -1.0

        actual_cost = self._partition_cost(processed)

        if actual_cost <= self.optimal_cost + 1e-9:
            return 1.0

        # Score: 2 * optimal / actual - 1, clamped to [-1.0, +1.0]
        # This gives:
        #   optimal       -> +1.0
        #   2 * optimal   ->  0.0
        #   3 * optimal   -> -0.333...
        #   very large    -> -1.0 (asymptotically)
        ratio = self.optimal_cost / actual_cost
        score = 2.0 * ratio - 1.0
        return max(-1.0, min(1.0, score))
