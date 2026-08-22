# Environment for MaxDifferentGroupPairDivision
# scorer(output) -> float in [-1.0, +1.0].

import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class MaxDifferentGroupPairDivision_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given an array A of {N} integers: {A}\n'
        '\n'
        'Initially, the entire array is one single block. Let S = 0. '
        'You need to perform the following operation exactly {K} times:\n'
        '- Choose a position `i` such that A[i] and A[i + 1] are still in the same block.\n'
        '- Split the block into two parts: the first ends at A[i], the second starts at A[i + 1].\n'
        '- Let `sum1` and `sum2` be the sums of the two blocks. Then, update S += sum1 × sum2.\n'
        '\n'
        'After {K} operations, you will have {K} + 1 blocks. '
        'Try your best to **maximize** the final value of S.\n'
        '\n'
        '**Output Format:** A single line containing {K} integers — the positions `i` you chose in order, '
        r'separated by spaces. Put your answer inside \boxed{...} tags.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(self) -> None:
        """Generate a random problem instance and precompute optimal/worst S."""
        rng = random.Random(self.seed)

        # Sample N and K
        N = rng.randint(5, 15)
        K = rng.randint(1, N - 1)

        # Sample array elements
        A = [rng.randint(-10, 10) for _ in range(N)]

        # Compute optimal (max) and worst (min) S via DP
        optimal_S, worst_S = self._compute_bounds(A, K)

        self.parameter = {
            "N": N,
            "K": K,
            "A": A,
            "optimal_S": optimal_S,
            "worst_S": worst_S,
        }

    def _compute_bounds(self, A: List[int], K: int) -> Tuple[int, int]:
        """Compute the maximum and minimum possible S for partitioning A into K+1 blocks.
        
        S = sum_{i<j} s_i * s_j = (T^2 - sum(s_i^2)) / 2
        where T = sum(A).
        Maximizing S is equivalent to minimizing sum of squared block sums.
        Minimizing S is equivalent to maximizing sum of squared block sums.
        """
        N = len(A)
        # Prefix sums for O(1) range sum
        pref = [0]
        for x in A:
            pref.append(pref[-1] + x)

        def range_sum(l, r):
            return pref[r] - pref[l]

        # DP for minimizing sum of squares (maximizes S)
        # dp_min[i][k] = min sum of squares for first i elements into k blocks
        INF = 10**18
        dp_min = [[INF] * (K + 2) for _ in range(N + 1)]
        dp_min[0][0] = 0
        for i in range(1, N + 1):
            max_k = min(i, K + 1)
            for k in range(1, max_k + 1):
                best = INF
                for j in range(k - 1, i):
                    s = range_sum(j, i)
                    val = dp_min[j][k - 1] + s * s
                    if val < best:
                        best = val
                dp_min[i][k] = best

        # DP for maximizing sum of squares (minimizes S)
        NEG_INF = -10**18
        dp_max = [[NEG_INF] * (K + 2) for _ in range(N + 1)]
        dp_max[0][0] = 0
        for i in range(1, N + 1):
            max_k = min(i, K + 1)
            for k in range(1, max_k + 1):
                best = NEG_INF
                for j in range(k - 1, i):
                    s = range_sum(j, i)
                    val = dp_max[j][k - 1] + s * s
                    if val > best:
                        best = val
                dp_max[i][k] = best

        T = pref[N]
        min_sum_sq = dp_min[N][K + 1]
        max_sum_sq = dp_max[N][K + 1]

        # S = (T^2 - sum_sq) / 2
        optimal_S = (T * T - min_sum_sq) // 2  # max S
        worst_S = (T * T - max_sum_sq) // 2    # min S

        return optimal_S, worst_S

    def _prompt_generate(self) -> str:
        """Format the prompt with the generated parameters."""
        p = self.parameter
        return self.prompt_template.format(
            N=p["N"],
            K=p["K"],
            A=str(p["A"]),
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the model's answer into a list of K positions (0-indexed)."""
        if answer is None:
            return None

        # Try to extract integers from the answer
        import re
        # Find all integers (possibly negative)
        numbers = re.findall(r'-?\d+', answer)
        if not numbers:
            return None

        try:
            positions = [int(x) for x in numbers]
        except ValueError:
            return None

        # We need exactly K positions
        K = self.parameter["K"]
        if len(positions) != K:
            return None

        return positions

    def _compute_S(self, positions: List[int]) -> Optional[int]:
        """Compute S from a sequence of split positions.
        
        Since order doesn't matter, we just sort positions to get the final partition.
        Returns None if positions are invalid.
        """
        N = self.parameter["N"]
        K = self.parameter["K"]
        A = self.parameter["A"]

        # Check all positions are valid and distinct
        if len(set(positions)) != K:
            return None
        for p in positions:
            if p < 0 or p >= N - 1:
                return None

        # Sort to get partition boundaries
        sorted_positions = sorted(positions)
        boundaries = [0] + [p + 1 for p in sorted_positions] + [N]

        # Compute block sums
        block_sums = []
        for i in range(len(boundaries) - 1):
            l, r = boundaries[i], boundaries[i + 1]
            block_sums.append(sum(A[l:r]))

        # S = sum_{i<j} s_i * s_j
        total = sum(block_sums)
        sum_sq = sum(s * s for s in block_sums)
        S = (total * total - sum_sq) // 2
        return S

    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].
        
        -1.0 : invalid output
        +1.0 : optimal S achieved
        Linear interpolation between worst_S (-1) and optimal_S (+1).
        """
        processed = self.processor(output)
        if processed is None:
            return -1.0

        S = self._compute_S(processed)
        if S is None:
            return -1.0

        optimal_S = self.parameter["optimal_S"]
        worst_S = self.parameter["worst_S"]

        if optimal_S == worst_S:
            # All partitions give the same S; any valid answer is optimal
            return 1.0

        # Map [worst_S, optimal_S] to [-1, 1]
        score = 2.0 * (S - worst_S) / (optimal_S - worst_S) - 1.0
        # Clamp for safety (floating point)
        score = max(-1.0, min(1.0, score))
        return float(score)
