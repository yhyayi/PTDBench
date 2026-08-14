import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class QuadraticFunctionSegmentation_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given {N} numbers A[1], A[2], ..., A[{N}]. '
        'The values are given as: {A}\n\n'
        'You may divide these numbers (in order) into some **consecutive batches**. '
        'Let the total number of batches be k (1 ≤ k ≤ {N}), '
        'and let end[1], end[2], ..., end[k] (1 ≤ end[1] < end[2] < ... < end[k] = {N}) '
        'denote the last index in each batch. This means:\n'
        '- Batch 1 contains elements A[1] to A[end[1]]\n'
        '- Batch 2 contains elements A[end[1] + 1] to A[end[2]]\n'
        '- ...\n'
        '- Batch k contains elements A[end[k−1] + 1] to A[end[k]] (with end[k] = {N})\n\n'
        'Define the value of a batch with sum X as: '
        '**{A_coef} × X² + {B_coef} × X + {C_coef}**. '
        'The total value of the division is the **sum of values of all batches**. '
        'I am asking you to find a batch division that **maximizes** this total value.\n\n'
        'Output a single line containing `end[1] end[2] ... end[k]`, separated by spaces '
        '(with `end[k]` always equal to {N}`).\n'
        'Example: `1 2 {N}` means:\n'
        '- There are 3 batches,\n'
        '- First batch ends at index 1,\n'
        '- Second ends at index 2,\n'
        '- Third ends at index {N} and includes the remaining elements.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Will be set during _generate
        self.N: int = 0
        self.A: List[int] = []
        self.A_coef: int = 0
        self.B_coef: int = 0
        self.C_coef: int = 0
        self.optimal_value: int = 0
        self.worst_value: int = 0

    # ------------------------------------------------------------------
    #  Core helpers
    # ------------------------------------------------------------------
    def compute_value(self, X: int) -> int:
        """Value of a single batch whose element-sum is X."""
        return self.A_coef * X * X + self.B_coef * X + self.C_coef

    def _compute_dp(self) -> Tuple[int, int, List[int]]:
        """
        DP for both maximisation (used for optimal answer) and minimisation
        (used for the worst-possible value, to anchor partial-credit scoring).

        Returns (optimal_value, worst_value, one_optimal_partition_end_indices).
        """
        N = self.N
        # prefix sums: pref[i] = sum of A[0..i-1]
        pref = [0]
        for v in self.A:
            pref.append(pref[-1] + v)

        # ---- maximise ----
        dp_max = [-(10**30)] * (N + 1)
        dp_max[0] = 0
        best_prev = [-1] * (N + 1)
        for i in range(1, N + 1):
            best_val = -(10**30)
            best_j = -1
            for j in range(i):
                seg_sum = pref[i] - pref[j]
                val = dp_max[j] + self.compute_value(seg_sum)
                if val > best_val:
                    best_val = val
                    best_j = j
            dp_max[i] = best_val
            best_prev[i] = best_j

        optimal_value = dp_max[N]

        # recover one optimal partition (end indices)
        partition_ends = []
        cur = N
        while cur > 0:
            partition_ends.append(cur)
            cur = best_prev[cur]
        partition_ends.reverse()  # now ascending

        # ---- minimise ----
        dp_min = [10**30] * (N + 1)
        dp_min[0] = 0
        for i in range(1, N + 1):
            best_val = 10**30
            for j in range(i):
                seg_sum = pref[i] - pref[j]
                val = dp_min[j] + self.compute_value(seg_sum)
                if val < best_val:
                    best_val = val
            dp_min[i] = best_val

        worst_value = dp_min[N]

        return optimal_value, worst_value, partition_ends

    # ------------------------------------------------------------------
    #  Environment interface
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate a random problem instance using self.seed."""
        # Read N from the parameter dict (set by the parameter controller).
        N = self.parameter.get("N", 4)
        self.N = N

        # Generate array A: random integers in a moderate range
        self.A = [random.randint(-20, 20) for _ in range(N)]

        # Generate quadratic coefficients; A_coef must be non-zero to keep
        # the function truly quadratic.
        self.A_coef = random.choice([-3, -2, -1, 1, 2, 3])
        self.B_coef = random.randint(-5, 5)
        self.C_coef = random.randint(-5, 5)

        # Compute optimal / worst values and one optimal partition.
        opt_val, worst_val, opt_ends = self._compute_dp()
        self.optimal_value = opt_val
        self.worst_value = worst_val

        # Store the answer in the parameter dict so that it is available for
        # reference (though we score by value, not by exact partition).
        self.parameter["reference_answer"] = " ".join(str(e) for e in opt_ends)
        self.parameter["optimal_value"] = opt_val
        self.parameter["worst_value"] = worst_val

    def _prompt_generate(self) -> str:
        """Return the filled prompt."""
        A_str = " ".join(str(x) for x in self.A)
        return self.prompt_template.format(
            N=self.N,
            A=A_str,
            A_coef=self.A_coef,
            B_coef=self.B_coef,
            C_coef=self.C_coef,
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """
        Parse the model answer (text between <answer> … </answer>) into a list
        of end indices.  Returns None when parsing fails.
        """
        if answer is None:
            return None

        # Strip whitespace and split on whitespace / commas
        text = answer.strip()
        if not text:
            return None

        parts = text.replace(",", " ").split()
        try:
            ends = [int(p) for p in parts]
        except ValueError:
            return None

        # Validate basic structure
        if len(ends) == 0:
            return None

        # Must be strictly increasing
        for i in range(1, len(ends)):
            if ends[i] <= ends[i - 1]:
                return None

        # Last element must equal N
        if ends[-1] != self.N:
            return None

        # Every element must be in [1, N]
        for e in ends:
            if e < 1 or e > self.N:
                return None

        return ends

    def scorer(self, output: str) -> float:
        """
        Score the model output in [-1.0, +1.0].

        - Invalid output          → -1.0
        - Optimal partition       → +1.0
        - Otherwise: linear interpolation between worst_value (-1.0)
          and optimal_value (+1.0), clamped to [-1.0, +1.0].
        """
        # Use processor to extract answer from <answer>…</answer> if present.
        processed = self.processor(output)
        if processed is None:
            return -1.0

        ends: List[int] = processed

        # Compute the total value of the model's partition.
        pref = [0]
        for v in self.A:
            pref.append(pref[-1] + v)

        model_val = 0
        prev_end = 0
        for e in ends:
            seg_sum = pref[e] - pref[prev_end]
            model_val += self.compute_value(seg_sum)
            prev_end = e

        opt = self.optimal_value
        worst = self.worst_value

        if opt == worst:
            # Every partition gives the same value → perfect score.
            return 1.0

        # Map worst → -1.0, opt → +1.0
        score = -1.0 + 2.0 * (model_val - worst) / (opt - worst)

        # Clamp for numerical safety
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0

        return score
