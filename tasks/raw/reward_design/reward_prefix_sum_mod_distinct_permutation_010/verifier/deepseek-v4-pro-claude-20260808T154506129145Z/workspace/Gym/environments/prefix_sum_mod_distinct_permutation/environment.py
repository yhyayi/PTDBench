# Environment: PrefixSumMODDistinctPermutation
# Task: Find a permutation of {1..N} such that all prefix sums are distinct modulo N.
#
# Mathematical note: a solution exists iff N is even or N = 1.
# For odd N > 1 the total sum N(N+1)/2 ≡ 0 (mod N), which forces the last
# prefix sum to be 0 mod N; but then either 0 appears twice or the set of
# residues {0,...,N-1} cannot be fully covered.
#
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List

from Gym.environment import VerifiableEnvironment


class PrefixSumMODDistinctPermutation_Environment(VerifiableEnvironment):
    prompt_template = (
        'Please find a permutation of the numbers from 1 to {N} such that all {N} '
        'prefix sums (i.e., the sum of the first i numbers for all i from 1 to {N}) '
        'are **distinct modulo {N}**. Output the permutation as {N} integers '
        '(in order) in one line, separated by spaces.'
    )

    # Only even N >= 2 guarantee a solution exists.
    # N=1 is trivial; we start from N=2 for meaningful problems.
    _ALLOWED_N = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ---------------------------------------------------------------- #
    #  _generate
    # ---------------------------------------------------------------- #
    def _generate(self) -> None:
        """Sample a feasible N and store it in self.parameter."""
        if "N" not in self.parameter:
            self.parameter["N"] = random.choice(self._ALLOWED_N)

    # ---------------------------------------------------------------- #
    #  _prompt_generate
    # ---------------------------------------------------------------- #
    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        return self.prompt_template.format(N=N)

    # ---------------------------------------------------------------- #
    #  _process
    # ---------------------------------------------------------------- #
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string into a list of integers.

        Returns None if the answer cannot be parsed or contains no integers.
        """
        if answer is None:
            return None

        numbers = re.findall(r'-?\d+', answer)
        if not numbers:
            return None

        try:
            parsed = [int(x) for x in numbers]
        except (ValueError, OverflowError):
            return None

        return parsed

    # ---------------------------------------------------------------- #
    #  scorer
    # ---------------------------------------------------------------- #
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].

        Scoring strategy (two-tier):
        ----------------------------------------
        Tier 1 – Not a valid permutation of {1..N}:
            score = -1.0 + coverage
            where coverage = (distinct in-range numbers) / N.
            This yields a score in [-1.0, 0.0).  The model gets credit for
            producing numbers that belong to the target set.

        Tier 2 – Valid permutation:
            Let k = number of distinct prefix-sum residues modulo N
            (1 ≤ k ≤ N).
            score = -1.0 + 2.0 * (k - 1) / (N - 1)   for N > 1,
            score = 1.0 for N = 1.
            This linearly maps k=1 → -1.0 and k=N → +1.0.
        """
        N = self.parameter.get("N", 2)
        if N is None or N < 1:
            N = 2

        # Extract the answer portion from the raw output.
        answer_text = self._extract_answer_from_output(output)
        parsed = self._process(answer_text)

        # --- No integers found -------------------------------------- #
        if parsed is None or len(parsed) == 0:
            return -1.0

        # --- Take the first N entries as the candidate permutation --- #
        candidate = parsed[:N]

        # Numbers in [1, N] that appear (deduplicated)
        in_range = [x for x in candidate if 1 <= x <= N]
        distinct_in_range = set(in_range)

        # --- Tier 1: not a full permutation ------------------------- #
        if len(distinct_in_range) < N:
            coverage = len(distinct_in_range) / N  # ∈ [0, 1)
            score = -1.0 + coverage
            return max(-1.0, min(score, 1.0))

        # --- Tier 2: valid permutation → score by residue diversity - #
        perm = candidate  # exactly N elements, all in [1,N], all distinct

        if N == 1:
            return 1.0

        # Compute prefix sums modulo N
        seen_residues = set()
        current = 0
        for x in perm:
            current += x
            seen_residues.add(current % N)

        k = len(seen_residues)  # ∈ [1, N]
        score = -1.0 + 2.0 * (k - 1) / (N - 1)
        return max(-1.0, min(score, 1.0))

    # ---------------------------------------------------------------- #
    #  Helper: extract answer from full model output
    # ---------------------------------------------------------------- #
    @staticmethod
    def _extract_answer_from_output(output: str) -> Optional[str]:
        """Isolate the answer portion of a full model output.

        Mirrors the base-class ``processor`` logic (strips assistant prefix,
        then looks for ``<answer>…</answer>``) so that ``scorer`` can be
        called on raw model outputs.
        """
        if output is None:
            return None

        text = output
        # Strip everything before the assistant turn.
        for marker in ("Assistant:", "<|im_start|>assistant"):
            if marker in text:
                text = text.split(marker, 1)[1]
                break

        # Try to extract content between <answer> … </answer>.
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Fallback: use the whole (post-assistant) text.
        return text.strip()
