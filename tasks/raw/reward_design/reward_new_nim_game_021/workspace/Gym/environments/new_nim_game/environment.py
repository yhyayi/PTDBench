# scorer(output) -> float in [-1.0, +1.0].

import random
from typing import Optional, List, Set
from Gym.environment import VerifiableEnvironment


class NewNimGame_Environment(VerifiableEnvironment):
    prompt_template = (
        "You are given a Nim-like game with heaps of matches. "
        "There are {N} heaps with the following sizes (1-indexed): {A}\n"
        "Game rules:\n"
        "- **First round** has two phases:\n"
        "  1) **Your move (first player):** You may remove **any number of entire heaps** "
        "(possibly zero), but you are **not allowed** to remove **all** heaps.\n"
        "  2) **Opponent's move (second player):** Then the opponent may remove "
        "**any number of entire heaps** (possibly zero), but likewise cannot remove "
        "**all remaining** heaps.\n"
        "- **From the second round onward:** Standard Nim rules apply on the remaining "
        "heaps: players alternate; a move removes any positive number of matches from "
        "**exactly one** heap; the player who takes the last match **wins**.\n"
        "- Both players play optimally.\n\n"
        "Your task: Choose which heaps to remove **in your first move** so that you "
        "**guarantee a win**; if multiple winning choices exist, choose one that "
        "**minimizes the total number of matches** you remove (i.e., the sum of sizes "
        "of the heaps you remove). Output the distinct *indices** (1-based) of the "
        "heaps you remove in your first move, in any order, separated by spaces; if "
        "you can guarantee victory without removing any heap, output an **empty line**."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N: int = 0
        self.A: List[int] = []          # heap sizes (0-indexed)

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #
    def _generate(self) -> None:
        """Sample N random heap sizes in [1, 2^N] so dependencies may occur."""
        N = int(self.parameter.get("N", 3))
        self.N = N
        max_val = max(1, (1 << N) - 1)   # up to 2^N - 1
        self.A = [random.randint(1, max_val) for _ in range(N)]

    # ------------------------------------------------------------------ #
    #  Prompt
    # ------------------------------------------------------------------ #
    def _prompt_generate(self) -> str:
        a_str = "[" + ", ".join(str(x) for x in self.A) + "]"
        return self.prompt_template.format(N=self.N, A=a_str)

    # ------------------------------------------------------------------ #
    #  Answer parsing
    # ------------------------------------------------------------------ #
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string into a list of 1-based heap indices."""
        if answer is None:
            return None
        answer = answer.strip()
        if answer == "":
            return []                     # empty line  -> remove nothing
        try:
            parts = answer.split()
            indices = [int(p) for p in parts]
            return indices
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    #  Linear‑independence helpers over GF(2) (XOR)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_independent(values: List[int]) -> bool:
        """Return True iff *values* are linearly independent over GF(2)."""
        basis: List[int] = []             # each element has a unique msb
        for x in values:
            y = x
            for b in basis:
                y = min(y, y ^ b)         # eliminate highest set bit if possible
            if y > 0:
                basis.append(y)
                basis.sort(reverse=True)  # keep largest msb first
            else:
                return False
        return True

    def _max_weight_independent_set(self) -> Set[int]:
        """
        Greedy max-weight independent set over GF(2).
        Returns the set of *kept* heap indices (0-based) that maximises total size.
        """
        N = self.N
        A = self.A
        # sort indices by size descending
        order = sorted(range(N), key=lambda i: A[i], reverse=True)
        basis: List[int] = []
        kept: Set[int] = set()
        for idx in order:
            x = A[idx]
            y = x
            for b in basis:
                y = min(y, y ^ b)
            if y > 0:
                basis.append(y)
                basis.sort(reverse=True)
                kept.add(idx)
        return kept

    def _optimal_removal(self) -> Set[int]:
        """Return the 1‑based indices that should be removed for an optimal win."""
        kept_0based = self._max_weight_independent_set()
        all_0based = set(range(self.N))
        removed_0based = all_0based - kept_0based
        return {i + 1 for i in removed_0based}

    # ------------------------------------------------------------------ #
    #  Scoring
    # ------------------------------------------------------------------ #
    def scorer(self, output: str) -> float:
        """
        Score the model answer.
        Reward scheme:
            +1.00   perfect answer (winning AND minimal removed sum)
            +0.40..+0.99  winning but sum is sub‑optimal (ratio‑based)
            -1.00   unparseable or losing strategy
        """
        # 1.  Parse
        parsed = self.processor(output)           # calls _process internally
        if parsed is None:
            return -1.0

        S = set(parsed)                           # 1‑based indices the model wants to remove
        all_indices = set(range(1, self.N + 1))

        # 2.  Validate
        if not S.issubset(all_indices):
            return -1.0                           # out‑of‑range index
        if S == all_indices:
            return -1.0                           # cannot remove ALL heaps

        # 3.  Check winning condition
        kept = all_indices - S
        kept_sizes = [self.A[i - 1] for i in kept]
        if not self._is_independent(kept_sizes):
            return -1.0                           # losing strategy

        # 4.  Winning – reward by optimality
        optimal_removed = self._optimal_removal()
        optimal_sum = sum(self.A[i - 1] for i in optimal_removed)
        actual_sum = sum(self.A[i - 1] for i in S)

        if actual_sum == optimal_sum:
            return 1.0

        # Partial credit: the closer the removed sum is to optimal, the better.
        # When optimal_sum == 0 the only perfect answer is empty, so actual_sum>0
        # means sub‑optimal; give a floor reward.
        if optimal_sum == 0:
            return 0.4

        ratio = optimal_sum / actual_sum          # in (0, 1)
        # Map ratio ∈ (0,1)  →  reward ∈ [0.4, 1.0)
        reward = 0.4 + 0.6 * ratio
        return float(reward)
