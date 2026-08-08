# CampfireParty environment: rearrange students in a circle using cyclic rotations.
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Dict, Tuple

from Gym.environment import VerifiableEnvironment


class CampfireParty_Environment(VerifiableEnvironment):
    prompt_template = (
        "There are {N} students labeled from `0` to `{N_minus_1}`. "
        "At the beginning, they are sitting in a **circle** in the order: "
        "`0, 1, ..., {N_minus_1}`. Each student has **two specific friends** "
        "they want to sit next to. Your task is to rearrange the students "
        "around the circle so that **every student is adjacent to both of "
        "their desired neighbours**.\n"
        "{desired_neighbours}\n\n"
        "To achieve this, you may perform a series of operations. Each "
        "operation is represented as a tuple `(b_1, b_2, ..., b_m)`, where:\n"
        "- The student `b_1` moves to the position of `b_2`, `b_2` moves to "
        "the position of `b_3`, ..., and `b_m` moves to the position of `b_1`.\n"
        "- The cost of an operation is equal to the number of students "
        "involved (`m`).\n"
        "- No student may appear more than once in a single operation.\n\n"
        "Your goal is to achieve the desired circular arrangement using the "
        "**minimum total cost** across all operations.\n\n"
        "**Output Format:**\n"
        "Your final answer should contain K lines, where K is the number of "
        "operations you perform. The K lines should each describe one "
        "operation: a space-separated list of the students involved in that "
        "operation, in the order `(b_1, b_2, ..., b_m)`.\n"
        "Example (do **NOT** include the backticks or quotes):\n"
        "```\n"
        "0 1 2\n"
        "1 2\n"
        "2 3\n"
        "```\n"
        "This means:\n"
        "- There are 3 operations,\n"
        "- The first operation rotates students 0 → 1 → 2 → 0,\n"
        "- The second rotates (swaps) students 1 ↔ 2,\n"
        "- The third rotates (swaps) students 2 ↔ 3,\n"
        "- And the total cost is `3 + 2 + 2 = 7`.\n"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Problem generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate a random problem instance and precompute the optimal cost."""
        N: int = self.parameter.get("N", 4)
        self.parameter["N"] = N

        # Random target circular order – a permutation of 0..N-1.
        target_order: List[int] = list(range(N))
        random.shuffle(target_order)
        self.parameter["target_order"] = target_order

        # Derive desired neighbours from the target order.
        desired: Dict[int, List[int]] = {}
        for i, student in enumerate(target_order):
            left = target_order[(i - 1) % N]
            right = target_order[(i + 1) % N]
            desired[student] = sorted([left, right])
        self.parameter["desired"] = desired

        # Precompute optimal cost over all rotations and reversal of the circle.
        self.parameter["optimal_cost"], self.parameter["best_target_perm"] = (
            self._best_cost_for_circle(target_order)
        )

    # ------------------------------------------------------------------
    #  Prompt generation
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        N: int = self.parameter["N"]
        desired: Dict[int, List[int]] = self.parameter["desired"]

        lines: List[str] = []
        for student in range(N):
            a, b = desired[student]
            lines.append(
                f"- Student {student}: wants to sit next to {a} and {b}"
            )
        desired_str = "\n".join(lines)

        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            desired_neighbours=desired_str,
        )

    # ------------------------------------------------------------------
    #  Output processing / parsing
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[List[int]]]:
        """Parse the model answer into a list of operations.

        Returns:
            ``None`` – *answer* is ``None`` (no answer-marker tag found).
            ``[]``   – explicit empty answer (whitespace-only, or non-empty
                       text that contains no parseable integer cycles).
            Otherwise a list of integer lists.
        """
        if answer is None:
            return None

        text = answer.strip()
        # Strip code fences if present.
        text = re.sub(r"```[a-zA-Z]*\s*", "", text).strip()

        if text == "":
            return []  # explicit empty answer

        operations: List[List[int]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                op = [int(p) for p in parts]
            except ValueError:
                continue  # skip lines that are not pure integers
            if len(op) < 2:
                continue
            operations.append(op)

        # Return the (possibly empty) list – the model used answer markers
        # so we treat this as a deliberate answer.
        return operations

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0]."""
        N: int = self.parameter["N"]
        desired: Dict[int, List[int]] = self.parameter["desired"]
        optimal_cost: int = self.parameter["optimal_cost"]

        # --- 1. Extract operations ------------------------------------
        answer_text = self._extract_answer(output)
        operations = self._process(answer_text)

        if operations is None:
            # No answer-marker tag – try an aggressive parse of the full output.
            operations = self._parse_operations_from_text(output)
            if operations is None:
                # Truly unparseable – reward -1.0.
                # Exception: optimal cost is 0 and output is empty
                # (model may have realised no moves are needed).
                if output.strip() == "" and optimal_cost == 0:
                    return 1.0
                return -1.0

        # --- 2. Validate operations -----------------------------------
        for op in operations:
            if len(set(op)) != len(op):
                return -1.0  # duplicate student in one operation
            for s in op:
                if not (0 <= s < N):
                    return -1.0  # out-of-range label

        # --- 3. Simulate ----------------------------------------------
        # Position i holds student positions[i].
        positions: List[int] = list(range(N))

        total_cost = 0
        for op in operations:
            # Build student → current position mapping.
            student_to_pos: Dict[int, int] = {
                student: pos for pos, student in enumerate(positions)
            }
            m = len(op)
            total_cost += m

            new_positions = list(positions)
            for j in range(m):
                student = op[j]
                target_pos = student_to_pos[op[(j + 1) % m]]
                new_positions[target_pos] = student
            positions = new_positions

        # --- 4. Check constraints -------------------------------------
        satisfied_count = 0
        for pos, student in enumerate(positions):
            left_neighbour = positions[(pos - 1) % N]
            right_neighbour = positions[(pos + 1) % N]
            expected = set(desired[student])
            actual = {left_neighbour, right_neighbour}
            if actual == expected:
                satisfied_count += 1

        if satisfied_count < N:
            # Partial credit: fraction of satisfied students.
            fraction = satisfied_count / N
            score = -1.0 + 2.0 * fraction
            # Never reach +1.0 unless perfect.
            return min(score, 0.999)

        # --- 5. Cost-based scoring (all constraints satisfied) --------
        if total_cost <= optimal_cost:
            return 1.0

        # Valid but suboptimal.
        # excess = (actual - optimal) / max(1, optimal+1)
        # score  = 1 - 2 * excess, clamped to [-1.0, 1.0].
        excess = (total_cost - optimal_cost) / max(1.0, float(optimal_cost) + 1.0)
        score = 1.0 - 2.0 * excess
        return max(-1.0, score)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _extract_answer(self, output: str) -> Optional[str]:
        """Return the text between the configured answer markers."""
        left, right = self.answer_markers
        pattern = re.escape(left) + r"(.*?)" + re.escape(right)
        matches = list(re.finditer(pattern, output, re.DOTALL))
        if matches:
            return matches[-1].group(1)
        return None

    @staticmethod
    def _parse_operations_from_text(text: str) -> Optional[List[List[int]]]:
        """Aggressive fallback parse: find lines that look like cycles.

        Only called when no answer-marker tag was found.  Returns ``None``
        when the text is empty/whitespace or when no integer cycles can be
        extracted – the model did not follow the required format in a
        recoverable way.
        """
        if text.strip() == "":
            return None

        text = re.sub(r"```[a-zA-Z]*\s*", "", text)
        ops: List[List[int]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                op = [int(p) for p in parts]
            except ValueError:
                continue
            if len(op) >= 2:
                ops.append(op)

        if not ops:
            return None  # no cycles found → unparseable
        return ops

    @staticmethod
    def _cycle_cost(perm: List[int]) -> int:
        """Sum of lengths of non-trivial cycles in *perm* (perm[i] = image of i)."""
        n = len(perm)
        visited = [False] * n
        total = 0
        for i in range(n):
            if not visited[i]:
                length = 0
                j = i
                while not visited[j]:
                    visited[j] = True
                    length += 1
                    j = perm[j]
                if length > 1:
                    total += length
        return total

    @classmethod
    def _best_cost_for_circle(
        cls, target_order: List[int]
    ) -> Tuple[int, List[int]]:
        """Find the minimum cycle-decomposition cost over all rotations and
        reversal of *target_order*.  Returns ``(best_cost, best_perm)`` where
        *best_perm* is the position→student mapping that achieves *best_cost*.
        """
        n = len(target_order)
        best_cost = 10 ** 9
        best_perm: List[int] = []

        # All rotations of the forward order.
        for shift in range(n):
            perm = [target_order[(pos + shift) % n] for pos in range(n)]
            c = cls._cycle_cost(perm)
            if c < best_cost:
                best_cost = c
                best_perm = perm

        # All rotations of the reversed order.
        rev_order = list(reversed(target_order))
        for shift in range(n):
            perm = [rev_order[(pos + shift) % n] for pos in range(n)]
            c = cls._cycle_cost(perm)
            if c < best_cost:
                best_cost = c
                best_perm = perm

        return best_cost, best_perm
