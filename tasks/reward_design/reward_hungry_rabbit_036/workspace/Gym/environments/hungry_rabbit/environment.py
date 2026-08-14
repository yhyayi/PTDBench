# HungryRabbit: Construct M sets of K integers from [1,N] such that
# consecutive sets differ by at most L elements.
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Dict, Any, Tuple

from Gym.environment import VerifiableEnvironment


class HungryRabbit_Environment(VerifiableEnvironment):
    prompt_template = (
        "Let's construct {M} sets of integers S(1), S(2), ..., S(M), "
        "where each set contains exactly {K} integers chosen from 1 to {N}. "
        "The following conditions must hold:\n"
        "- For all i (2 ≤ i ≤ {M}), we have {K} - |S(i) ∩ S(i - 1)| ≤ {L}.\n"
        "{constraints}"
        "\nOutput {M} lines, where the i-th line contains the {K} integers "
        "(in the range of [1, {N}]) in S(i), separated by spaces."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.M: int = 0
        self.N: int = 0
        self.K: int = 0
        self.L: int = 0
        self.additional_constraints_text: str = ""
        self._reference_solution: Optional[List[List[int]]] = None
        self._fixed_first: Optional[List[int]] = None
        self._fixed_last: Optional[List[int]] = None
        self._required_elements: Dict[int, List[int]] = {}
        self._forbidden_elements: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    #  Generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """
        Sample parameters and constraints, then build a feasible reference.
        Retry with simpler constraints if infeasible.
        """
        MAX_N_M = self.parameter.get("MAX_N_M", 4)

        for _attempt in range(30):
            N = random.randint(max(3, MAX_N_M), max(4, MAX_N_M + 4))
            M = random.randint(2, min(MAX_N_M + 2, 8))
            K = random.randint(max(2, N // 4), min(N - 1, max(3, N // 2 + 1)))
            L = random.randint(max(1, K // 3), K)

            self.N = N
            self.M = M
            self.K = K
            self.L = L

            self._fixed_first = None
            self._fixed_last = None
            self._required_elements = {}
            self._forbidden_elements = {}
            constraint_parts: List[str] = []

            r = random.random()

            if r < 0.30:
                first_set = sorted(random.sample(range(1, N + 1), K))
                self._fixed_first = first_set
                constraint_parts.append(
                    f"- S(1) must be exactly {{{', '.join(str(x) for x in first_set)}}}."
                )
            elif r < 0.50:
                last_set = sorted(random.sample(range(1, N + 1), K))
                self._fixed_last = last_set
                constraint_parts.append(
                    f"- S({M}) must be exactly {{{', '.join(str(x) for x in last_set)}}}."
                )
            elif r < 0.65:
                first_set = sorted(random.sample(range(1, N + 1), K))
                last_set = self._make_reachable_set(first_set, M - 1, L, N, K)
                self._fixed_first = first_set
                self._fixed_last = last_set
                constraint_parts.append(
                    f"- S(1) must be exactly {{{', '.join(str(x) for x in first_set)}}}."
                )
                constraint_parts.append(
                    f"- S({M}) must be exactly {{{', '.join(str(x) for x in last_set)}}}."
                )
            elif r < 0.80:
                pos = random.randint(1, M)
                elem = random.randint(1, N)
                self._required_elements[pos] = [elem]
                constraint_parts.append(
                    f"- S({pos}) must contain the integer {elem}."
                )
            else:
                pos = random.randint(1, M)
                elem = random.randint(1, N)
                self._forbidden_elements[pos] = [elem]
                constraint_parts.append(
                    f"- S({pos}) must NOT contain the integer {elem}."
                )

            self.additional_constraints_text = "\n".join(constraint_parts)
            if constraint_parts:
                self.additional_constraints_text += "\n"

            ref = self._build_feasible_solution()
            if ref is not None:
                self._reference_solution = ref
                return

        # Fallback: simple problem
        self._fixed_first = None
        self._fixed_last = None
        self._required_elements = {}
        self._forbidden_elements = {}
        self.additional_constraints_text = ""
        self.N = max(3, MAX_N_M)
        self.M = 2
        self.K = max(2, self.N // 2)
        self.L = self.K
        s = sorted(random.sample(range(1, self.N + 1), self.K))
        self._reference_solution = [s[:] for _ in range(self.M)]

    def _make_reachable_set(
        self, start: List[int], steps: int, L: int, N: int, K: int
    ) -> List[int]:
        """Walk `steps` random steps from `start`, changing ≤ L per step."""
        cur = set(start)
        for _ in range(steps):
            cur = self._random_step(cur, L, N, K)
        return sorted(cur)

    def _random_step(self, cur: set, L: int, N: int, K: int) -> set:
        """Change at most L elements of `cur`, keeping size K."""
        num_changes = random.randint(0, L)
        result = set(cur)
        if num_changes > 0 and result:
            to_remove = random.sample(sorted(result), min(num_changes, len(result)))
            result -= set(to_remove)
        available = set(range(1, N + 1)) - result
        need = K - len(result)
        if need > 0 and available:
            add = random.sample(sorted(available), min(need, len(available)))
            result |= set(add)
        while len(result) < K:
            avail = set(range(1, N + 1)) - result
            if not avail:
                break
            result.add(random.choice(sorted(avail)))
        while len(result) > K:
            result.remove(random.choice(sorted(result)))
        return result

    def _step_towards(self, cur: set, target: set, L: int, N: int, K: int) -> set:
        """Move from cur towards target, changing at most L elements."""
        to_add = list(target - cur)
        to_remove = list(cur - target)
        random.shuffle(to_add)
        random.shuffle(to_remove)

        result = set(cur)
        changes = 0

        # Remove elements not in target
        for elem in to_remove:
            if changes >= L:
                break
            result.discard(elem)
            changes += 1

        # Add elements from target
        for elem in to_add:
            if changes >= L or len(result) >= K:
                break
            result.add(elem)
            changes += 1

        # Ensure exactly K elements
        while len(result) < K:
            avail = set(range(1, N + 1)) - result
            if not avail:
                break
            result.add(random.choice(sorted(avail)))
        while len(result) > K:
            result.remove(random.choice(sorted(result)))
        return result

    def _distance(self, a: set, b: set) -> int:
        """Minimum number of element changes to go from a to b."""
        return max(len(a - b), len(b - a))

    def _build_feasible_solution(self) -> Optional[List[List[int]]]:
        """
        Build S(1)..S(M) satisfying all constraints.
        Strategy:
        - If fixed_last is set (without fixed_first), build backwards from M to 1.
        - Otherwise build forwards from 1 to M.
        - For fixed_first+fixed_last, build forwards, stepping towards target
          allocating changes across steps to ensure arrival.
        """
        M, N, K, L = self.M, self.N, self.K, self.L

        # ---- Build forwards (default) ----
        if self._fixed_last is None or self._fixed_first is not None:
            return self._build_forwards()

        # ---- Build backwards (only fixed_last) ----
        return self._build_backwards()

    def _build_forwards(self) -> Optional[List[List[int]]]:
        """Build S(1)..S(M) forwards, respecting fixed_first if set."""
        M, N, K, L = self.M, self.N, self.K, self.L

        # --- Position 1 ---
        cur = self._make_initial_set(1)
        if cur is None:
            return None
        seq = [sorted(cur)]

        # --- Positions 2..M ---
        for i in range(2, M + 1):
            prev = seq[-1]

            if i == M and self._fixed_last is not None:
                target = set(self._fixed_last)
                # Check if reachable in one step
                d = self._distance(set(prev), target)
                if d <= L:
                    cur = target
                else:
                    return None  # Not reachable; should not happen with good generation
            else:
                cur = self._random_step(set(prev), L, N, K)

            # Enforce position constraints
            cur = self._enforce_pos_constraints(i, cur)
            if cur is None:
                return None

            # Verify overlap
            if K - len(set(prev) & cur) > L:
                return None

            seq.append(sorted(cur))

        # Final check
        for i, s in enumerate(seq):
            if not self._check_pos(i + 1, set(s)):
                return None
        if self._fixed_last is not None and set(seq[-1]) != set(self._fixed_last):
            return None

        return seq

    def _build_backwards(self) -> Optional[List[List[int]]]:
        """Build backwards from S(M) to S(1) when only fixed_last is set."""
        M, N, K, L = self.M, self.N, self.K, self.L

        # --- Position M ---
        cur = set(self._fixed_last)
        if not self._check_pos(M, cur):
            return None
        seq_rev = [sorted(cur)]

        # --- Positions M-1 down to 1 ---
        for i in range(M - 1, 0, -1):
            next_set = seq_rev[-1]  # The "later" set already built
            # Generate a predecessor within L changes
            cur = self._random_step(set(next_set), L, N, K)
            cur = self._enforce_pos_constraints(i, cur)
            if cur is None:
                return None
            if K - len(cur & set(next_set)) > L:
                return None
            if self._fixed_first is not None and i == 1 and cur != set(self._fixed_first):
                return None
            seq_rev.append(sorted(cur))

        seq = list(reversed(seq_rev))

        if self._fixed_first is not None and set(seq[0]) != set(self._fixed_first):
            return None
        return seq

    def _make_initial_set(self, pos: int) -> Optional[set]:
        """Create a valid set for position `pos`, respecting constraints."""
        if pos == 1 and self._fixed_first is not None:
            s = set(self._fixed_first)
            return s if self._check_pos(pos, s) else None

        forbidden = set(self._forbidden_elements.get(pos, []))
        required = set(self._required_elements.get(pos, []))
        available = set(range(1, self.N + 1)) - forbidden
        if len(available) < self.K or len(required) > self.K:
            return None
        s = set(required)
        available -= s
        need = self.K - len(s)
        if need > len(available):
            return None
        s |= set(random.sample(sorted(available), need))
        return s

    def _enforce_pos_constraints(self, pos: int, cur: set) -> Optional[set]:
        """Adjust `cur` to satisfy required/forbidden at `pos` while keeping size K."""
        result = set(cur)

        # Remove forbidden elements
        for elem in self._forbidden_elements.get(pos, []):
            result.discard(elem)

        # Add required elements
        for elem in self._required_elements.get(pos, []):
            if elem not in result:
                if len(result) >= self.K:
                    removable = result - set(self._required_elements.get(pos, []))
                    if not removable:
                        return None
                    result.remove(random.choice(sorted(removable)))
                result.add(elem)

        # Fix size
        while len(result) < self.K:
            avail = set(range(1, self.N + 1)) - result - set(self._forbidden_elements.get(pos, []))
            if not avail:
                return None
            result.add(random.choice(sorted(avail)))
        while len(result) > self.K:
            removable = result - set(self._required_elements.get(pos, []))
            if not removable:
                return None
            result.remove(random.choice(sorted(removable)))

        return result

    def _check_pos(self, pos: int, s: set) -> bool:
        for elem in self._required_elements.get(pos, []):
            if elem not in s:
                return False
        for elem in self._forbidden_elements.get(pos, []):
            if elem in s:
                return False
        if pos == 1 and self._fixed_first is not None and s != set(self._fixed_first):
            return False
        if pos == self.M and self._fixed_last is not None and s != set(self._fixed_last):
            return False
        return True

    # ------------------------------------------------------------------
    #  Prompt
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        return self.prompt_template.format(
            M=self.M,
            N=self.N,
            K=self.K,
            L=self.L,
            constraints=self.additional_constraints_text,
        )

    # ------------------------------------------------------------------
    #  Process (parse model output)
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[List[int]]]:
        if answer is None:
            return None

        text = answer.strip()
        text = re.sub(r"```(?:python|text|plain)?\s*", "", text)
        text = text.replace("```", "")
        text = text.strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        data_lines = []
        for line in lines:
            cleaned = line.replace(",", " ").strip()
            tokens = cleaned.split()
            if tokens and all(re.match(r"^\d+$", t) for t in tokens):
                data_lines.append(tokens)

        if len(data_lines) < self.M:
            return None

        result: List[List[int]] = []
        for i in range(self.M):
            tokens = data_lines[i]
            nums = [int(t) for t in tokens]
            if len(nums) != self.K:
                return None
            result.append(nums)

        return result

    # ------------------------------------------------------------------
    #  Scorer
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """
        Score in [-1.0, +1.0].

        Components:
          format (0.20) – correct shape, values, no per-set duplicates
          overlap (0.50) – consecutive sets differ by ≤ L
          extras  (0.30) – fixed first/last, required, forbidden

        Partial credit proportional within each category.
        """
        parsed = self.processor(output)

        if parsed is None:
            return -1.0

        sets = parsed

        # --- 1. Format (0.20) ---
        for s in sets:
            if len(s) != self.K:
                return -0.5
            for val in s:
                if not (1 <= val <= self.N):
                    return -0.5
            if len(set(s)) != len(s):
                return -0.5
        format_score = 0.20

        # --- 2. Overlap (0.50) ---
        ok = 0
        total = self.M - 1
        for i in range(1, self.M):
            inter = len(set(sets[i - 1]) & set(sets[i]))
            if self.K - inter <= self.L:
                ok += 1
        overlap_score = 0.50 if total == 0 else 0.50 * (ok / total)

        # --- 3. Extras (0.30) ---
        items = 0
        met = 0
        if self._fixed_first is not None:
            items += 1
            if set(sets[0]) == set(self._fixed_first):
                met += 1
        if self._fixed_last is not None:
            items += 1
            if set(sets[-1]) == set(self._fixed_last):
                met += 1
        for pos, elems in self._required_elements.items():
            items += 1
            if all(e in sets[pos - 1] for e in elems):
                met += 1
        for pos, elems in self._forbidden_elements.items():
            items += 1
            if all(e not in sets[pos - 1] for e in elems):
                met += 1
        extra_score = 0.30 if items == 0 else 0.30 * (met / items)

        total = format_score + overlap_score + extra_score
        return max(-1.0, min(1.0, total))
