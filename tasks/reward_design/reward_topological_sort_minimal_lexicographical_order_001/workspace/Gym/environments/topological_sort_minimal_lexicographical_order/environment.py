# TopologicalSort_MinimalLexicographicalOrder — RLVE verifiable environment.
# scorer(output) -> float in [-1.0, +1.0].

import heapq
import random
import re
from typing import Optional, List, Dict, Tuple

from Gym.environment import VerifiableEnvironment


class TopologicalSort_MinimalLexicographicalOrder_Environment(VerifiableEnvironment):
    prompt_template = (
        'Please find a permutation of `0` to `{N_minus_1}` ({N} integers in total) '
        'such that the following conditions are satisfied:\n'
        '{before_conditions}\n\n'
        'If multiple permutations satisfy the conditions, choose the one where:\n'
        '(1) `0` should appear as early as possible;\n'
        '(2) Subject to that, `1` should appear as early as possible;\n'
        '(3) Subject to that, `2` should appear as early as possible;\n'
        '(4) And so on...\n\n'
        '**Output Format:** Your final answer should be a single line containing '
        'the permutation `p(0), p(1), ..., p({N_minus_1})`, separated by spaces.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N: int = 0
        self.edges: List[Tuple[int, int]] = []
        self.optimal_order: List[int] = []
        self.adj: Dict[int, List[int]] = {}
        self.in_degree: Dict[int, int] = {}
        self._optimal_pos: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Kahn's algorithm with min-heap → lexicographically minimal topo    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lexicographically_minimal_topo(
        N: int, adj: Dict[int, List[int]], in_degree: Dict[int, int]
    ) -> List[int]:
        indeg = dict(in_degree)
        heap = [v for v in range(N) if indeg.get(v, 0) == 0]
        heapq.heapify(heap)
        order: List[int] = []
        while heap:
            u = heapq.heappop(heap)
            order.append(u)
            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, v)
        return order

    # ------------------------------------------------------------------ #
    #  _generate                                                         #
    # ------------------------------------------------------------------ #
    def _generate(self) -> None:
        N: int = self.parameter.get("N", 3)
        assert isinstance(N, int) and N >= 2, f"N must be int >= 2, got {N}"
        self.N = N

        # Build a random DAG via a secret permutation.
        secret_order = list(range(N))
        random.shuffle(secret_order)
        pos_in_secret = {v: idx for idx, v in enumerate(secret_order)}

        # Target ~1.5·N edges, probability bounded to [0.05, 0.5]
        max_possible = N * (N - 1) // 2
        target_edges = max(1, int(1.5 * N))
        edge_prob = (
            min(0.5, max(0.05, target_edges / max_possible))
            if max_possible > 0
            else 0.0
        )

        edges: List[Tuple[int, int]] = []
        adj: Dict[int, List[int]] = {v: [] for v in range(N)}
        in_degree: Dict[int, int] = {v: 0 for v in range(N)}

        for i in range(N):
            for j in range(i + 1, N):
                u, v = secret_order[i], secret_order[j]
                if random.random() < edge_prob:
                    edges.append((u, v))
                    adj[u].append(v)
                    in_degree[v] += 1

        # Guarantee at least one edge when N ≥ 2.
        if len(edges) == 0 and N >= 2:
            u, v = random.sample(range(N), 2)
            if pos_in_secret[u] > pos_in_secret[v]:
                u, v = v, u
            edges.append((u, v))
            adj[u].append(v)
            in_degree[v] += 1

        self.edges = edges
        self.adj = adj
        self.in_degree = in_degree

        # True lexicographically-minimal topological order.
        self.optimal_order = self._lexicographically_minimal_topo(N, adj, in_degree)
        self._optimal_pos = {v: idx for idx, v in enumerate(self.optimal_order)}

    # ------------------------------------------------------------------ #
    #  _prompt_generate                                                   #
    # ------------------------------------------------------------------ #
    def _prompt_generate(self) -> str:
        N = self.N
        N_minus_1 = N - 1

        if not self.edges:
            conditions_str = "  (None — there are no ordering constraints.)\n"
        else:
            lines = [f"  - `{u}` must appear before `{v}`" for u, v in self.edges]
            conditions_str = "\n".join(lines)

        return self.prompt_template.format(
            N=N,
            N_minus_1=N_minus_1,
            before_conditions=conditions_str,
        )

    # ------------------------------------------------------------------ #
    #  _extract_answer  (mirrors parent-class processor logic)            #
    # ------------------------------------------------------------------ #
    def _extract_answer(self, output: str) -> Optional[str]:
        """Extract answer text from the raw model output.

        Uses the same logic as VerifiableEnvironment.processor():
        1. Strip "Assistant:" / "<|im_start|>assistant" prefix.
        2. Find the last pair of answer markers.
        """
        if "Assistant:" in output:
            output = output.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in output:
            output = output.split("<|im_start|>assistant", 1)[1]

        answer_pattern = (
            re.escape(self.answer_markers[0])
            + r"(.*?)"
            + re.escape(self.answer_markers[1])
        )
        matches = list(re.finditer(answer_pattern, output, re.DOTALL))
        if matches:
            return matches[-1].group(1)
        return None

    # ------------------------------------------------------------------ #
    #  _process                                                           #
    # ------------------------------------------------------------------ #
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the answer string into a permutation (list of ints)."""
        if answer is None:
            return None

        text = answer.strip()
        # Remove common bracket characters.
        for ch in "[](){}":
            text = text.replace(ch, " ")
        tokens = re.split(r"[,\s]+", text.strip())

        numbers: List[int] = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            try:
                numbers.append(int(tok))
            except ValueError:
                return None

        if len(numbers) != self.N:
            return None
        if set(numbers) != set(range(self.N)):
            return None
        return numbers

    # ------------------------------------------------------------------ #
    #  scorer                                                             #
    # ------------------------------------------------------------------ #
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].

        -1.0 : not a valid permutation, or violates a "before" constraint.
        +1.0 : exactly the lexicographically minimal topological order.
        Intermediate values : linear in the first index k where the
        position of element k differs from the optimal (priority: 0,1,2,…).
        """
        answer_text = self._extract_answer(output)
        permutation = self._process(answer_text)

        if permutation is None:
            return -1.0

        # Check every "before" constraint.
        pos = {v: idx for idx, v in enumerate(permutation)}
        for u, v in self.edges:
            if pos[u] >= pos[v]:
                return -1.0

        # Valid — measure lexicographical distance.
        N = self.N
        first_diff = N
        for k in range(N):
            if pos[k] != self._optimal_pos[k]:
                first_diff = k
                break

        score = 2.0 * first_diff / N - 1.0
        return max(-1.0, min(1.0, score))
