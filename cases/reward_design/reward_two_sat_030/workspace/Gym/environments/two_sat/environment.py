# TwoSAT verifiable environment for RLVE reward design.
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Tuple

from Gym.environment import VerifiableEnvironment, manual_seed


class TwoSAT_Environment(VerifiableEnvironment):
    prompt_template = (
        'There are {N} boolean (0/1) values x[0], x[1], ..., x[{N_minus_1}]. '
        'Each of the following {M} expressions (`|` means OR, `!` means NOT) must equal 1:\n'
        '{expressions}\n\n'
        'Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies the conditions above.\n\n'
        'Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], '
        'separated by **spaces**.\n'
        'Example: `{N_boolean}` (do **NOT** include quotes or backticks).'
    )

    # Fallback generation bounds
    MIN_N = 5
    MAX_N = 20
    MIN_CLAUSES_FACTOR = 1
    MAX_CLAUSES_FACTOR = 4

    MAX_ALLOWED_N = 50
    MAX_ALLOWED_M = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N: int = 0
        self.clauses: List[Tuple[int, int]] = []
        self.solution: List[int] = []

    # ------------------------------------------------------------------
    # Helpers: load/save problem data from/to self.parameter
    # ------------------------------------------------------------------
    def _save_problem(self) -> None:
        """Store problem data in self.parameter so get_config/set_config preserves it."""
        self.parameter['_N'] = self.N
        self.parameter['_clauses'] = self.clauses
        self.parameter['_solution'] = self.solution

    def _ensure_loaded(self) -> None:
        """If problem data not in memory, load from self.parameter (set by set_config)."""
        if self.N == 0 and '_N' in self.parameter:
            self.N = self.parameter['_N']
            self.clauses = self.parameter['_clauses']
            self.solution = self.parameter['_solution']

    def set_config(self, config: dict) -> None:
        super().set_config(config)
        # Restore problem from parameter if present
        self._ensure_loaded()

    # ------------------------------------------------------------------
    # 2-SAT solver
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_2sat(N: int, clauses: List[Tuple[int, int]]) -> Optional[List[int]]:
        adj = [[] for _ in range(2 * N)]

        for a, b in clauses:
            def lit_node(lit):
                if lit > 0:
                    return 2 * (lit - 1)
                else:
                    return 2 * (-lit - 1) + 1
            def neg_node(node):
                return node ^ 1
            u, v = lit_node(a), lit_node(b)
            adj[neg_node(u)].append(v)
            adj[neg_node(v)].append(u)

        visited = [False] * (2 * N)
        order = []
        def dfs(u):
            visited[u] = True
            for w in adj[u]:
                if not visited[w]:
                    dfs(w)
            order.append(u)
        for i in range(2 * N):
            if not visited[i]:
                dfs(i)

        radj = [[] for _ in range(2 * N)]
        for u in range(2 * N):
            for v in adj[u]:
                radj[v].append(u)

        comp = [-1] * (2 * N)
        cur = 0
        def rdfs(u):
            comp[u] = cur
            for w in radj[u]:
                if comp[w] == -1:
                    rdfs(w)
        for u in reversed(order):
            if comp[u] == -1:
                rdfs(u)
                cur += 1

        assignment = [0] * N
        for i in range(N):
            if comp[2 * i] == comp[2 * i + 1]:
                return None
            assignment[i] = 1 if comp[2 * i] > comp[2 * i + 1] else 0
        return assignment

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        param_N = self.parameter.get('N', None)
        param_M = self.parameter.get('M', None)

        if param_N is not None and param_M is not None:
            N = int(param_N)
            M = int(param_M)
            N = max(2, min(N, self.MAX_ALLOWED_N))
            M = max(1, min(M, self.MAX_ALLOWED_M))
        elif param_N is not None:
            N = int(param_N)
            N = max(2, min(N, self.MAX_ALLOWED_N))
            M = random.randint(max(N, self.MIN_CLAUSES_FACTOR * N),
                               max(N, self.MAX_CLAUSES_FACTOR * N))
        else:
            N = random.randint(self.MIN_N, self.MAX_N)
            M = random.randint(max(N, self.MIN_CLAUSES_FACTOR * N),
                               max(N, self.MAX_CLAUSES_FACTOR * N))

        max_attempts = 1000
        for _ in range(max_attempts):
            clauses = []
            for _ in range(M):
                v1 = random.randint(1, N)
                v2 = random.randint(1, N)
                while v2 == v1:
                    v2 = random.randint(1, N)
                if random.random() < 0.5:
                    v1 = -v1
                if random.random() < 0.5:
                    v2 = -v2
                clauses.append((v1, v2))
            solution = self._solve_2sat(N, clauses)
            if solution is not None:
                self.N = N
                self.clauses = clauses
                self.solution = solution
                self._save_problem()
                return

        # Fallback: reduce clauses until satisfiable
        while M > N:
            M = max(N, M // 2)
            for _ in range(max_attempts):
                clauses = []
                for _ in range(M):
                    v1 = random.randint(1, N)
                    v2 = random.randint(1, N)
                    while v2 == v1:
                        v2 = random.randint(1, N)
                    if random.random() < 0.5:
                        v1 = -v1
                    if random.random() < 0.5:
                        v2 = -v2
                    clauses.append((v1, v2))
                solution = self._solve_2sat(N, clauses)
                if solution is not None:
                    self.N = N
                    self.clauses = clauses
                    self.solution = solution
                    self._save_problem()
                    return

        # Absolute fallback
        self.N = max(2, N)
        self.clauses = [(1, 2)]
        self.solution = [1] * self.N
        self._save_problem()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        expressions_lines = []
        for i, (a, b) in enumerate(self.clauses):
            def lit_str(lit):
                if lit > 0:
                    return f"x[{lit - 1}]"
                else:
                    return f"!x[{-lit - 1}]"
            expressions_lines.append(f"  {i + 1}. {lit_str(a)} | {lit_str(b)} = 1")

        expressions = "\n".join(expressions_lines)
        example = " ".join(str(i % 2) for i in range(self.N))

        return self.prompt_template.format(
            N=self.N,
            N_minus_1=self.N - 1,
            M=len(self.clauses),
            expressions=expressions,
            N_boolean=example,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        self._ensure_loaded()
        if answer is None or self.N == 0:
            return None

        lines = answer.strip().split('\n')
        candidates = []

        for line in reversed(lines):
            numbers = re.findall(r'\b([01])\b', line)
            if len(numbers) == self.N:
                candidates.append([int(x) for x in numbers])

        if candidates:
            return candidates[0]

        all_numbers = re.findall(r'\b([01])\b', answer)
        if len(all_numbers) >= self.N:
            return [int(x) for x in all_numbers[:self.N]]

        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        self._ensure_loaded()
        assignment = self.processor(output)

        if assignment is None:
            return -1.0

        satisfied = 0
        for a, b in self.clauses:
            def eval_lit(lit):
                if lit > 0:
                    return assignment[lit - 1]
                else:
                    return 1 - assignment[-lit - 1]
            if eval_lit(a) or eval_lit(b):
                satisfied += 1

        total = len(self.clauses)
        fraction = satisfied / total if total > 0 else 1.0
        score = 2.0 * fraction - 1.0
        return max(-1.0, min(1.0, score))
