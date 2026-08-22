# scorer(output) -> float in [-1.0, +1.0].

import random
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from Gym.environment import VerifiableEnvironment, manual_seed

class BEZMinimalistSecurity_Environment(VerifiableEnvironment):
    prompt_template = (
        "There is an array P of length {N}. Initially, P is: {P}\n\n"
        "Now we want to construct a new array P' of length {N}, "
        "where 0 <= P'[i] <= P[i] for all i. "
        "Additionally, there are some constraints of the form "
        "P'[u] + P'[v] = w, where u and v are indices and w is a "
        "constant (it is guaranteed that P[u] + P[v] >= w). "
        "The constraints are:\n{constraints}\n\n"
        "Please output P'[0], P'[1], ..., P'[{N_minus_1}], "
        "separated by spaces, such that they satisfy all the "
        "constraints and their sum is {minimized_or_maximized}. "
        "Put your final answer inside {answer_marker_open}...{answer_marker_close}."
    )

    def __init__(self, **kwargs):
        # Default to \boxed{} markers, consistent with the default
        # --answer-marker-type used by rlve_rm when the preprocessor is
        # ChatTemplate_NoSystemPrompt (the training preset).
        if 'answer_markers' not in kwargs:
            kwargs['answer_markers'] = (r'\boxed{', r'}')
        super().__init__(**kwargs)
        self.N = None
        self.P = None
        self.constraints = None  # list of (u, v, w)
        self.minimize = None     # True -> minimize sum, False -> maximize
        self.optimal_sum = None  # best achievable sum
        self.worst_sum = None    # worst feasible sum (opposite extreme)

    # ------------------------------------------------------------------
    #  Override set_config to regenerate the problem deterministically
    # ------------------------------------------------------------------
    def set_config(self, config: Dict) -> None:
        """Restore problem state and regenerate the instance.

        The base class only restores seed + parameter, but the actual
        problem data (N, P, constraints, optimal_sum, etc.) must be
        regenerated via _generate for the scorer to work.
        """
        super().set_config(config)
        # Regenerate the exact same problem via the stored seed.
        manual_seed(self.seed)
        self._generate()

    # ------------------------------------------------------------------
    #  Generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        N = self.parameter['N']
        edge_ratio = self.parameter['edge_ratio']
        M_target = max(1, int(N * edge_ratio))
        max_edges = N * (N - 1) // 2
        M_target = min(M_target, max_edges)

        # Try a few times to get a non-degenerate instance
        for _attempt in range(10):
            # --- 1. initial bound array P ---
            P_init = [random.randint(5, 25) for _ in range(N)]

            # --- 2. build a random spanning forest ---
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(all_pairs)

            parent = list(range(N))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                x, y = find(x), find(y)
                if x != y:
                    parent[x] = y
                    return True
                return False

            forest_edges = []
            for u, v in all_pairs:
                if union(u, v):
                    forest_edges.append((u, v))

            # --- 3. ensure at least one odd-sized component ---
            isolated_node = None
            if N % 2 == 0:
                # compute component roots via path compression
                comp_of = list(range(N))
                for u, v in forest_edges:
                    # find root of u
                    ru = u
                    while comp_of[ru] != ru:
                        comp_of[ru] = comp_of[comp_of[ru]]
                        ru = comp_of[ru]
                    # find root of v
                    rv = v
                    while comp_of[rv] != rv:
                        comp_of[rv] = comp_of[comp_of[rv]]
                        rv = comp_of[rv]
                    if ru != rv:
                        comp_of[rv] = ru
                for i in range(N):
                    r = i
                    while comp_of[r] != r:
                        comp_of[r] = comp_of[comp_of[r]]
                        r = comp_of[r]
                    comp_of[i] = r

                comp_sizes: Dict[int, int] = {}
                for i in range(N):
                    r = comp_of[i]
                    comp_sizes[r] = comp_sizes.get(r, 0) + 1

                all_even = all(sz % 2 == 0 for sz in comp_sizes.values())
                if all_even:
                    isolated_node = random.randrange(N)
                    forest_edges = [(u, v) for (u, v) in forest_edges
                                    if u != isolated_node and v != isolated_node]

            # --- 4. build adjacency and bipartite colouring ---
            adj = [[] for _ in range(N)]
            for u, v in forest_edges:
                adj[u].append(v)
                adj[v].append(u)

            colour = [-1] * N
            for root in range(N):
                if colour[root] != -1:
                    continue
                colour[root] = 0
                stack = [root]
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if colour[v] == -1:
                            colour[v] = 1 - colour[u]
                            stack.append(v)

            # --- 5. assign a concrete feasible P' on the forest ---
            visited = [False] * N
            P_concrete = [0] * N

            for root in range(N):
                if visited[root]:
                    continue
                P_concrete[root] = random.randint(0, P_init[root])
                stack = [root]
                visited[root] = True
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if not visited[v]:
                            P_concrete[v] = random.randint(0, P_init[v])
                            visited[v] = True
                            stack.append(v)

            # --- 6. derive w for forest edges ---
            w_dict: Dict[Tuple[int, int], int] = {}
            for u, v in forest_edges:
                key = (min(u, v), max(u, v))
                w_dict[key] = P_concrete[u] + P_concrete[v]

            # --- 7. add extra edges preserving bipartiteness ---
            all_edges = list(forest_edges)
            for u, v in all_pairs:
                if len(all_edges) >= M_target:
                    break
                if isolated_node is not None and (u == isolated_node or v == isolated_node):
                    continue
                key = (min(u, v), max(u, v))
                if key not in w_dict and colour[u] != colour[v]:
                    all_edges.append((u, v))
                    w_dict[key] = P_concrete[u] + P_concrete[v]

            # --- 8. enlarge P for headroom ---
            P = [P_concrete[i] + random.randint(5, 25) for i in range(N)]

            # --- 9. build constraint list ---
            constraints = []
            for u, v in all_edges:
                key = (min(u, v), max(u, v))
                constraints.append((u, v, w_dict[key]))

            # --- 10. decide optimisation direction ---
            minimize = random.choice([True, False])

            # --- 11. solve for optimal and worst sums ---
            #       store temporarily so we can check the gap
            self.N = N
            self.P = P
            self.constraints = constraints
            self.minimize = minimize

            opt, worst = self._solve_extremes()
            if abs(opt - worst) >= 0.5:
                self.optimal_sum = opt
                self.worst_sum = worst
                return  # success

        # If we exhaust retries, keep the last attempt (degenerate but valid)
        self.optimal_sum = opt
        self.worst_sum = worst

    # ------------------------------------------------------------------
    #  Solver
    # ------------------------------------------------------------------
    def _solve_extremes(self) -> Tuple[float, float]:
        """Returns (optimal_sum, worst_sum).

        For bipartite graphs each component has one degree of freedom x.
        P'[i] = coeff[i]·x + const[i]  with coeff[i] ∈ {+1, -1}.
        The component sum is s·x + t; extremes lie at the bounds of
        the feasible interval for x.
        """
        N = self.N
        P = self.P
        constraints = self.constraints
        minimize = self.minimize

        adj = [[] for _ in range(N)]
        for u, v, w in constraints:
            adj[u].append((v, w))
            adj[v].append((u, w))

        visited: List[bool] = [False] * N
        total_optimal = 0.0
        total_worst = 0.0

        for root in range(N):
            if visited[root]:
                continue

            # ----------  BFS: assign expressions  ----------
            coeff: Dict[int, int] = {root: 1}
            const: Dict[int, int] = {root: 0}
            queue = [root]
            visited[root] = True
            fixed_x = None
            feasible = True

            while queue:
                u = queue.pop()
                for v, w in adj[u]:
                    new_coeff = -coeff[u]
                    new_const = w - const[u]

                    if v not in coeff:
                        coeff[v] = new_coeff
                        const[v] = new_const
                        visited[v] = True
                        queue.append(v)
                    else:
                        if coeff[v] == new_coeff:
                            if const[v] != new_const:
                                feasible = False
                        else:
                            cand = (new_const - const[v]) / (2.0 * coeff[v])
                            if fixed_x is None:
                                fixed_x = cand
                            elif abs(fixed_x - cand) > 1e-9:
                                feasible = False

            if not feasible:
                continue

            # ----------  feasible interval for x  ----------
            L = -float('inf')
            R = float('inf')
            for i in coeff:
                if coeff[i] == 1:
                    L = max(L, -const[i])
                    R = min(R, P[i] - const[i])
                else:
                    L = max(L, const[i] - P[i])
                    R = min(R, const[i])

            if L > R + 1e-9:
                continue

            s = sum(coeff.values())
            t = sum(const.values())

            if fixed_x is not None:
                if L - 1e-9 <= fixed_x <= R + 1e-9:
                    val = s * fixed_x + t
                    total_optimal += val
                    total_worst += val
            else:
                f_L = s * L + t
                f_R = s * R + t

                if minimize:
                    total_optimal += min(f_L, f_R)
                    total_worst += max(f_L, f_R)
                else:
                    total_optimal += max(f_L, f_R)
                    total_worst += min(f_L, f_R)

        return total_optimal, total_worst

    # ------------------------------------------------------------------
    #  Prompt generation
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        P_str = "[" + ", ".join(str(p) for p in self.P) + "]"

        lines = []
        for u, v, w in self.constraints:
            lines.append(f"P'[{u}] + P'[{v}] = {w}")
        constraints_str = "\n".join(lines)

        N_minus_1 = self.N - 1
        minimized_or_maximized = "minimized" if self.minimize else "maximized"

        return self.prompt_template.format(
            N=self.N,
            P=P_str,
            constraints=constraints_str,
            N_minus_1=N_minus_1,
            minimized_or_maximized=minimized_or_maximized,
            answer_marker_open=self.answer_markers[0],
            answer_marker_close=self.answer_markers[1],
        )

    # ------------------------------------------------------------------
    #  Answer processing
    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        if answer is None:
            return None
        try:
            parts = answer.strip().split()
            if len(parts) != self.N:
                return None
            values = [int(p) for p in parts]
            return values
        except (ValueError, AttributeError):
            return None

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        processed = self.processor(output)
        if processed is None:
            return -1.0

        P_prime = processed

        # --- bound checks ---
        for i in range(self.N):
            if not (0 <= P_prime[i] <= self.P[i]):
                return -1.0

        # --- constraint checks ---
        for u, v, w in self.constraints:
            if P_prime[u] + P_prime[v] != w:
                return -1.0

        actual_sum = float(sum(P_prime))

        # --- degenerate case: unique solution ---
        if abs(self.optimal_sum - self.worst_sum) < 1e-9:
            if abs(actual_sum - self.optimal_sum) < 1e-9:
                return 1.0
            return -1.0

        # --- score: linear between optimal (+1) and worst (-1) ---
        if self.minimize:
            if actual_sum <= self.optimal_sum + 1e-9:
                return 1.0
            raw = 1.0 - 2.0 * (actual_sum - self.optimal_sum) / (self.worst_sum - self.optimal_sum)
        else:
            if actual_sum >= self.optimal_sum - 1e-9:
                return 1.0
            raw = 1.0 - 2.0 * (self.optimal_sum - actual_sum) / (self.optimal_sum - self.worst_sum)

        return max(-1.0, min(1.0, raw))
