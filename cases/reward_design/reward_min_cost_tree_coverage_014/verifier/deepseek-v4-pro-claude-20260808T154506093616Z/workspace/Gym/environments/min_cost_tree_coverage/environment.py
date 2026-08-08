# MinCostTreeCoverage — RLVE verifiable training environment.
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List

from Gym.environment import VerifiableEnvironment


class MinCostTreeCoverage_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. '
        'The tree contains {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v`**:\n'
        '{edges}\n\n'
        'You may select any subset of vertices. When a vertex `u` is selected, it **covers** all vertices that are reachable from `u` by a path containing at most {D} edges '
        '(i.e., within distance \u2264 {D} in terms of edge count). You are required to cover the following vertices: {covered_vertices}\n'
        'Each selected vertex `u` incurs a cost of `W[u]`. The cost array is: {W}\n'
        'Try your best to **minimize the total cost** of the selected vertices while ensuring all required vertices are covered.\n\n'
        '**Output Format:** A single line containing the selected vertex indices in any order, separated by **spaces**.'
    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_adj(N: int, edges: List[List[int]]) -> List[List[int]]:
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    @staticmethod
    def _bfs_distances(N: int, adj: List[List[int]]) -> List[List[int]]:
        """All-pairs shortest-path distances on a tree (unweighted edges)."""
        dist = [[0] * N for _ in range(N)]
        for src in range(N):
            d = [-1] * N
            d[src] = 0
            q = [src]
            for u in q:
                for v in adj[u]:
                    if d[v] == -1:
                        d[v] = d[u] + 1
                        q.append(v)
            dist[src] = d
        return dist

    @staticmethod
    def _compute_optimal_cost(
        N: int,
        dist: List[List[int]],
        D: int,
        W: List[int],
        covered_vertices: List[int],
    ) -> int:
        """Brute-force optimal cost via 2^N subset enumeration. Returns minimal total cost."""
        best = float('inf')
        covered_set = set(covered_vertices)
        for mask in range(1 << N):
            cost = 0
            covered = set()
            for v in range(N):
                if mask & (1 << v):
                    cost += W[v]
                    for u in range(N):
                        if dist[v][u] <= D:
                            covered.add(u)
            if covered_set.issubset(covered):
                if cost < best:
                    best = cost
        return int(best)

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """
        Sample a random tree, vertex costs, distance D, and target vertex set.
        Store everything in self.parameter so it is reproducible via self.seed.
        """
        # Use parameter dict if provided (e.g. from ParameterController), else sample.
        param_N = self.parameter.get('N', None) if self.parameter else None

        if param_N is not None:
            N = int(param_N)
            N = max(4, min(N, 14))  # keep within reasonable bounds
        else:
            N = random.randint(6, 10)

        # Build a random labelled tree: for i = 1..N-1 connect i to random j in [0, i-1].
        # This produces a uniformly random labelled tree (Cayley's formula).
        edges = []
        for i in range(1, N):
            j = random.randint(0, i - 1)
            edges.append([j, i])

        # Shuffle edge order for presentation variety
        random.shuffle(edges)

        # Distance threshold D
        D = random.randint(1, min(3, N - 1))

        # Vertex costs: random integers in [1, 10]
        W = [random.randint(1, 10) for _ in range(N)]

        # Which vertices need to be covered: either all, or a random subset of size >= ceil(N/2)
        if random.random() < 0.5:
            covered_vertices = list(range(N))
        else:
            k = random.randint(max(2, (N + 1) // 2), N)
            covered_vertices = sorted(random.sample(range(N), k))

        # Compute optimal cost via brute force
        adj = self._build_adj(N, edges)
        dist = self._bfs_distances(N, adj)
        optimal_cost = self._compute_optimal_cost(N, dist, D, W, covered_vertices)

        self.parameter = {
            'N': N,
            'edges': edges,
            'D': D,
            'W': W,
            'covered_vertices': covered_vertices,
            'optimal_cost': optimal_cost,
            'dist': dist,
        }

    def _prompt_generate(self) -> str:
        p = self.parameter
        N = p['N']
        edges_str = '\n'.join(f'({u}, {v})' for u, v in p['edges'])
        covered_str = ', '.join(str(v) for v in p['covered_vertices'])
        W_str = str(p['W'])
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges=edges_str,
            D=p['D'],
            covered_vertices=covered_str,
            W=W_str,
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """
        Parse the extracted answer (content between answer markers) into a list of
        selected vertex indices. Returns None if unparseable.
        """
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:
            return None

        # Extract all integer tokens: split on any non-digit runs (handles commas,
        # semicolons, and other separators gracefully).
        tokens = re.findall(r'-?\d+', answer)
        if not tokens:
            return None

        try:
            selected = [int(t) for t in tokens]
        except (ValueError, TypeError):
            return None

        return selected

    def scorer(self, output: str) -> float:
        """
        Score in [-1.0, +1.0].

        Scoring regime:
        - Cannot parse or invalid vertices  ->  -1.0
        - Does not cover all required vertices:
            score = -1.0 + 0.5 * coverage_ratio   (range: [-1.0, -0.5))
        - Covers all required vertices:
            score = 1.0 - (cost - opt) / opt
            clipped to [-0.5, +1.0]
            (optimal = +1.0; cost = 2*opt -> 0.0; worse -> -0.5 floor)
        """
        p = self.parameter
        N = p['N']
        D = p['D']
        W = p['W']
        covered_vertices = p['covered_vertices']
        opt = p['optimal_cost']
        dist = p['dist']

        selected = self.processor(output)

        # --- parsing / validity failures ---
        if selected is None:
            return -1.0

        # Check for out-of-range vertex indices
        for v in selected:
            if v < 0 or v >= N:
                return -1.0

        # Deduplicate — selecting a vertex multiple times costs the same as once.
        selected_set = set(selected)
        cost = sum(W[v] for v in selected_set)

        # --- coverage check ---
        covered: set = set()
        for v in selected_set:
            for u in range(N):
                if dist[v][u] <= D:
                    covered.add(u)

        required_set = set(covered_vertices)
        covered_required = covered & required_set
        coverage_ratio = len(covered_required) / len(required_set) if required_set else 1.0

        if coverage_ratio < 1.0:
            # Partial coverage: score in [-1.0, -0.5)
            return -1.0 + 0.5 * coverage_ratio

        # --- full coverage: score based on cost relative to optimal ---
        if opt <= 0:
            opt = 1  # safeguard; should never trigger with positive costs

        # Linear: optimal -> +1.0, 2*opt -> 0.0, 3*opt -> -1.0 (but floor at -0.5)
        raw = 1.0 - (cost - opt) / float(opt)
        return max(-0.5, min(1.0, raw))
