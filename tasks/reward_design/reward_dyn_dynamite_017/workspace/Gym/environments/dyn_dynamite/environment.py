# DynDynamite: tree k-center with key vertices.
# Minimize max distance from any key vertex to the nearest selected center.
# scorer(output) -> float in [-1.0, +1.0].

import math
import random
import re
from typing import Optional, List, Set, Tuple

from Gym.environment import VerifiableEnvironment


class DynDynamite_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given a **tree** (i.e., a connected undirected graph with no cycles) '
        'with {N} vertices, labeled from `0` to `{N_minus_1}`. '
        'It contains the following {N_minus_1} undirected edges. '
        'Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge '
        '**connecting vertex `u` to vertex `v`**:\n'
        '{edges}\n\n'
        'You are also given a list of key vertices: {key_vertices}\n'
        'Please select exactly {M} vertices (from all {N} vertices) to serve as **centers**. '
        'Your goal is to **minimize the maximum distance** (measured in number of edges) '
        'from any key vertex to its nearest selected center.\n'
        'Output format: A single line containing the {M} selected centers, separated by spaces.'
    )

    # ------------------------------------------------------------------
    # Helpers: tree generation, optimal-distance solver
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_random_tree(N: int, rng: random.Random) -> List[Tuple[int, int]]:
        """Return N-1 edges forming a tree on vertices 0..N-1.

        Uses the simple method: for i in 1..N-1, pick parent uniformly
        from {0, …, i-1}.  This yields a labelled recursive tree (not
        uniformly random among all N^(N-2) trees, but diverse enough for
        training).
        """
        edges: List[Tuple[int, int]] = []
        for i in range(1, N):
            parent = rng.randint(0, i - 1)
            edges.append((i, parent))
        return edges

    @staticmethod
    def _build_adj(N: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    @staticmethod
    def _bfs_distances(
        N: int, adj: List[List[int]], sources: List[int]
    ) -> List[int]:
        """Multi-source BFS returning shortest distance to any source."""
        INF = 10 ** 9
        dist = [INF] * N
        from collections import deque

        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    @staticmethod
    def _can_cover(
        N: int,
        adj: List[List[int]],
        key_set: Set[int],
        D: int,
        M: int,
    ) -> bool:
        """Return True if M centers suffice to cover all key vertices within distance D.

        Classic greedy DP on tree (BOI 2003 "Dynamite" style).
        """
        if D < 0:
            return False
        if M >= len(key_set):
            return True  # put a center on every key vertex

        NEG_INF = -10 ** 9
        POS_INF = 10 ** 9

        # Root the tree at 0, compute parent and post-order.
        parent = [-1] * N
        order: List[int] = []
        stack = [0]
        parent[0] = -2  # mark visited
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    stack.append(v)

        need = [NEG_INF] * N  # farthest uncovered key distance in subtree
        have = [POS_INF] * N  # nearest center distance in subtree
        centers_used = 0

        for u in reversed(order):
            nd = NEG_INF
            hv = POS_INF
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if need[v] != NEG_INF:
                    nd = max(nd, need[v] + 1)
                if have[v] != POS_INF:
                    hv = min(hv, have[v] + 1)

            if u in key_set:
                nd = max(nd, 0)

            if nd != NEG_INF and hv != POS_INF and nd + hv <= D:
                nd = NEG_INF

            if nd == D:
                centers_used += 1
                nd = NEG_INF
                hv = 0

            need[u] = nd
            have[u] = hv

        if need[0] != NEG_INF:
            centers_used += 1

        return centers_used <= M

    @staticmethod
    def _optimal_max_distance(
        N: int,
        adj: List[List[int]],
        key_set: Set[int],
        M: int,
    ) -> int:
        """Binary search for the minimum feasible D."""
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            if DynDynamite_Environment._can_cover(N, adj, key_set, mid, M):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ----- generation -------------------------------------------------

    def _generate(self) -> None:
        """Create a random tree, select key vertices and M, and store everything.

        Uses ``self.seed`` (already set by the framework) for reproducibility.
        """
        N: int = self.parameter["N"]
        rng = random.Random(self.seed)

        # 1. Tree  (random recursive tree)
        edges = self._generate_random_tree(N, rng)
        adj = self._build_adj(N, edges)

        # 2. Key vertices: random count between N/4 and 3N/4
        min_key = max(1, N // 4)
        max_key = max(min_key, (3 * N) // 4)
        num_key = rng.randint(min_key, max_key)
        all_vertices = list(range(N))
        key_vertices = sorted(rng.sample(all_vertices, num_key))
        key_set = set(key_vertices)

        # 3. Number of centers M: random between 1 and N/3
        M = rng.randint(1, max(1, N // 3))

        # 4. Optimal answer
        optimal_D = self._optimal_max_distance(N, adj, key_set, M)

        # Store everything needed by later stages
        self.parameter.update({
            "N": N,
            "edges": edges,
            "adj": adj,
            "key_vertices": key_vertices,
            "key_set": key_set,
            "M": M,
            "optimal_D": optimal_D,
        })

    # ----- prompt -----------------------------------------------------

    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        edges = self.parameter["edges"]
        key_vertices = self.parameter["key_vertices"]
        M = self.parameter["M"]

        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)

        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges=edges_str,
            key_vertices=key_vertices,
            M=M,
        )

    # ----- processing -------------------------------------------------

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Extract M integer vertices from the model answer.

        Returns a list of vertex indices, or None if parsing fails.
        """
        if answer is None:
            return None

        # Extract all integers from the answer string
        numbers = re.findall(r"-?\d+", answer)
        if not numbers:
            return None

        try:
            vertices = [int(x) for x in numbers]
        except ValueError:
            return None

        N = self.parameter["N"]
        M = self.parameter["M"]

        # Filter to valid vertex indices
        valid_verts = [v for v in vertices if 0 <= v < N]

        if len(valid_verts) < M:
            return None

        # Take the first M valid vertices
        return valid_verts[:M]

    # ----- scoring ----------------------------------------------------

    def scorer(self, output: str) -> float:
        """Compute reward ∈ [-1.0, +1.0] for the model output.

        Strategy:
          * Invalid / unparseable → -1.0
          * Valid selection → compute D_model (max key→center distance)
          * Compare to D_opt:
            - D_model == D_opt  → +1.0
            - D_opt == 0        → +1.0 if D_model==0 else -1.0
            - D_model >  D_opt  → partial: max(-1.0, 2.0 - D_model/D_opt)
        """
        parsed = self.processor(output)
        if parsed is None:
            return -1.0

        N = self.parameter["N"]
        adj = self.parameter["adj"]
        key_set = self.parameter["key_set"]
        optimal_D = self.parameter["optimal_D"]

        centers = parsed  # list of M valid vertex indices

        # Compute distances from selected centers to all vertices
        dist = self._bfs_distances(N, adj, centers)

        # Maximum distance to any key vertex
        D_model = max(dist[v] for v in key_set)

        if optimal_D == 0:
            return 1.0 if D_model == 0 else -1.0

        # Partial credit: linear in ratio D_model/D_opt
        ratio = D_model / optimal_D  # always ≥ 1.0
        score = 2.0 - ratio
        return max(-1.0, min(1.0, score))
