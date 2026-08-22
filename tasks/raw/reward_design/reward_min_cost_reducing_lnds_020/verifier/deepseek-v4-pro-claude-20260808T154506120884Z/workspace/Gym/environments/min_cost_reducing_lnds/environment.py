# MinCostReducingLNDS — Minimum Cost to Reduce Longest Non-Decreasing Subsequence
#
# Problem: Given arrays A and B of length N, choose a set of indices to delete
# such that the length of the longest non-decreasing subsequence (LNDS) of A
# decreases by at least 1, while minimizing the sum of B[i] for deleted indices.
#
# Optimal solution: minimum vertex cut in the DAG of positions that belong to
# at least one maximum-length LNDS. Solved via max-flow (Dinic).

import random
import re
from collections import deque
from typing import Optional, List

from Gym.environment import VerifiableEnvironment


class Dinic:
    """Dinic max-flow for minimum s-t cut."""

    def __init__(self, n: int):
        self.n = n
        self.graph: List[List[int]] = [[] for _ in range(n)]
        self.edges: List[List[int]] = []  # [to, rev, cap]

    def add_edge(self, fr: int, to: int, cap: int):
        self.graph[fr].append(len(self.edges))
        self.edges.append([to, len(self.edges) + 1, cap])
        self.graph[to].append(len(self.edges))
        self.edges.append([fr, len(self.edges) - 1, 0])

    def _bfs(self, s: int, t: int, level: List[int]) -> bool:
        for i in range(self.n):
            level[i] = -1
        q = deque([s])
        level[s] = 0
        while q:
            v = q.popleft()
            for ei in self.graph[v]:
                to, rev, cap = self.edges[ei]
                if cap > 0 and level[to] < 0:
                    level[to] = level[v] + 1
                    q.append(to)
        return level[t] >= 0

    def _dfs(self, v: int, t: int, f: int, level: List[int], it: List[int]) -> int:
        if v == t:
            return f
        for i in range(it[v], len(self.graph[v])):
            it[v] = i
            ei = self.graph[v][i]
            to, rev, cap = self.edges[ei]
            if cap > 0 and level[v] + 1 == level[to]:
                ret = self._dfs(to, t, min(f, cap), level, it)
                if ret > 0:
                    self.edges[ei][2] -= ret
                    self.edges[rev][2] += ret
                    return ret
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        level = [0] * self.n
        INF = 10 ** 18
        while self._bfs(s, t, level):
            it = [0] * self.n
            while True:
                f = self._dfs(s, t, INF, level, it)
                if f == 0:
                    break
                flow += f
        return flow


def _compute_lns_length(arr: List[int]) -> int:
    """Return length of longest non-decreasing subsequence (O(N log N))."""
    if not arr:
        return 0
    tails = []
    for x in arr:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails)


def _compute_dp_end(arr: List[int]) -> List[int]:
    """dp_end[i] = length of LNDS ending at i (inclusive). O(N^2)."""
    n = len(arr)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
    return dp


def _compute_dp_start(arr: List[int]) -> List[int]:
    """dp_start[i] = length of LNDS starting at i (going forward). O(N^2)."""
    n = len(arr)
    dp = [1] * n
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if arr[i] <= arr[j] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
    return dp


def _min_cost_to_reduce_lns(A: List[int], B: List[int]) -> int:
    """
    Compute minimum total cost to reduce LNDS length by at least 1.

    Uses minimum vertex cut in the DAG of positions that belong to at least one
    maximum-length LNDS.  Each vertex v is split into v_in → v_out with capacity
    B[v]; edges of the LNDS DAG connect v_out → w_in with infinite capacity.
    Super-source connects to all layer-1 vertices; all layer-L vertices connect
    to super-sink.  The min s-t cut value equals the minimum deletion cost.

    Returns the optimal cost (integer).
    """
    n = len(A)
    if n == 0:
        return 0

    dp_end = _compute_dp_end(A)
    L = max(dp_end)

    if L <= 1:
        # Every single element is an LNDS of length 1.  To reach length 0 we
        # must delete *all* elements.
        return sum(B)

    dp_start = _compute_dp_start(A)

    # Vertices that belong to at least one maximum-length LNDS
    critical = [i for i in range(n) if dp_end[i] + dp_start[i] - 1 == L]
    if not critical:
        return 0

    # Zero-cost vertices make the answer trivially 0 (delete them for free).
    if any(B[i] == 0 for i in critical):
        return 0

    # Map original index → contiguous node id inside the flow graph
    idx_to_node = {orig: node_id for node_id, orig in enumerate(critical)}
    m = len(critical)

    # Node numbering:
    #   0           = super-source S
    #   1+2*k       = k-th critical vertex  in-node
    #   1+2*k+1     = k-th critical vertex out-node
    #   T = 1+2*m   = super-sink
    S = 0
    T = 1 + 2 * m
    total_nodes = T + 1
    dinic = Dinic(total_nodes)
    INF = 10 ** 15

    for node_id, orig in enumerate(critical):
        i_in = 1 + 2 * node_id
        i_out = 1 + 2 * node_id + 1
        dinic.add_edge(i_in, i_out, B[orig])

        if dp_end[orig] == 1:
            dinic.add_edge(S, i_in, INF)
        if dp_end[orig] == L:
            dinic.add_edge(i_out, T, INF)

    # Add DAG edges: i → j  iff  i < j, A[i] ≤ A[j], dp_end[i]+1 == dp_end[j]
    for ni, orig_i in enumerate(critical):
        i_out = 1 + 2 * ni + 1
        for nj, orig_j in enumerate(critical):
            if ni == nj:
                continue
            if (orig_i < orig_j and A[orig_i] <= A[orig_j]
                    and dp_end[orig_i] + 1 == dp_end[orig_j]):
                j_in = 1 + 2 * nj
                dinic.add_edge(i_out, j_in, INF)

    return dinic.max_flow(S, T)


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------

class MinCostReducingLNDS_Environment(VerifiableEnvironment):
    prompt_template = (
        'You are given two arrays A and B, both of length {N}:\n'
        'A: {A}\n'
        'B: {B}\n'
        'You may erase any (distinct) elements from A. '
        'When you erase element A[i], you must pay a cost of B[i]. '
        'Please reduce the length of the **longest non-decreasing subsequence** '
        '(not necessarily contiguous) of A by **at least 1**, while minimizing '
        'the total cost of the erased elements.\n'
        '**Output Format:** Output a single line containing the **indices** of '
        'the elements you choose to erase, separated by spaces. If you choose to '
        'erase no elements, output an empty line.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A: List[int] = []
        self.B: List[int] = []
        self.N: int = 0
        self._optimal_cost: int = 0
        self._original_lns: int = 0

    # ------------------------------------------------------------------
    #  Generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Generate a random instance using self.seed and self.parameter."""
        N = self.parameter.get("N", 5)
        self.N = N

        # A values in [1, max(5, N)] — keep them small enough that LNDS
        # structure is non-trivial but not too degenerate.
        max_val = max(5, N)
        self.A = [random.randint(1, max_val) for _ in range(N)]

        # B values in [1, 100] — positive costs.
        self.B = [random.randint(1, 100) for _ in range(N)]

        self._original_lns = _compute_lns_length(self.A)
        self._optimal_cost = _min_cost_to_reduce_lns(self.A, self.B)

    def _prompt_generate(self) -> str:
        A_str = "[" + ", ".join(str(x) for x in self.A) + "]"
        B_str = "[" + ", ".join(str(x) for x in self.B) + "]"
        return self.prompt_template.format(N=self.N, A=A_str, B=B_str)

    # ------------------------------------------------------------------
    #  Answer parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_indices(text: str) -> Optional[List[int]]:
        """Try to parse a string as a whitespace-separated list of integers."""
        text = text.strip()
        if text == "":
            return []
        parts = text.split()
        try:
            return [int(p) for p in parts]
        except (ValueError, TypeError):
            return None

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse answer extracted from <answer>...</answer> tags (if present)."""
        if answer is None:
            return None
        return self._parse_indices(answer)

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    def scorer(self, output: str) -> float:
        """
        Score the model's output.

        Strategy
        --------
        1. Try to extract the answer from ``<answer>…</answer>`` tags via the
           standard ``processor`` pipeline.
        2. If that yields nothing, fall back to scanning the raw output line-by-
           line (last line that looks like a whitespace-separated integer list).

        Once indices are obtained:
        * If parsing fails or indices are invalid (out-of-range, duplicates) → -1.0
        * If deleting those indices does **not** reduce LNDS length by ≥ 1 → -1.0
        * Otherwise:
            score = 2·(optimal_cost / actual_cost) – 1
          which yields +1.0 when actual = optimal and approaches -1.0 as the
          submitted cost grows without bound.
        """
        indices = self._extract_indices(output)
        if indices is None:
            return -1.0

        # --- validity checks ---
        if len(set(indices)) != len(indices):          # duplicates
            return -1.0
        for idx in indices:
            if not (0 <= idx < self.N):                # out of range
                return -1.0

        # --- does it actually reduce the LNDS? ---
        delete_set = set(indices)
        remaining_A = [self.A[i] for i in range(self.N) if i not in delete_set]
        new_lns = _compute_lns_length(remaining_A)
        if new_lns >= self._original_lns:
            return -1.0

        actual_cost = sum(self.B[i] for i in indices)

        # --- score from cost ratio ---
        optimal = self._optimal_cost
        if optimal == 0:
            return 1.0 if actual_cost == 0 else -1.0

        ratio = optimal / actual_cost          # in (0, 1]
        score = 2.0 * ratio - 1.0              # in (-1, +1]
        return max(-1.0, min(1.0, score))

    def _extract_indices(self, output: str) -> Optional[List[int]]:
        """
        Extract the list of indices from the model's raw output.

        First tries the standard ``processor`` (which looks for
        ``<answer>…</answer>`` tags).  If that returns ``None``, falls back to
        a heuristic scan of the raw text.
        """
        # 1 — try the tagged answer path
        processed = self.processor(output)
        if processed is not None and isinstance(processed, list):
            return processed

        # 2 — fallback: scan lines for index-looking patterns
        text = output
        # Strip "Assistant:" prefix if present (same logic as base class)
        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant", 1)[1]

        lines = text.splitlines()
        # Search backwards — last such line is most likely the answer
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # Strategy A: whole line is only digits and whitespace
            if re.fullmatch(r'[0-9\s]+', line):
                parsed = self._parse_indices(line)
                if parsed is not None:
                    return parsed

            # Strategy B: the line ends with a sequence of space-separated
            # digits (e.g. "The answer is: 0 1 2" or "indices: 3, 4, 5")
            m = re.search(r'(?:^|\s)((?:\d+\s+)*\d+)\s*$', line)
            if m:
                parsed = self._parse_indices(m.group(1))
                if parsed is not None:
                    return parsed

            # Strategy C: strip trailing non-digit suffix
            # (e.g. "1 2 3." → "1 2 3")
            cleaned = re.sub(r'[^0-9\s]+$', '', line).strip()
            if cleaned and re.fullmatch(r'[0-9\s]+', cleaned):
                parsed = self._parse_indices(cleaned)
                if parsed is not None:
                    return parsed

        return None
