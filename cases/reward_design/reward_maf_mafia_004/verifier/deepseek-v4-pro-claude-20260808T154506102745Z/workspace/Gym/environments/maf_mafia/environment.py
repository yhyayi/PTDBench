# scorer(output) -> float in [-1.0, +1.0].

import random
from typing import Optional, List, Dict, Tuple
from Gym.environment import VerifiableEnvironment


def simulate(N: int, TO: List[int], perm: List[int]) -> int:
    """Simulate the game given a permutation and return number killed."""
    alive = [True] * N
    killed_count = 0
    for p in perm:
        if alive[p]:
            target = TO[p]
            if alive[target]:
                alive[target] = False
                killed_count += 1
    return killed_count


def brute_force_optimal(N: int, TO: List[int], minimize: bool) -> int:
    """Compute optimal kill count by enumerating all permutations."""
    import itertools
    best = None
    for perm in itertools.permutations(range(N)):
        k = simulate(N, TO, list(perm))
        if best is None:
            best = k
        elif minimize:
            best = min(best, k)
        else:
            best = max(best, k)
    return best


# ---------- MIN KILLS: DP on functional graph ----------

def compute_tree_dp_min(N: int, TO: List[int], cycle_nodes: set) -> Tuple[List[float], List[float], List[float]]:
    INF = float('inf')
    
    children = [[] for _ in range(N)]
    for u in range(N):
        v = TO[u]
        children[v].append(u)
    
    visited = [False] * N
    order = []
    
    def dfs(u):
        visited[u] = True
        for c in children[u]:
            if not visited[c] and c not in cycle_nodes:
                dfs(c)
        if u not in cycle_nodes:
            order.append(u)
    
    for u in range(N):
        if u not in cycle_nodes and not visited[u]:
            dfs(u)
    
    surv = [0.0] * N
    die = [INF] * N
    die_ext = [0.0] * N
    
    for v in order:
        child_list = [c for c in children[v] if c not in cycle_nodes]
        
        if not child_list:
            surv[v] = 0.0
            die[v] = INF
            die_ext[v] = 0.0
        else:
            total = 0.0
            possible = True
            for c in child_list:
                if die[c] == INF:
                    possible = False
                    break
                total += die[c]
            surv[v] = total if possible else INF
            
            total_ext = 0.0
            for c in child_list:
                total_ext += min(surv[c], die[c])
            die_ext[v] = total_ext
            
            best_sum = 0.0
            sacrifices = []
            for c in child_list:
                best_c = min(surv[c], die[c])
                best_sum += best_c
                sacrifices.append(surv[c] - best_c)
            
            min_sacrifice = min(sacrifices) if sacrifices else INF
            if min_sacrifice == INF:
                die[v] = INF
            else:
                die[v] = 1.0 + best_sum + min_sacrifice
    
    for v in cycle_nodes:
        child_list = [c for c in children[v] if c not in cycle_nodes]
        
        if not child_list:
            surv[v] = 0.0
            die[v] = INF
            die_ext[v] = 0.0
        else:
            total = 0.0
            possible = True
            for c in child_list:
                if die[c] == INF:
                    possible = False
                    break
                total += die[c]
            surv[v] = total if possible else INF
            
            total_ext = 0.0
            for c in child_list:
                total_ext += min(surv[c], die[c])
            die_ext[v] = total_ext
            
            best_sum = 0.0
            sacrifices = []
            for c in child_list:
                best_c = min(surv[c], die[c])
                best_sum += best_c
                sacrifices.append(surv[c] - best_c)
            min_sacrifice = min(sacrifices)
            die[v] = 1.0 + best_sum + min_sacrifice
    
    return surv, die, die_ext


def solve_cycle_dp_min(L: int, tree_surv: List[float], tree_die: List[float], tree_die_ext: List[float]) -> float:
    INF = float('inf')
    
    if L == 0:
        return 0.0
    
    if L == 1:
        v = 0
        best = INF
        if tree_die[v] != INF:
            best = tree_die[v]
        self_kill_cost = tree_die_ext[v] + 1.0
        if best == INF:
            best = self_kill_cost
        else:
            best = min(best, self_kill_cost)
        return best
    
    def run_linear_dp(first_state, first_cost, L):
        dp = [{'S': INF, 'D': INF} for _ in range(L)]
        dp[0][first_state] = first_cost
        
        for i in range(1, L):
            if tree_surv[i] != INF and dp[i-1]['D'] != INF:
                dp[i]['S'] = dp[i-1]['D'] + tree_surv[i]
            
            if dp[i-1]['S'] != INF:
                cost_pred = dp[i-1]['S'] + tree_die_ext[i] + 1.0
                if cost_pred < dp[i]['D']:
                    dp[i]['D'] = cost_pred
            if dp[i-1]['D'] != INF and tree_die[i] != INF:
                cost_tree = dp[i-1]['D'] + tree_die[i]
                if cost_tree < dp[i]['D']:
                    dp[i]['D'] = cost_tree
        
        return dp
    
    best_overall = INF
    
    # Case 1: node 0 = S => L-1 must be D
    if tree_surv[0] != INF:
        dp = run_linear_dp('S', tree_surv[0], L)
        if dp[L-1]['D'] != INF:
            best_overall = min(best_overall, dp[L-1]['D'])
    
    # Case 2: node 0 = D (killed by tree) => L-1 must be D
    if tree_die[0] != INF:
        dp = run_linear_dp('D', tree_die[0], L)
        if dp[L-1]['D'] != INF:
            best_overall = min(best_overall, dp[L-1]['D'])
    
    # Case 3: node 0 = D (killed by L-1) => L-1 must be S; +1 added at wrap
    dp = run_linear_dp('D', tree_die_ext[0], L)
    if dp[L-1]['S'] != INF:
        best_overall = min(best_overall, dp[L-1]['S'] + 1.0)
    
    return best_overall


def compute_min_kills(N: int, TO: List[int]) -> float:
    """Compute minimum kills using DP on functional graph."""
    state = [0] * N
    cycle_nodes = set()
    cycles = []
    
    for start in range(N):
        if state[start] != 0:
            continue
        path = []
        cur = start
        while state[cur] == 0:
            state[cur] = 1
            path.append(cur)
            cur = TO[cur]
        if state[cur] == 1:
            idx = path.index(cur)
            cycle = path[idx:]
            cycles.append(cycle)
            for v in cycle:
                cycle_nodes.add(v)
        for v in path:
            state[v] = 2
    
    tree_surv, tree_die, tree_die_ext = compute_tree_dp_min(N, TO, cycle_nodes)
    
    total = 0.0
    for cycle in cycles:
        L = len(cycle)
        ts = [tree_surv[v] for v in cycle]
        td = [tree_die[v] for v in cycle]
        tde = [tree_die_ext[v] for v in cycle]
        
        cycle_cost = solve_cycle_dp_min(L, ts, td, tde)
        if cycle_cost == float('inf'):
            return float(brute_force_optimal(N, TO, True))
        total += cycle_cost
    
    return total


# ---------- MAX KILLS: formula-based ----------

def compute_max_kills(N: int, TO: List[int]) -> float:
    indeg = [0] * N
    for u in range(N):
        indeg[TO[u]] += 1
    
    indeg0_count = sum(1 for i in range(N) if indeg[i] == 0)
    
    # Find cycles
    state = [0] * N
    cycles = []
    
    for start in range(N):
        if state[start] != 0:
            continue
        path = []
        cur = start
        while state[cur] == 0:
            state[cur] = 1
            path.append(cur)
            cur = TO[cur]
        if state[cur] == 1:
            idx = path.index(cur)
            cycle = path[idx:]
            cycles.append(cycle)
        for v in path:
            state[v] = 2
    
    isolated_cycles = 0
    for cycle in cycles:
        if len(cycle) == 1:
            continue
        external_targets = False
        for v in cycle:
            for u in range(N):
                if TO[u] == v and u not in cycle:
                    external_targets = True
                    break
            if external_targets:
                break
        if not external_targets:
            isolated_cycles += 1
    
    return float(N - indeg0_count - isolated_cycles)


# ---------- Wrapper ----------

def compute_optimal(N: int, TO: List[int], minimize: bool) -> float:
    if N <= 8:
        return float(brute_force_optimal(N, TO, minimize))
    
    if minimize:
        result = compute_min_kills(N, TO)
        if result == float('inf'):
            return float(brute_force_optimal(N, TO, True))
        return result
    else:
        return compute_max_kills(N, TO)


# ---------- Environment ----------

class MafMafia_Environment(VerifiableEnvironment):
    prompt_template = 'There are {N} participants in a game, labeled from 0 to {N_minus_1}. Each participant `i` has a target participant TO[i]. The array TO is given as: {TO}\n\nYou are to determine a permutation P[0], P[1], ..., P[{N_minus_1}] of the {N} participants, representing the order in which they act. The game proceeds in that order as follows:\n- When a participant takes their turn, if they are still alive, they attempt to kill their target TO[i].\n- If the target has already been killed earlier, nothing happens.\n- A participant who has already been killed cannot act.\n\nPlease find a permutation that **{minimize_or_maximize}s the number of participants who get killed** by the end of the game. Output a single line containing the permutation P[0], P[1], ..., P[{N_minus_1}], separated by spaces.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(self) -> None:
        # Read N from parameter (default 3)
        N = self.parameter.get('N', 3)
        
        # Generate random TO array
        TO = [random.randrange(N) for _ in range(N)]
        
        # Randomly choose minimize or maximize
        minimize = random.choice([True, False])
        
        # Compute optimal and worst-case kills
        optimal_kills = compute_optimal(N, TO, minimize)
        if minimize:
            worst_kills = compute_optimal(N, TO, False)
        else:
            worst_kills = compute_optimal(N, TO, True)
        
        # Store everything in parameter for set_config roundtrip
        self.parameter['N'] = N
        self.parameter['TO'] = TO
        self.parameter['minimize'] = minimize
        self.parameter['optimal_kills'] = optimal_kills
        self.parameter['worst_kills'] = worst_kills

    def _prompt_generate(self) -> str:
        N = self.parameter['N']
        TO = self.parameter['TO']
        minimize = self.parameter['minimize']
        minimize_or_maximize = "minimize" if minimize else "maximize"
        TO_str = str(TO)
        
        return self.prompt_template.format(
            N=N,
            N_minus_1=N-1,
            TO=TO_str,
            minimize_or_maximize=minimize_or_maximize,
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        if answer is None:
            return None
        
        try:
            parts = answer.strip().split()
            perm = [int(p) for p in parts]
            N = self.parameter['N']
            if len(perm) != N:
                return None
            if set(perm) != set(range(N)):
                return None
            return perm
        except (ValueError, AttributeError):
            return None

    def scorer(self, output: str) -> float:
        perm = self.processor(output)
        
        if perm is None:
            return -1.0
        
        N = self.parameter['N']
        TO = self.parameter['TO']
        minimize = self.parameter['minimize']
        optimal = self.parameter['optimal_kills']
        worst = self.parameter['worst_kills']
        
        kills = simulate(N, TO, perm)
        
        if optimal == worst:
            return 1.0
        
        if minimize:
            if kills == optimal:
                return 1.0
            score = 1.0 - 2.0 * (kills - optimal) / (worst - optimal)
            return max(-1.0, min(1.0, score))
        else:
            if kills == optimal:
                return 1.0
            score = 1.0 - 2.0 * (optimal - kills) / (optimal - worst)
            return max(-1.0, min(1.0, score))
