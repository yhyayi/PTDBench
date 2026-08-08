# Implementation of TetrisAttack environment for RLVE verifiable training.
# scorer(output) -> float in [-1.0, +1.0].

import random
from typing import Optional, List, Tuple
from collections import deque
from Gym.environment import VerifiableEnvironment


class TetrisAttack_Environment(VerifiableEnvironment):
    prompt_template = (
        'There is an array A (initially it is of length 2 × {N}, containing each integer from '
        '0 to {N_minus_1} exactly twice). Initially, the array A is: {A}\n\n'
        'The array follows this rule:\n'
        '- If there are two adjacent equal elements A[i] == A[i + 1], they are both removed '
        'from the array.\n'
        '- After each removal, the array is compacted (i.e., elements are re-indexed from 0 '
        'to the new length), and the process continues as long as such adjacent pairs exist.\n\n'
        'Once the array becomes stable (i.e., no adjacent equal pairs remain), you may perform '
        'a **swap** between any two adjacent elements A[i] and A[i + 1] '
        '(0 ≤ i < current array length - 1). After a swap, the same removal process restarts '
        'and continues until stable again. Please **remove all elements from the array**, using '
        'the **minimum number of swaps**. Output a single line containing the indices of the '
        'swaps (space-separated), where each index `i` indicates a swap between A[i] and '
        'A[i + 1].'
    )

    # Maximum value of N for which we attempt exact BFS.
    BFS_MAX_N = 5
    # For N=6, try BFS with this many states before falling back.
    BFS_MAX_STATES_N6 = 200_000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    #  Core mechanics
    # ------------------------------------------------------------------ #

    @staticmethod
    def reduce(arr: List[int]) -> List[int]:
        """
        Repeatedly remove adjacent equal elements until stable.
        Equivalent to stack-based reduction: push; if top == new, pop else push.
        """
        stack = []
        for x in arr:
            if stack and stack[-1] == x:
                stack.pop()
            else:
                stack.append(x)
        return stack

    @staticmethod
    def apply_swap(arr: List[int], idx: int) -> List[int]:
        """Swap arr[idx] and arr[idx+1], then reduce until stable. Returns new array."""
        if idx < 0 or idx >= len(arr) - 1:
            raise ValueError(f"Swap index {idx} out of range for array of length {len(arr)}")
        new_arr = arr[:]
        new_arr[idx], new_arr[idx + 1] = new_arr[idx + 1], new_arr[idx]
        return TetrisAttack_Environment.reduce(new_arr)

    @staticmethod
    def simulate_swaps(initial_arr: List[int], swaps: List[int]) -> Tuple[List[int], int]:
        """
        Simulate a sequence of swaps starting from initial_arr.
        First reduces the initial array, then applies each swap followed by reduction.
        Returns (final_array, number_of_valid_swaps_applied).
        Stops early if a swap index is out of bounds.
        """
        arr = TetrisAttack_Environment.reduce(initial_arr)
        valid_swaps = 0
        for idx in swaps:
            if idx < 0 or idx >= len(arr) - 1:
                break
            arr = TetrisAttack_Environment.apply_swap(arr, idx)
            valid_swaps += 1
            if len(arr) == 0:
                break
        return arr, valid_swaps

    # ------------------------------------------------------------------ #
    #  Exact solver via BFS (for small N)
    # ------------------------------------------------------------------ #

    @staticmethod
    def bfs_optimal(arr: List[int], max_states: int = 500_000) -> Optional[int]:
        """
        BFS to find the minimum number of swaps to clear the array.
        Returns the minimum swaps, or None if not found within max_states.
        """
        start = tuple(TetrisAttack_Environment.reduce(arr))
        if len(start) == 0:
            return 0

        queue = deque([(start, 0)])
        visited = {start: 0}

        while queue:
            state, dist = queue.popleft()
            n = len(state)

            if n == 0:
                continue

            for i in range(n - 1):
                if state[i] == state[i + 1]:
                    continue

                new_arr = TetrisAttack_Environment.apply_swap(list(state), i)
                new_state = tuple(new_arr)

                if new_state in visited:
                    continue

                visited[new_state] = dist + 1
                if len(new_state) == 0:
                    return dist + 1

                queue.append((new_state, dist + 1))

                if len(visited) >= max_states:
                    return None

        return None

    # ------------------------------------------------------------------ #
    #  Multi-start randomized greedy (fallback)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _greedy_one_pass(arr: List[int], rng: random.Random, max_swaps: int = 500) -> int:
        """
        Single pass of randomized greedy.
        At each step, prefers swaps that reduce length; breaks ties randomly.
        Tracks visited states to avoid cycles.
        """
        cur = TetrisAttack_Environment.reduce(arr)
        swaps = 0
        visited = {tuple(cur)}

        while len(cur) > 0 and swaps < max_swaps:
            # Collect all candidate swaps
            improving = []  # swaps that reduce length
            neutral_new = []  # swaps that keep length but lead to new states
            neutral_old = []  # swaps that keep length and lead to visited states

            for i in range(len(cur) - 1):
                if cur[i] == cur[i + 1]:
                    continue
                new_arr = TetrisAttack_Environment.apply_swap(cur, i)
                new_len = len(new_arr)
                if new_len < len(cur):
                    improving.append((i, new_len))
                elif tuple(new_arr) not in visited:
                    neutral_new.append(i)
                else:
                    neutral_old.append(i)

            if improving:
                # Pick the swap that minimizes the resulting length;
                # break ties randomly.
                min_len = min(l for _, l in improving)
                candidates = [i for i, l in improving if l == min_len]
                best_swap = rng.choice(candidates)
            elif neutral_new:
                best_swap = rng.choice(neutral_new)
            elif neutral_old:
                # All swaps lead to visited states; pick randomly to escape
                best_swap = rng.choice(neutral_old)
            else:
                break  # no valid swap

            cur = TetrisAttack_Environment.apply_swap(cur, best_swap)
            visited.add(tuple(cur))
            swaps += 1

        return swaps if len(cur) == 0 else max_swaps + 1  # penalty if not cleared

    @staticmethod
    def greedy_solve(arr: List[int], num_restarts: int = 50) -> int:
        """
        Multi-start randomized greedy. Runs multiple passes with different random
        seeds and returns the best (minimum) number of swaps found.
        """
        base_rng = random.Random(42)  # fixed base for reproducibility
        best = None
        for _ in range(num_restarts):
            seed = base_rng.randint(0, 2**31 - 1)
            rng = random.Random(seed)
            s = TetrisAttack_Environment._greedy_one_pass(arr, rng)
            if best is None or s < best:
                best = s
        return best

    # ------------------------------------------------------------------ #
    #  API methods
    # ------------------------------------------------------------------ #

    def _generate(self) -> None:
        """
        Generate a problem instance.
        Uses self.seed and self.parameter (which must contain 'N').
        Creates a random permutation of 2N elements (0..N-1 each twice),
        computes the optimal answer (or best known), and stores it.
        """
        N = self.parameter.get("N", 3)
        self.parameter["N"] = N

        # Generate the array: each number 0..N-1 appears exactly twice
        arr = []
        for i in range(N):
            arr.extend([i, i])
        random.shuffle(arr)

        self.parameter["initial_array"] = arr

        # Compute optimal / best-known answer
        reduced = self.reduce(arr)
        if len(reduced) == 0:
            self.parameter["optimal_swaps"] = 0
            self.parameter["solution_method"] = "trivial"
            return

        # Try BFS for exact optimal
        if N <= self.BFS_MAX_N:
            optimal = self.bfs_optimal(arr)
            if optimal is not None:
                self.parameter["optimal_swaps"] = optimal
                self.parameter["solution_method"] = "bfs"
                return
        elif N == 6:
            # Try BFS with limited states
            optimal = self.bfs_optimal(arr, max_states=self.BFS_MAX_STATES_N6)
            if optimal is not None:
                self.parameter["optimal_swaps"] = optimal
                self.parameter["solution_method"] = "bfs"
                return

        # Fallback: multi-start randomized greedy
        greedy_swaps = self.greedy_solve(arr)
        self.parameter["optimal_swaps"] = greedy_swaps
        self.parameter["solution_method"] = "greedy"

    def _prompt_generate(self) -> str:
        """Format the prompt using the generated parameters."""
        N = self.parameter["N"]
        arr = self.parameter["initial_array"]
        arr_str = "[" + ", ".join(str(x) for x in arr) + "]"
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            A=arr_str,
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """
        Parse the answer string into a list of swap indices.
        Returns None if the answer is missing or malformed.
        """
        if answer is None:
            return None
        try:
            parts = answer.strip().split()
            return [int(p) for p in parts]
        except (ValueError, TypeError):
            return None

    def scorer(self, output: str) -> float:
        """
        Score the model output.
        Returns a float in [-1.0, +1.0].

        Scoring logic:
        - Parse the swap sequence from the output.
        - Simulate the swaps on the initial array.
        - If the array is not cleared: score = -1.0
        - If cleared with S swaps and optimal is S*:
          - S <= S*: score = 1.0
          - S > S*: score = 1.0 - 2*(S - S*)/max(S*, 1), clamped to [-1, 1]
        """
        swaps = self.processor(output)

        initial_arr = self.parameter.get("initial_array")
        if initial_arr is None:
            return -1.0

        optimal = self.parameter.get("optimal_swaps")
        if optimal is None:
            return -1.0

        # If no valid swaps parsed
        if swaps is None:
            if optimal == 0:
                return 1.0
            return -1.0

        # Simulate
        final_arr, valid_swaps = self.simulate_swaps(initial_arr, swaps)

        if len(final_arr) != 0:
            return -1.0

        S = valid_swaps
        S_star = optimal

        if S <= S_star:
            return 1.0

        # Partial credit: linear degradation from 1.0 to -1.0
        # At S = 2*S_star, score becomes -1.0
        ratio = (S - S_star) / max(S_star, 1)
        score = 1.0 - 2.0 * ratio
        return max(-1.0, min(1.0, score))
