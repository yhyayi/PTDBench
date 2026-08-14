# Implemented for reward_design: DifferenceConstraintSystemDAG
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Tuple

from Gym.environment import VerifiableEnvironment


class DifferenceConstraintSystemDAG_Environment(VerifiableEnvironment):
    prompt_template = (
        'There are {N} **positive integers** x[0], x[1], ..., x[{N_minus_1}]. '
        'They satisfy the following {M} equations/inequations:\n'
        '{relations}\n\n'
        'Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies '
        'all of the equations/inequations. Try your best to minimize '
        'x[0] + x[1] + ... + x[{N_minus_1}].\n\n'
        'Output Format: Your final answer should be a single line containing '
        'x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    #  _generate
    # ------------------------------------------------------------------ #
    def _generate(self) -> None:
        """Sample a random DAG difference-constraint system and compute the
        pointwise-minimal positive-integer solution (the 'optimal' solution).

        The problem is always feasible because constraints are derived from a
        witness (a randomly sampled feasible point).
        """

        N = random.randint(4, 8)

        # ---- sample a witness (feasible point) ---------------------------
        witness = [random.randint(1, 12) for _ in range(N)]

        # ---- pick a set of DAG edges ------------------------------------
        possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        max_edges = len(possible_edges)
        M_target = random.randint(N, min(3 * N, max_edges))
        chosen_edges = random.sample(possible_edges, M_target)

        # ---- derive constraints that the witness satisfies ---------------
        constraints: List[Tuple[int, int, int, str]] = []  # (u, v, d, '='|'>=')
        for u, v in chosen_edges:
            diff = witness[v] - witness[u]
            if random.random() < 0.5 and diff >= 1:
                # equality — tight for the witness
                constraints.append((u, v, diff, '='))
            else:
                # lower-bound inequality:  x[v] - x[u] >= d
                # choose d in [1, diff + small_slack] so witness satisfies it
                max_d = max(1, diff + random.randint(0, 3))
                d = random.randint(1, max_d)
                constraints.append((u, v, d, '>='))

        M = len(constraints)

        # ---- compute the TRUE minimal positive-integer solution ----------
        # Use relaxation from all-ones: start from the absolute minimum and
        # increase variables only when forced by a constraint.
        # Equality  x[v] - x[u] = d  is treated as two lower-bound constraints:
        #   x[v] >= x[u] + d   AND   x[u] >= x[v] - d.
        # Inequality x[v] - x[u] >= d  is a single lower-bound constraint.

        x = [1] * N
        changed = True
        iteration = 0
        max_iterations = 2000

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for u, v, d, typ in constraints:
                if typ == '=':
                    if x[v] < x[u] + d:
                        x[v] = x[u] + d
                        changed = True
                    if x[u] < x[v] - d:
                        x[u] = x[v] - d
                        changed = True
                else:  # '>='
                    if x[v] < x[u] + d:
                        x[v] = x[u] + d
                        changed = True

        # Safety: if relaxation did not converge, regenerate.
        if iteration >= max_iterations:
            self._generate()
            return

        optimal_solution = x
        optimal_sum = sum(optimal_solution)

        self.parameter = {
            'N': N,
            'M': M,
            'constraints': constraints,
            'optimal_solution': optimal_solution,
            'optimal_sum': optimal_sum,
        }

    # ------------------------------------------------------------------ #
    #  _prompt_generate
    # ------------------------------------------------------------------ #
    def _prompt_generate(self) -> str:
        N = self.parameter['N']
        M = self.parameter['M']
        constraints = self.parameter['constraints']

        lines: List[str] = []
        for u, v, d, typ in constraints:
            if typ == '=':
                lines.append(f'x[{v}] - x[{u}] = {d}')
            else:
                lines.append(f'x[{v}] - x[{u}] \u2265 {d}')  # ≥

        relations_str = '\n'.join(lines)
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=M,
            relations=relations_str,
        )

    # ------------------------------------------------------------------ #
    #  _process  (called by the base-class processor)
    # ------------------------------------------------------------------ #
    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Extract the first N integers from the answer string."""
        if answer is None:
            return None

        numbers = re.findall(r'-?\d+', answer)
        if not numbers:
            return None

        try:
            parsed = [int(n) for n in numbers]
        except ValueError:
            return None

        N = self.parameter['N']
        if len(parsed) < N:
            return None
        return parsed[:N]

    # ------------------------------------------------------------------ #
    #  _extract_answer
    # ------------------------------------------------------------------ #
    def _extract_answer(self, output: str) -> Optional[str]:
        """Extract the answer portion from raw model output.

        Tries, in order:
          1. Content inside <answer>…</answer> tags.
          2. If no tags found, use everything after 'Assistant:' / assistant marker.
        """
        work = output
        if 'Assistant:' in work:
            work = work.split('Assistant:', 1)[1]
        elif '<|im_start|>assistant' in work:
            work = work.split('<|im_start|>assistant', 1)[1]

        # Try tagged extraction first
        answer_pattern = (
            re.escape(self.answer_markers[0])
            + r'(.*?)'
            + re.escape(self.answer_markers[1])
        )
        matches = list(re.finditer(answer_pattern, work, re.DOTALL))
        if matches:
            return matches[-1].group(1)

        # Fallback: use whatever follows the assistant marker
        if 'Assistant:' in output or '<|im_start|>assistant' in output:
            return work

        # Last resort: use the whole output
        return output

    # ------------------------------------------------------------------ #
    #  scorer
    # ------------------------------------------------------------------ #
    def scorer(self, output: str) -> float:
        """Score the model output in [-1.0, +1.0].

        Reward design:
          * -1.0            – answer not parseable as N integers
          * [-1.0, -0.5]    – parsed but some variables are not positive
                               (linear penalty per non-positive variable)
          * (-0.5, 0.0]     – all positive but some constraints violated
                               (linear interpolation by fraction satisfied)
          * [0.0, 1.0)      – all constraints satisfied, sum sub-optimal
                               (decays with excess over optimal sum)
          * +1.0            – all constraints satisfied AND sum equals the
                               minimal possible sum (optimal solution)
        """

        # ----- extract answer from raw output --------------------------------
        answer = self._extract_answer(output)

        # ----- parse --------------------------------------------------------
        parsed = self._process(answer)
        if parsed is None:
            return -1.0

        x = parsed
        N = self.parameter['N']
        constraints = self.parameter['constraints']
        optimal_sum = self.parameter['optimal_sum']

        # ----- positivity ---------------------------------------------------
        pos_violations = sum(1 for v in x if v <= 0)
        if pos_violations > 0:
            return -1.0 + 0.5 * (pos_violations / N)

        # ----- constraint satisfaction --------------------------------------
        total_c = len(constraints)
        c_violations = 0
        for u, v, d, typ in constraints:
            if typ == '=':
                if x[v] - x[u] != d:
                    c_violations += 1
            else:  # '>='
                if x[v] - x[u] < d:
                    c_violations += 1

        if c_violations > 0:
            frac_satisfied = (total_c - c_violations) / total_c
            return -0.5 + 0.5 * frac_satisfied

        # ----- all constraints satisfied — score optimality -----------------
        actual_sum = sum(x)
        if actual_sum == optimal_sum:
            return 1.0

        # Sub-optimal: score decays with excess over optimum.
        excess = actual_sum - optimal_sum
        score = max(0.0, 1.0 - excess / optimal_sum)
        return float(score)
