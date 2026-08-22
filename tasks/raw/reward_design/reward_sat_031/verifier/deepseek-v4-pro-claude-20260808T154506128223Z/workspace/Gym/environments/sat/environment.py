# SAT Environment: Boolean Satisfiability
# The model is given a CNF formula and must find a satisfying assignment.
# scorer(output) -> float in [-1.0, +1.0].

import re
import random
from typing import Optional, List, Dict, Any

from Gym.environment import VerifiableEnvironment


class SAT_Environment(VerifiableEnvironment):
    prompt_template = (
        'There are {N} boolean (0/1) values x[0], x[1], ..., x[{N_minus_1}]. '
        'Each of the following {M} expressions (`|` means OR, `!` means NOT) must equal 1:\n'
        '{expressions}\n\n'
        'Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies '
        'the conditions above.\n\n'
        'Output Format: Your final answer should be a single line containing '
        'x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.\n'
        'Example: `{N_boolean}` (do **NOT** include quotes or backticks).'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Internal state set during _generate
        self.clauses: List[List[int]] = []       # each clause = list of literals
        self.reference_answer: List[int] = []    # the hidden satisfying assignment

    # ------------------------------------------------------------------
    #  Config (needed so verification can reconstruct the same problem)
    # ------------------------------------------------------------------

    def set_config(self, config: Dict) -> None:
        """Restore config and regenerate the problem deterministically."""
        super().set_config(config)
        # Regenerate the same problem using the stored seed
        self.generator(seed=self.seed, parameter=self.parameter)

    # ------------------------------------------------------------------
    #  Problem generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """
        Generate a random satisfiable CNF formula.

        self.parameter must contain:
          N       (int) – number of boolean variables
          M       (int) – number of clauses
          density (float, optional) – controls clause width;
                   k = max(2, min(N, 2 + round(density * N)))
        """
        N = int(self.parameter["N"])
        M = int(self.parameter["M"])
        density = float(self.parameter.get("density", 0.3))

        # Clause width – density drives how many distinct variables appear per clause.
        k = max(2, min(N, 2 + round(density * N)))

        # 1. Hidden satisfying assignment
        self.reference_answer = [random.randint(0, 1) for _ in range(N)]

        # 2. Generate M clauses guaranteed to be satisfied by it
        self.clauses = []
        all_vars = list(range(N))
        for _ in range(M):
            clause = self._make_satisfied_clause(N, k, all_vars)
            self.clauses.append(clause)

        # Persist for later stages
        self.parameter["reference_answer"] = " ".join(
            str(v) for v in self.reference_answer
        )

    def _make_satisfied_clause(
        self, N: int, k: int, all_vars: List[int]
    ) -> List[int]:
        """
        Return a clause (list of int literals) that *is* satisfied by
        self.reference_answer.

        Literal encoding (1-indexed so sign works naturally):
          + (i+1)  →  x[i] == 1
          - (i+1)  →  x[i] == 0
        """
        # Choose k distinct variables
        chosen = random.sample(all_vars, k)

        # Random initial polarities
        clause = []
        for v in chosen:
            clause.append(v + 1 if random.random() < 0.5 else -(v + 1))

        # If the clause is not satisfied, flip one random literal so it is.
        if not self._eval_clause(clause, self.reference_answer):
            idx = random.randrange(len(clause))
            var = abs(clause[idx]) - 1
            clause[idx] = var + 1 if self.reference_answer[var] == 1 else -(var + 1)

        return clause

    # ------------------------------------------------------------------
    #  Prompt generation
    # ------------------------------------------------------------------

    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        M = self.parameter["M"]

        lines = []
        for clause in self.clauses:
            parts = []
            for lit in clause:
                var = abs(lit) - 1
                parts.append(f"x[{var}]" if lit > 0 else f"!x[{var}]")
            lines.append("(" + " | ".join(parts) + ")")

        # Example output line (random, just to illustrate format)
        example = " ".join(str(random.randint(0, 1)) for _ in range(N))

        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=M,
            expressions="\n".join(lines),
            N_boolean=example,
        )

    # ------------------------------------------------------------------
    #  Answer parsing
    # ------------------------------------------------------------------

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """
        Parse the answer string (content between answer markers) into a
        list of 0/1 integers.  Returns None if parsing fails.
        """
        if answer is None:
            return None

        answer = answer.strip().strip('"').strip("'").strip("`")

        result: List[int] = []
        for token in answer.split():
            token = token.strip().rstrip(",;")
            try:
                v = int(token)
                if v in (0, 1):
                    result.append(v)
            except ValueError:
                continue

        return result if result else None

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    def scorer(self, output: str) -> float:
        """
        Score the model's raw output.

        Returns a float in [-1.0, +1.0]:
          +1.0   every clause satisfied
          -1.0   parsing failure or zero clauses satisfied
          Linear interpolation otherwise:
              score = 2 * (satisfied / total) - 1
        """
        M = len(self.clauses)
        N = self.parameter.get("N", 0)
        if M == 0 or N == 0:
            return 1.0  # vacuously satisfied

        # ---- extract the answer from inside the answer markers ----
        pat = re.escape(self.answer_markers[0]) + r"(.*?)" + re.escape(self.answer_markers[1])
        matches = list(re.finditer(pat, output, re.DOTALL))

        if matches:
            answer_str = matches[-1].group(1)
        else:
            # Fallback: scan for a line that looks like a truth assignment
            answer_str = self._fallback_extract(output, N)

        assignment = self._process(answer_str)

        if assignment is None or len(assignment) == 0:
            return -1.0

        # Pad / truncate to N
        if len(assignment) < N:
            assignment += [0] * (N - len(assignment))
        else:
            assignment = assignment[:N]

        # Count satisfied clauses
        satisfied = sum(
            1 for clause in self.clauses if self._eval_clause(clause, assignment)
        )

        return 2.0 * (satisfied / M) - 1.0

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_clause(clause: List[int], assignment: List[int]) -> bool:
        """True if at least one literal in *clause* is satisfied by *assignment*."""
        for lit in clause:
            var = abs(lit) - 1
            if var >= len(assignment):
                return False
            if (lit > 0 and assignment[var] == 1) or (lit < 0 and assignment[var] == 0):
                return True
        return False

    def _fallback_extract(self, output: str, expected_len: int) -> Optional[str]:
        """
        When the model does not use \\boxed{…}, scan the output for a line
        that looks like a truth assignment of *expected_len* 0/1 values.
        """
        for line in output.split("\n"):
            line = line.strip()
            parts = line.split()
            if len(parts) >= expected_len:
                try:
                    vals = [int(p) for p in parts]
                    if all(v in (0, 1) for v in vals):
                        return line
                except ValueError:
                    continue
        return None
