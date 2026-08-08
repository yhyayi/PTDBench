# Integer Programming environment: find integer solutions to a system of linear inequalities.
# scorer(output) -> float in [-1.0, +1.0].

import re
import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class IntegerProgramming_Environment(VerifiableEnvironment):
    # Template will be filled by _prompt_generate (handles N=1 edge case)
    prompt_template = (
        'There {is_are} {N} integer{s_plural} {x_list}. '
        'They satisfy the following {M} inequations:\n'
        '{inequations}\n\n'
        'Please find any solution {x_list} that satisfies the inequations.\n\n'
        'Output your solution as space-separated integers inside <answer> tags.\n'
        'Example: <answer>{example_values}</answer>.\n'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Will be populated by _generate
        self.N: int = 0
        self.M: int = 0
        self.inequalities: List[Tuple[List[int], str, int]] = []  # (coeffs, op, rhs)
        self.known_solution: List[int] = []

    def _generate(self) -> None:
        """Generate a random integer programming problem with a known feasible solution."""
        N = self.parameter.get("N", 3)
        M = self.parameter.get("M", 4)

        self.N = N
        self.M = M
        self.inequalities = []
        self.known_solution = []

        # Generate a known feasible solution: random integers in [-5, 10]
        x_feasible = [random.randint(-5, 10) for _ in range(N)]
        self.known_solution = x_feasible

        # Allowed operators (weighted towards inequalities for more interesting problems)
        ops_pool = ["<=", "<=", ">=", ">=", "="]

        for _ in range(M):
            # Generate random coefficients in [-5, 5] (excluding all zeros)
            while True:
                coeffs = [random.randint(-5, 5) for _ in range(N)]
                if any(c != 0 for c in coeffs):
                    break

            # Pick a random operator
            op = random.choice(ops_pool)

            # Compute LHS for the feasible solution
            lhs = sum(c * x for c, x in zip(coeffs, x_feasible))

            if op == "<=":
                # Add random non-negative slack [0, 10]
                rhs = lhs + random.randint(0, 10)
            elif op == ">=":
                # Subtract random non-negative slack [0, 10]
                rhs = lhs - random.randint(0, 10)
            else:  # "="
                rhs = lhs

            self.inequalities.append((coeffs, op, rhs))

    def _prompt_generate(self) -> str:
        """Format the prompt with the generated inequalities."""
        N = self.N
        M = self.M

        # Build variable list description
        if N == 1:
            x_list = "x[0]"
            example_values = "5"
        else:
            x_list = f"x[0], x[1], ..., x[{N - 1}]"
            example_values = " ".join(str(i) for i in range(1, N + 1))

        is_are = "is" if N == 1 else "are"
        s_plural = "" if N == 1 else "s"

        # Format inequalities as readable strings
        ineq_lines = []
        for idx, (coeffs, op, rhs) in enumerate(self.inequalities):
            # Build the left-hand side string
            terms = []
            for j, c in enumerate(coeffs):
                if c == 0:
                    continue
                if not terms:
                    # First non-zero term
                    if c == 1:
                        term = f"x[{j}]"
                    elif c == -1:
                        term = f"-x[{j}]"
                    else:
                        term = f"{c}*x[{j}]"
                else:
                    if c == 1:
                        term = f"+ x[{j}]"
                    elif c == -1:
                        term = f"- x[{j}]"
                    elif c > 0:
                        term = f"+ {c}*x[{j}]"
                    else:
                        term = f"- {abs(c)}*x[{j}]"
                terms.append(term)

            if not terms:
                lhs_str = "0"
            else:
                lhs_str = " ".join(terms)

            ineq_lines.append(f"  ({idx + 1}) {lhs_str} {op} {rhs}")

        ineq_str = "\n".join(ineq_lines)

        return self.prompt_template.format(
            N=N,
            M=M,
            x_list=x_list,
            is_are=is_are,
            s_plural=s_plural,
            inequations=ineq_str,
            example_values=example_values,
        )

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """Parse the model's answer into a list of N integers."""
        if answer is None:
            return None

        # Strip whitespace
        answer = answer.strip()

        # Look for integers (possibly negative) in the answer
        numbers = re.findall(r'-?\d+', answer)

        if len(numbers) < self.N:
            return None

        try:
            parsed = [int(x) for x in numbers[:self.N]]
            return parsed
        except ValueError:
            return None

    def scorer(self, output: str) -> float:
        """Score the model's output. Returns value in [-1.0, +1.0].

        Score = 2 * (num_satisfied / M) - 1, giving:
        - +1.0 when all constraints satisfied
        - 0.0 when half satisfied
        - -1.0 when none satisfied (or unparseable)
        """
        # First try to extract answer using the base processor (looks for <answer> tags)
        parsed = self.processor(output)

        # Fallback: if no tagged answer found, try to parse from the whole output
        if parsed is None:
            parsed = self._parse_from_raw_output(output)

        if parsed is None:
            return -1.0

        if len(parsed) != self.N:
            return -1.0

        # Count how many inequalities are satisfied
        satisfied = 0
        for coeffs, op, rhs in self.inequalities:
            lhs = sum(c * x for c, x in zip(coeffs, parsed))
            if op == "<=":
                if lhs <= rhs:
                    satisfied += 1
            elif op == ">=":
                if lhs >= rhs:
                    satisfied += 1
            else:  # "="
                if lhs == rhs:
                    satisfied += 1

        # Linear mapping: ratio in [0, M] -> score in [-1, 1]
        ratio = satisfied / self.M
        score = 2.0 * ratio - 1.0

        # Clamp to [-1, 1] for safety
        return max(-1.0, min(1.0, score))

    def _parse_from_raw_output(self, output: str) -> Optional[List[int]]:
        """Fallback: parse N integers from raw output without requiring tags."""
        # Remove common prefixes
        cleaned = output
        if "Assistant:" in cleaned:
            cleaned = cleaned.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in cleaned:
            cleaned = cleaned.split("<|im_start|>assistant", 1)[1]

        numbers = re.findall(r'-?\d+', cleaned)
        if len(numbers) < self.N:
            return None

        try:
            return [int(x) for x in numbers[:self.N]]
        except ValueError:
            return None
