# SetSplitting environment: partition a ground set into S1, S2 such that
# every given subset contains at least one element from each side.
# scorer(output) -> float in [-1.0, +1.0].

import random
import re
from typing import Optional, List, Set

from Gym.environment import VerifiableEnvironment


class SetSplitting_Environment(VerifiableEnvironment):
    prompt_template = (
        'Define the full set `S` as all {N} integers from `0` to `{N_minus_1}`.\n'
        '\n'
        'Your task is to partition `S` into two **disjoint subsets** `S1` and `S2` such that:\n'
        '- `S1 ∪ S2 = S` and `S1 ∩ S2 = ∅`\n'
        '- For each of the following {M} subsets (each a subset of `S`), the subset is '
        '**not fully contained** in either `S1` or `S2`. That is, each subset must contain '
        '**at least one element from S1** and **at least one element from S2**.\n'
        '\n'
        'The list of {M} subsets is as follows:\n'
        '{Sets}\n'
        '\n'
        '**Output Format:** Your final answer should be a single line containing the elements '
        'of `S1`, separated by spaces. (Subset `S2` is implicitly defined as `S \\ S1`.)'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Generation
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """
        Create a solvable SetSplitting instance.
        Strategy: first pick a hidden partition (S1_hidden, S2_hidden),
        then sample subsets that are consistent with it (each subset
        includes at least one element from each side).
        """
        # Reasonable ranges; can be overridden via self.parameter
        N = self.parameter.get("N", None)
        M = self.parameter.get("M", None)

        if N is None:
            N = random.randint(4, 12)
        if M is None:
            M = random.randint(3, 15)

        # Build universe S = {0, 1, ..., N-1}
        universe = list(range(N))

        # Randomly split into two non-empty parts (hidden solution)
        k = random.randint(1, N - 1)
        shuffled = random.sample(universe, N)
        S1_hidden = set(shuffled[:k])
        S2_hidden = set(shuffled[k:])

        # Generate M subsets, each guaranteed to be "splittable"
        subsets: List[List[int]] = []
        for _ in range(M):
            # pick at least one from each side
            elem_s1 = random.choice(list(S1_hidden))
            elem_s2 = random.choice(list(S2_hidden))
            core = {elem_s1, elem_s2}
            # Optionally add more random elements
            extras = random.sample(
                universe, random.randint(0, min(N - 2, N // 2))
            )
            subset_set = core | set(extras)
            subset = sorted(subset_set)
            subsets.append(subset)

        # Store in parameter dict for prompt generation & scoring
        self.parameter["N"] = N
        self.parameter["M"] = M
        self.parameter["subsets"] = subsets
        self.parameter["S1_hidden"] = sorted(S1_hidden)
        self.parameter["S2_hidden"] = sorted(S2_hidden)

    # ------------------------------------------------------------------
    #  Prompt
    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        M = self.parameter["M"]
        subsets = self.parameter["subsets"]

        # Format subsets nicely: one per line, e.g. "{0, 3, 5}"
        lines = []
        for i, s in enumerate(subsets):
            set_str = "{" + ", ".join(str(x) for x in s) + "}"
            lines.append(f"  Subset {i}: {set_str}")
        sets_block = "\n".join(lines)

        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=M,
            Sets=sets_block,
        )

    # ------------------------------------------------------------------
    #  Process (parse answer)
    # ------------------------------------------------------------------
    # Pattern: integer not embedded in a larger identifier (e.g. won't
    # match the '1' inside 'S1' or '2' inside 'S2').
    _INT_RE = re.compile(r'(?<![a-zA-Z])-?\d+(?![a-zA-Z])')

    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        """
        Try to parse the model answer into a list of integers (S1).
        Returns None if parsing fails.
        """
        if answer is None:
            return None

        answer = answer.strip()
        if not answer:
            return None

        # Extract all integer tokens that are not part of a word
        matches = self._INT_RE.findall(answer)
        if not matches:
            return None

        parsed = [int(x) for x in matches]
        return parsed

    # ------------------------------------------------------------------
    #  Scorer
    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """
        Score in [-1.0, +1.0].

        -1.0 : unparseable answer or completely invalid partition.
        Partial credit: linearly scaled by fraction of subsets that are
        properly split, assuming the partition is valid.
        +1.0 : perfect answer — every subset is split.
        """
        N = self.parameter["N"]
        M = self.parameter["M"]
        subsets = self.parameter["subsets"]

        # Step 1 — parse the answer into S1.
        # Delegate to processor to handle <answer> tags.
        parsed = self.processor(output)  # calls _process internally

        if parsed is None or len(parsed) == 0:
            return -1.0

        # Step 2 — validate the partition
        S1 = set(parsed)
        universe = set(range(N))

        # S1 must be a subset of the universe
        if not S1.issubset(universe):
            return -1.0

        # Check for duplicates (len difference)
        if len(parsed) != len(S1):
            return -0.8  # duplicate elements — slight penalty

        # S2 is implicitly the complement
        S2 = universe - S1

        # Step 3 — count how many subsets are split
        split_count = 0
        for sub in subsets:
            sub_set = set(sub)
            has_s1 = bool(sub_set & S1)
            has_s2 = bool(sub_set & S2)
            if has_s1 and has_s2:
                split_count += 1

        # Step 4 — map [0, M] → [-1.0, +1.0]
        if M == 0:
            score = 1.0
        else:
            score = 2.0 * (split_count / M) - 1.0

        # Clamp to [-1.0, +1.0] for safety
        score = max(-1.0, min(1.0, score))
        return score
