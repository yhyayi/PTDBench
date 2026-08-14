"""
VisibleLine — upper envelope of N lines.

A line y = m*x + b is **visible** (looking downward from y = +∞) if there
exists at least one x where it attains the maximum y among all lines.

The model must return the *indices* (0-based) of every visible line.
"""

import random
import re
from typing import Optional, List, Tuple, Set

from Gym.environment import VerifiableEnvironment


class VisibleLine_Environment(VerifiableEnvironment):
    prompt_template = (
        "You are given {N} lines on the 2D plane:\n"
        "{lines}\n\n"
        "We say a line is **visible** if any portion of it can be seen when "
        "viewed from y = +∞ (i.e., looking vertically downward). That is, a "
        "line is visible if there exists at least one x-coordinate such that "
        "this line lies on top (i.e., has the maximum y-value) at that x "
        "among all lines.\n\n"
        "**Output Format:** A single line containing the indices of all "
        "visible lines, in any order, separated by spaces."
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_upper_envelope(lines: List[Tuple[int, int]]) -> Set[int]:
        """
        Given *lines* as list of (slope m, intercept b) with their index
        implicit in the list position, return the set of indices that belong
        to the upper envelope.
        """
        n = len(lines)

        # ---- 1. group by slope; keep only max-intercept entries ----
        # slope -> list of (intercept, idx)
        slope_map: dict = {}
        for idx, (m, b) in enumerate(lines):
            slope_map.setdefault(m, []).append((b, idx))

        # For each slope keep only the entries whose intercept equals the
        # maximum for that slope (coincident lines are all visible if that
        # slope survives the envelope).
        candidates: List[Tuple[int, int, List[int]]] = []  # (m, b, [indices])
        for m, entries in slope_map.items():
            max_b = max(b for b, _ in entries)
            idxs = [idx for b, idx in entries if b == max_b]
            candidates.append((m, max_b, idxs))

        # ---- 2. sort by slope then intercept ----
        candidates.sort(key=lambda x: (x[0], x[1]))

        # ---- 3. upper envelope via monotonic stack ----
        # Each stack element: (m, b, [indices])
        stack: List[Tuple[int, int, List[int]]] = []

        for m, b, idxs in candidates:
            # If top of stack has the same slope it must also have the same
            # intercept (we already filtered). Just merge indices.
            if stack and stack[-1][0] == m:
                # same slope & intercept → coincident lines
                stack[-1][2].extend(idxs)
                continue

            # Pop while the second-from-top line makes the top line redundant.
            while len(stack) >= 2:
                m1, b1, _ = stack[-2]
                m2, b2, _ = stack[-1]

                # Condition for L2 (top) to be redundant:
                #   x_intersect(L2, L_new) < x_intersect(L1, L2)
                # In cross-product form with positive denominators:
                left  = (b - b2) * (m2 - m1)   # (b_c - b_b)*(m_b - m_a)
                right = (b2 - b1) * (m - m2)    # (b_b - b_a)*(m_c - m_b)
                if left > right:
                    stack.pop()
                else:
                    break

            stack.append((m, b, idxs))

        # ---- 4. collect visible indices ----
        visible: Set[int] = set()
        for _, _, idxs in stack:
            visible.update(idxs)

        return visible

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Populated by _generate:
        self.lines: List[Tuple[int, int]] = []   # (slope, intercept)
        self.visible_indices: Set[int] = set()

    # ------------------------------------------------------------------
    def _generate(self) -> None:
        """Generate N random non-vertical lines and compute the upper envelope."""
        N: int = self.parameter.get("N", 3)

        # Sample distinct lines.  Keep slopes / intercepts modest so that the
        # problem remains human-readable.
        MAX_ABS = 10
        seen: Set[Tuple[int, int]] = set()
        lines: List[Tuple[int, int]] = []

        # To avoid spending forever when the grid is small, fall back to
        # enlarging the range if needed.
        attempt_range = MAX_ABS
        while len(lines) < N:
            m = random.randint(-attempt_range, attempt_range)
            b = random.randint(-attempt_range, attempt_range)
            if (m, b) not in seen:
                seen.add((m, b))
                lines.append((m, b))
            # If we tried many times, expand the coefficient range slightly.
            if len(seen) > 4 * attempt_range * attempt_range:
                attempt_range += 5

        random.shuffle(lines)                     # scramble display order
        self.lines = lines
        self.N = N
        self.visible_indices = self._compute_upper_envelope(lines)

    # ------------------------------------------------------------------
    def _prompt_generate(self) -> str:
        """Format the prompt with the generated lines."""
        line_strs = []
        for i, (m, b) in enumerate(self.lines):
            # y = mx + b  (show sign naturally)
            if b >= 0:
                line_strs.append(f"  Line {i}: y = {m}x + {b}")
            else:
                line_strs.append(f"  Line {i}: y = {m}x - {abs(b)}")
        lines_text = "\n".join(line_strs)
        return self.prompt_template.format(N=self.N, lines=lines_text)

    # ------------------------------------------------------------------
    def _process(self, answer: Optional[str]) -> Optional[Set[int]]:
        """
        Parse the model's answer (the text between <answer>…</answer>) into a
        set of integer indices.  Returns None if the answer is missing or
        unparseable.
        """
        if answer is None:
            return None

        # Try to extract all integer tokens.
        tokens = re.findall(r"-?\d+", answer)
        if not tokens:
            return None

        try:
            indices = {int(t) for t in tokens}
        except ValueError:
            return None

        # Sanity: indices should be in [0, N-1].
        # We keep all parsed values; the scorer will handle out-of-range ones.
        return indices

    # ------------------------------------------------------------------
    def scorer(self, output: str) -> float:
        """
        Score in [-1.0, +1.0].

        Uses the Jaccard index between the predicted set of visible-line
        indices and the ground-truth set, mapped linearly:

            score = 2 * Jaccard - 1

        Jaccard = |intersection| / |union|, treating an empty prediction as
        having Jaccard = 0 (unless ground truth is also empty, which cannot
        happen here).

        Perfect match → +1.0.
        Completely wrong → -1.0.
        """
        predicted = self.processor(output)  # uses _process internally

        gt = self.visible_indices

        if predicted is None:
            # No parseable answer → worst score.
            return -1.0

        # Intersection (valid indices only: 0..N-1)
        valid_range = set(range(self.N))
        pred_valid = predicted & valid_range
        gt_valid = gt & valid_range

        inter = pred_valid & gt_valid
        union = pred_valid | gt_valid

        if len(union) == 0:
            # Should never happen (gt is non-empty for N>=1), but be safe.
            return 1.0

        jaccard = len(inter) / len(union)
        score = 2.0 * jaccard - 1.0

        # Clamp for floating-point safety.
        return max(-1.0, min(1.0, score))
