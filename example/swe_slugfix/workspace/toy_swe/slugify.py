from __future__ import annotations


def slugify(text: str) -> str:
    """Return a simple slug for a short phrase.

    Buggy on purpose: it only handles spaces, so punctuation and repeated
    separators still leak through.
    """
    return text.strip().lower().replace(" ", "-")
