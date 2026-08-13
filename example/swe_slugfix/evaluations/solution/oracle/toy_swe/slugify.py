from __future__ import annotations

import re


def slugify(text: str) -> str:
    """Return a stable slug for a short phrase."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return slug.strip("-")
