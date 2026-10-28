"""Read the private D-010 demonstration bank; publish only its sentence text."""

import json
from pathlib import Path

from .categories import CATEGORIES
from .diagnosis import MAX_TEXT_LENGTH


def load_bank(path):
    """Accept all 85 short sentences or none. Never expose source IDs or reference edits."""
    if not path:
        return {}
    try:
        path = Path(path).expanduser()
        if path.stat().st_size > 131072:
            return {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 85:
            return {}
        groups = {category: [] for category in CATEGORIES}
        groups["clean"] = []
        seen = set()
        for row in rows:
            text = row["text"]
            category = row.get("category") if row.get("category") is not None else "clean"
            if (category not in groups or not isinstance(text, str) or not text.strip()
                    or len(text) > MAX_TEXT_LENGTH or len(text.split()) > 15):
                return {}
            text.encode("utf-8")
            key = " ".join(text.casefold().split())
            if key in seen:
                return {}
            seen.add(key)
            groups[category].append(text)
        expected = {**{category: 10 for category in CATEGORIES}, "clean": 15}
        if {category: len(texts) for category, texts in groups.items()} != expected:
            return {}
        return groups
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return {}
