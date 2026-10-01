from __future__ import annotations


def heuristic_summary(messages: list[dict], previous: str = "") -> str:
    """Deterministic fallback summary; replace with model summarization later."""
    selected = []
    for m in messages:
        text = m.get("content", "").strip()
        if any(x in text.lower() for x in ("decide", "require", "error", "failed", "fix", "todo", "must", "changed", "test")):
            selected.append(f"{m.get('role')}: {text[:700]}")
    return (previous + "\n" + "\n".join(selected[-30:])).strip()[-12000:]
