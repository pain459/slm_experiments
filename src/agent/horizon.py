from __future__ import annotations

def classify_horizon(tool_calls: int) -> str:
    if tool_calls <= 3: return 'H1'
    if tool_calls <= 8: return 'H2'
    if tool_calls <= 20: return 'H3'
    return 'H4'
