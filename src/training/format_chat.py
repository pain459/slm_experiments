from __future__ import annotations

def record_to_messages(row: dict) -> list[dict[str,str]]:
    return [
        {"role":"system","content":"You are a precise Python coding assistant."},
        {"role":"user","content":row["prompt"]},
        {"role":"assistant","content":row.get("response","")},
    ]
