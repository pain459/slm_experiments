from __future__ import annotations

def approx_tokens(text: str) -> int: return max(1,len(text)//4)
def clip_text(text: str,budget: int) -> str:
    if approx_tokens(text)<=budget: return text
    return text[-budget*4:]
