from __future__ import annotations
from src.context_proxy.budget import clip_text


def build_context(*, system: str, current: str, recent: list[dict], summary: str, state: dict,
                  memories: list[dict], code: list[dict], cfg: dict) -> list[dict]:
    c = cfg["context"]
    parts = []
    if summary:
        parts.append("LONG_TERM_SUMMARY\n" + clip_text(summary, c["summary_budget"]))
    if state:
        parts.append("CURRENT_STATE\n" + clip_text(str(state), 1200))
    if memories:
        parts.append("RETRIEVED_MEMORY\n" + clip_text("\n\n".join(m["text"] for m in memories), c["memory_budget"]))
    if code:
        code_text = "\n\n".join(f"# {r['path']}::{r['symbol']}\n{r['text']}" for r in code)
        parts.append("RELEVANT_CODE\n" + clip_text(code_text, c["code_budget"]))
    rec = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    parts.append("RECENT_CONVERSATION\n" + clip_text(rec, c["recent_budget"]))
    total_context_budget = c["system_budget"] + c["summary_budget"] + c["memory_budget"] + c["code_budget"] + c["recent_budget"]
    sys = clip_text(system + "\n\n" + "\n\n".join(parts), total_context_budget)
    return [{"role": "system", "content": sys}, {"role": "user", "content": current}]
