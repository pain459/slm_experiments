from __future__ import annotations
import uuid
from src.context_proxy.context import build_context
from src.context_proxy.compaction import heuristic_summary


class ContextService:
    def __init__(self, store, memory, code_index, provider, cfg):
        self.store = store
        self.memory = memory
        self.code = code_index
        self.provider = provider
        self.cfg = cfg

    def chat(self, session_id: str, current: str, system: str = "You are a careful coding agent.", tools: list[dict] | None = None) -> dict:
        recent = self.store.recent(session_id, 20)
        memories = self.memory.search(current, k=6, where={"session_id": session_id})
        code = self.code.search(current, k=8) if self.code else []
        msgs = build_context(
            system=system,
            current=current,
            recent=recent,
            summary=self.store.summary(session_id),
            state=self.store.state(session_id),
            memories=memories,
            code=code,
            cfg=self.cfg,
        )
        self.store.add(session_id, "user", current)
        message = self.provider.chat(msgs, tools=tools)
        content = message.get("content", "") or ""
        self.store.add(session_id, "assistant", content, {"tool_calls": message.get("tool_calls") or []})
        memory_text = current + "\n" + content
        if message.get("tool_calls"):
            memory_text += "\nTOOL_CALLS: " + str(message["tool_calls"])
        self.memory.add(str(uuid.uuid4()), memory_text, {"session_id": session_id, "type": "conversation"})
        if self.store.count(session_id) >= self.cfg["memory"].get("compact_after_messages", 40):
            all_recent = self.store.recent(session_id, 80)
            self.store.set_summary(session_id, heuristic_summary(all_recent, self.store.summary(session_id)))
        return message
