from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
import yaml


def create_app(config_path="06_context_proxy/config.yaml", repo_path=None):
    from fastapi import FastAPI, Header
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, ConfigDict
    from src.context_proxy.store import ConversationStore
    from src.context_proxy.memory import MemoryIndex
    from src.context_proxy.code_index import CodeIndex
    from src.context_proxy.provider import OllamaProvider
    from src.context_proxy.service import ContextService

    cfg = yaml.safe_load(Path(config_path).read_text())
    store = ConversationStore(cfg["memory"]["sqlite_path"])
    mem = MemoryIndex(cfg["memory"]["chroma_path"], cfg["memory"]["collection"])
    code = CodeIndex(repo_path) if repo_path else CodeIndex()
    if repo_path:
        code.index()
    provider = OllamaProvider(cfg["model"]["base_url"], cfg["model"]["name"])
    svc = ContextService(store, mem, code, provider, cfg)
    app = FastAPI(title="Python Agent Context Proxy")

    class ChatBody(BaseModel):
        model_config = ConfigDict(extra="allow")
        model: str | None = None
        messages: list[dict]
        stream: bool = False
        user: str | None = None
        tools: list[dict] | None = None
        tool_choice: object | None = None

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": cfg["model"]["name"], "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(body: ChatBody, x_session_id: str | None = Header(default=None)):
        sid = x_session_id or body.user or "default"
        current = next((m.get("content", "") for m in reversed(body.messages) if m.get("role") == "user"), "")
        system = "\n".join(str(m.get("content", "")) for m in body.messages if m.get("role") == "system") or "You are a careful coding agent."
        message = svc.chat(sid, current, system, tools=body.tools)
        msg = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            # Ollama's tool-call shape is close to OpenAI's; normalize IDs/types.
            normalized = []
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                normalized.append({
                    "id": tc.get("id") or "call_" + uuid.uuid4().hex[:16],
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            msg["tool_calls"] = normalized

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": cfg["model"]["name"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "session_id": sid,
        }
        if not body.stream:
            return response

        def events():
            delta = {"role": "assistant"}
            if msg.get("content"):
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = msg["tool_calls"]
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": response["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield "data: " + json.dumps(chunk) + "\n\n"
            done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": response["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
            yield "data: " + json.dumps(done) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="06_context_proxy/config.yaml")
    ap.add_argument("--repo")
    args = ap.parse_args()
    app, cfg = create_app(args.config, args.repo)
    import uvicorn
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])


if __name__ == "__main__":
    main()
