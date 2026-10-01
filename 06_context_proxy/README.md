# Module 6: Context Virtualization Proxy

The frontend/OpenCode talks to this OpenAI-compatible proxy, not directly to Ollama. The proxy stores raw history in SQLite, semantic memory in ChromaDB when installed (lexical fallback otherwise), keeps structured session state, indexes Python symbols, compacts old conversation, retrieves relevant history/code and constructs a bounded working set before every Ollama call.

```bash
pip install -e '.[proxy]'
ollama serve
python -m src.context_proxy.server --repo /path/to/project
```

Then point OpenCode at `http://127.0.0.1:8787/v1` using the included example config.
