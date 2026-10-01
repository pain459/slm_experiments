from __future__ import annotations
from pathlib import Path
import ast
import re


class CodeIndex:
    def __init__(self, root: str | None = None):
        self.root = Path(root).resolve() if root else None
        self.records: list[dict] = []

    def index(self, root: str | None = None) -> int:
        self.root = Path(root or self.root).resolve()
        self.records = []
        for p in self.root.rglob("*.py"):
            try:
                text = p.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except Exception:
                continue
            lines = text.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    end = getattr(node, "end_lineno", node.lineno)
                    snippet = "\n".join(lines[node.lineno - 1 : end])
                    self.records.append(
                        {
                            "path": str(p.relative_to(self.root)),
                            "symbol": node.name,
                            "kind": node.__class__.__name__,
                            "text": snippet,
                        }
                    )
        return len(self.records)

    def search(self, query: str, k: int = 8) -> list[dict]:
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for r in self.records:
            hay = (r["path"] + " " + r["symbol"] + " " + r["text"]).lower()
            score = sum(3 if t == r["symbol"].lower() else 1 for t in terms if t in hay)
            if score:
                scored.append((score, r))
        return [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
