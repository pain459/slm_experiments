from __future__ import annotations
import shlex
import subprocess
import sys
from pathlib import Path
from src.verification.sandbox import run_python as sandbox_run


class WorkspaceTools:
    """Workspace-scoped coding tools.

    This is a research harness, not a hardened security sandbox. Run untrusted agent
    workloads inside an OS/container sandbox in production.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, p):
        q = (self.root / p).resolve()
        if self.root not in q.parents and q != self.root:
            raise ValueError("path outside workspace")
        return q

    def read_file(self, path: str):
        return {"status": "ok", "content": self._path(path).read_text(encoding="utf-8")}

    def write_file(self, path: str, content: str):
        q = self._path(path)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(content, encoding="utf-8")
        return {"status": "ok"}

    def edit_file(self, path: str, old: str, new: str, count: int = 1):
        q = self._path(path)
        text = q.read_text(encoding="utf-8")
        if old not in text:
            return {"status": "error", "error": "old text not found"}
        q.write_text(text.replace(old, new, count), encoding="utf-8")
        return {"status": "ok", "replacements": min(count, text.count(old))}

    def search_files(self, query: str):
        hits = []
        for p in self.root.rglob("*.py"):
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if query in line:
                        hits.append({"path": str(p.relative_to(self.root)), "line": i, "text": line[:300]})
            except Exception:
                pass
        return {"status": "ok", "hits": hits[:100]}

    def run_python(self, path: str):
        q = self._path(path)
        r = sandbox_run(q.read_text(encoding="utf-8"), cwd=str(self.root))
        return r.__dict__

    def run_tests(self, args: list[str] | None = None):
        cmd = [sys.executable, "-m", "pytest", "-q"] + (args or [])
        p = subprocess.run(cmd, cwd=self.root, text=True, capture_output=True, timeout=30)
        return {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode, "stdout": p.stdout[-5000:], "stderr": p.stderr[-3000:]}

    def run_shell(self, command: list[str] | str, timeout: int = 30):
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            return {"status": "error", "error": "empty command"}
        # Shell=False prevents command-string metacharacter expansion. This remains
        # non-hardened and should be containerized for untrusted trajectories.
        p = subprocess.run(argv, cwd=self.root, text=True, capture_output=True, timeout=timeout, shell=False)
        return {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode, "stdout": p.stdout[-5000:], "stderr": p.stderr[-3000:]}

    def git_diff(self):
        p = subprocess.run(["git", "diff", "--", "."], cwd=self.root, text=True, capture_output=True, timeout=20)
        return {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode, "diff": p.stdout[-12000:], "stderr": p.stderr[-2000:]}

    def call(self, name: str, arguments: dict):
        allowed = {"read_file", "write_file", "edit_file", "search_files", "run_python", "run_tests", "run_shell", "git_diff"}
        if name not in allowed:
            return {"status": "error", "error": "unknown tool"}
        try:
            return getattr(self, name)(**arguments)
        except Exception as e:
            return {"status": "error", "error": repr(e)}
