from __future__ import annotations
import os, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

BLOCKED_SNIPPETS = ["os.system(", "subprocess.", "socket.", "requests.", "urllib.request", "open('/etc/", 'open("/etc/']


def static_guard(code: str) -> tuple[bool, str]:
    lowered=code.lower()
    for token in BLOCKED_SNIPPETS:
        if token.lower() in lowered:
            return False, f"blocked token: {token}"
    return True, ""


def run_python(code: str, timeout: float = 3.0, cwd: str | None = None) -> RunResult:
    allowed, reason = static_guard(code)
    if not allowed:
        return RunResult(False, 126, "", reason)
    work = Path(cwd) if cwd else Path(tempfile.mkdtemp(prefix="pyagent_"))
    work.mkdir(parents=True, exist_ok=True)
    target=work/"main.py"; target.write_text(code, encoding="utf-8")
    env={"PATH":os.environ.get("PATH", ""), "PYTHONPATH":"", "PYTHONNOUSERSITE":"1", "HOME":str(work)}
    try:
        p=subprocess.run([sys.executable,"-I",str(target)],cwd=work,env=env,text=True,capture_output=True,timeout=timeout)
        return RunResult(p.returncode==0,p.returncode,p.stdout,p.stderr)
    except subprocess.TimeoutExpired as e:
        return RunResult(False,124,e.stdout or "",e.stderr or "timeout",True)
