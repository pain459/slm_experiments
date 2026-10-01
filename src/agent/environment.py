from __future__ import annotations
import json, shutil, tempfile
from pathlib import Path

class RepoEnvironment:
    def __init__(self, template: str|Path):
        self.template=Path(template)
        self.tmp=Path(tempfile.mkdtemp(prefix="agent_repo_"))
        shutil.copytree(self.template,self.tmp,dirs_exist_ok=True)
    @property
    def task(self):
        p=self.tmp/"task.json"
        return json.loads(p.read_text()) if p.exists() else {"task":"Solve the repository task."}
    def cleanup(self): shutil.rmtree(self.tmp,ignore_errors=True)
