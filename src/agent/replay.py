from __future__ import annotations
import argparse, json, shutil, tempfile
from pathlib import Path
from src.agent.tools import WorkspaceTools

def replay(repo: str|Path, trajectory: list[dict]) -> dict:
    src=Path(repo); tmp=Path(tempfile.mkdtemp(prefix='replay_')); shutil.copytree(src,tmp,dirs_exist_ok=True); tools=WorkspaceTools(tmp); valid=0
    try:
        for step in trajectory:
            call=step.get('tool_call') or step.get('call')
            if not call: continue
            tools.call(call.get('tool',''),call.get('arguments',{})); valid += 1
        final=tools.run_tests([]); return {'ok':final.get('returncode')==0,'valid_tool_calls':valid,'final_test':final}
    finally: shutil.rmtree(tmp,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--trajectory',required=True); args=ap.parse_args(); obj=json.loads(Path(args.trajectory).read_text()); print(json.dumps(replay(args.repo,obj['trajectory']),indent=2))
if __name__=='__main__': main()
