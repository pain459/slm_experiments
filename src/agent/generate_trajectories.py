from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from src.common.io import write_jsonl


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--repos",required=True); ap.add_argument("--output",required=True)
    args=ap.parse_args(); rows=[]
    for repo in sorted(Path(args.repos).iterdir()):
        if not repo.is_dir(): continue
        p=subprocess.run([sys.executable,"-m","src.agent.run_agent","--model",args.model,"--repo",str(repo)],text=True,capture_output=True)
        try:
            obj=json.loads(p.stdout); success=obj.get("final_test",{}).get("returncode")==0
            if success: rows.append({"id":repo.name,"task_type":"agent","category":"agent","skill":"repo_edit","difficulty":2,
                                     "source":"teacher_agent","language":"python","prompt":json.loads((repo/"task.json").read_text())["task"],
                                     "response":json.dumps(obj["trajectory"]),"tests":[],"verified":True,"metadata":{"repo":repo.name}})
        except Exception: pass
    write_jsonl(args.output,rows); print(f"saved={len(rows)}")
if __name__=="__main__": main()
