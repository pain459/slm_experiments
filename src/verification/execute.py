from __future__ import annotations
import argparse, json
from src.common.io import read_jsonl, write_jsonl
from src.verification.tests import verify_solution


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--accepted",required=True)
    ap.add_argument("--rejected",required=True); ap.add_argument("--timeout",type=float,default=4.0); args=ap.parse_args()
    ok=[]; bad=[]
    for row in read_jsonl(args.input):
        result=verify_solution(row.get("response",""), row.get("tests",[]), args.timeout)
        meta={**row.get("metadata",{}),"verification":{"returncode":result.returncode,"stdout":result.stdout[-2000:],"stderr":result.stderr[-2000:],"timed_out":result.timed_out}}
        if result.ok:
            ok.append({**row,"verified":True,"metadata":meta})
        else:
            bad.append({**row,"verified":False,"metadata":meta})
    write_jsonl(args.accepted,ok); write_jsonl(args.rejected,bad)
    print(json.dumps({"accepted":len(ok),"rejected":len(bad),"rate":len(ok)/max(1,len(ok)+len(bad))}))
if __name__=="__main__": main()
