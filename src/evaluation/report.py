from __future__ import annotations
import argparse, json
from src.common.io import read_jsonl
from src.evaluation.metrics import summarize

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("files",nargs="+"); ap.add_argument("--compare",action="store_true"); args=ap.parse_args()
    reports=[]
    for f in args.files:
        s=summarize(list(read_jsonl(f))); reports.append((f,s)); print(f"\n{f}\n"+json.dumps(s,indent=2))
    if len(reports)>1:
        base=reports[0][1]["pass_rate"]
        print("\nDeltas vs first:")
        for f,s in reports[1:]: print(f"{f}: {(s['pass_rate']-base)*100:+.2f} pp")
if __name__=="__main__": main()
