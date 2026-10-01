from __future__ import annotations
import argparse
from src.common.io import read_jsonl

def ids(path): return {r["id"] for r in read_jsonl(path)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",required=True); ap.add_argument("--eval",required=True); args=ap.parse_args()
    a,b=ids(args.train),ids(args.eval); overlap=a&b
    if overlap: raise SystemExit(f"FAIL: {len(overlap)} overlapping IDs")
    print(f"PASS: train={len(a)} eval={len(b)} overlap=0")
if __name__=="__main__": main()
