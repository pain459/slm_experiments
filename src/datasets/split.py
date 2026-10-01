from __future__ import annotations
import argparse, random
from pathlib import Path
from src.common.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validation-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows=[]
    for pattern in args.inputs:
        paths=list(Path().glob(pattern)) if any(x in pattern for x in "*?[") else [Path(pattern)]
        for p in paths:
            if p.exists(): rows.extend(read_jsonl(p))
    random.Random(args.seed).shuffle(rows)
    n=max(1, int(len(rows)*args.validation_ratio)) if rows else 0
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/"validation.jsonl", rows[:n]); write_jsonl(out/"train.jsonl", rows[n:])
    print(f"train={len(rows)-n} validation={n}")
if __name__ == "__main__": main()
