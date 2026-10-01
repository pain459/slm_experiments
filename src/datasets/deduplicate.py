from __future__ import annotations
import argparse, hashlib, re
from src.common.io import read_jsonl, write_jsonl


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def fingerprint(row: dict) -> str:
    base = norm(row.get("prompt", "")) + "\n" + norm(row.get("response", ""))
    return hashlib.sha256(base.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ids, fps, out = set(), set(), []
    for row in read_jsonl(args.input):
        fp = fingerprint(row)
        if row["id"] in ids or fp in fps:
            continue
        ids.add(row["id"]); fps.add(fp); out.append(row)
    write_jsonl(args.output, out)
    print(f"kept={len(out)}")

if __name__ == "__main__": main()
