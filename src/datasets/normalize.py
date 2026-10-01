from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
from src.common.io import read_jsonl, write_jsonl
from src.common.schema import validate_record


def normalize_row(row: dict, source: str) -> dict:
    prompt = row.get("prompt") or row.get("instruction") or row.get("question") or row.get("text") or ""
    response = row.get("response") or row.get("answer") or row.get("solution") or row.get("code") or ""
    rid = str(row.get("id") or f"{source}_{uuid.uuid4().hex[:16]}")
    return {
        "id": rid,
        "source": row.get("source", source),
        "license": row.get("license", "unknown"),
        "language": row.get("language", "python"),
        "category": row.get("category", "python"),
        "skill": row.get("skill", "general"),
        "difficulty": int(row.get("difficulty", 1)),
        "task_type": row.get("task_type", "implementation"),
        "prompt": str(prompt),
        "response": str(response),
        "tests": row.get("tests", []),
        "verified": bool(row.get("verified", False)),
        "metadata": row.get("metadata", {}),
    }


def iter_inputs(path: Path):
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    for f in files:
        for row in read_jsonl(f):
            yield normalize_row(row, f.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rejected", default=None)
    args = ap.parse_args()
    ok, bad = [], []
    for row in iter_inputs(Path(args.input)):
        errs = validate_record(row)
        (bad if errs else ok).append({**row, "validation_errors": errs} if errs else row)
    write_jsonl(args.output, ok)
    if args.rejected:
        write_jsonl(args.rejected, bad)
    print(json.dumps({"accepted": len(ok), "rejected": len(bad)}))

if __name__ == "__main__":
    main()
