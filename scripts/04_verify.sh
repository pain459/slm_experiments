#!/usr/bin/env bash
set -euo pipefail
IN=${1:-data/synthetic/batch_001.jsonl}
python -m src.verification.execute --input "$IN" --accepted data/verified/batch_001.jsonl --rejected data/rejected/batch_001.jsonl
