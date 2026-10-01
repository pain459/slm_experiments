#!/usr/bin/env bash
set -euo pipefail
python -m src.judge.run --input "${1:?input jsonl}" --output-dir "${2:-02_judge/outputs/run_v1}"
