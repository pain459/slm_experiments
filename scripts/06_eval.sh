#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?model path}
EVAL=${2:-data/eval_private/eval.jsonl}
OUT=${3:-$MODEL/results.jsonl}
python -m src.evaluation.evaluate --model "$MODEL" --dataset "$EVAL" --output "$OUT"
python -m src.evaluation.report "$OUT"
