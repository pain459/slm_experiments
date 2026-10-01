#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?model id/path}
EVAL=${2:-data/eval_private/eval.jsonl}
mkdir -p experiments/baseline_v0
python -m src.evaluation.evaluate --model "$MODEL" --dataset "$EVAL" --output experiments/baseline_v0/results.jsonl
python -m src.evaluation.report experiments/baseline_v0/results.jsonl
