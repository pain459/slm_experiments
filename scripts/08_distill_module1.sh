#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-rd_v1_pilot}"
OUT="${OUT:-01_distill/outputs/${RUN_ID}}"

python -m src.distill.cli \
  --curriculum 01_distill/curriculum/python.yaml 01_distill/curriculum/dsa.yaml \
  --teachers 01_distill/teachers.yaml \
  --task-types implementation debugging optimization test_generation explanation \
  --samples-per-cell "${SAMPLES_PER_CELL:-1}" \
  --max-cells "${MAX_CELLS:-20}" \
  --run-id "$RUN_ID" \
  --output "$OUT"
