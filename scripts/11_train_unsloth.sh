#!/usr/bin/env bash
set -euo pipefail
python -m src.unsloth_train.train --config "${1:-04_train/configs/unsloth_v1.yaml}"
