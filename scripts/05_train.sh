#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_sft --config "${1:-configs/train_v1.yaml}"
