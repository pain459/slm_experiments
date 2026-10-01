#!/usr/bin/env bash
set -euo pipefail
python -m src.curriculum_builder.build --inputs "$@" --output-dir 03_curriculum/outputs/dataset_v1
