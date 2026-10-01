#!/usr/bin/env bash
set -euo pipefail
TEACHER=${1:?teacher model id/path}
SEEDS=${2:-data/normalized/seeds_dedup.jsonl}
COUNT=${3:-100}
python -m src.generation.generate_batch --teacher "$TEACHER" --seeds "$SEEDS" --count "$COUNT" --output data/synthetic/batch_001.jsonl --rejected data/rejected/generation_001.jsonl
