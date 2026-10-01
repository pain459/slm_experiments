#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?model path}
REPO=${2:?mini repo path}
python -m src.agent.run_agent --model "$MODEL" --repo "$REPO"
