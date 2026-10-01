#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
python - <<'PY'
import torch, transformers, datasets, peft
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('CUDA', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
pytest
