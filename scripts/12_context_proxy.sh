#!/usr/bin/env bash
set -euo pipefail
python -m src.context_proxy.server "$@"
