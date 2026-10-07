#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${HYPERCLOVA_PYTHON:-python}"

# The client runs on the host; GPU assignments belong to OmniServe's Compose
# configuration. Dataset, output and endpoint defaults live in the runner.
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/run_omnistorybench.py" "$@"
