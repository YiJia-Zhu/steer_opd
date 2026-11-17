#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
export PYTHONPATH="$PWD/exploration/snapshot/src:$PWD/.runtime/easysteer-plugin"
export STEER_OPD_METRIC_WORKERS=8
exec "$STEER_OPD_PYTHON" exploration/launch.py "${1:---status}"
