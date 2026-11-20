#!/usr/bin/env bash
# CPU preview by default. Use --run with your chosen physical GPU pair.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
dynamic_src=$("$STEER_OPD_PYTHON" scripts/prepare_training_runtime.py --dynamic-sv)
export PYTHONPATH="$dynamic_src:$PYTHONPATH"
if [[ $# == 0 ]]; then set -- --check; fi
exec "$STEER_OPD_PYTHON" -m steer_opd.training.verl_entry --overlap-metrics "$@"
