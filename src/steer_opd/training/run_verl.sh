#!/usr/bin/env bash
# CPU preview by default. --run explicitly starts the formal recipe and validates its first full batch.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
source scripts/runtime_env.sh
if [[ $# == 0 ]]; then set -- --check; fi
exec "$STEER_OPD_PYTHON" -m steer_opd.training.verl_entry "$@"
