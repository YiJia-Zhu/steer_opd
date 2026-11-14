#!/usr/bin/env bash
# Edit this block. GPU IDs are physical nvidia-smi indices; busy GPUs are never used.
GPU_IDS="${GPU_IDS:-2,7}"
SCHEME_IDS="${SCHEME_IDS:-all}"       # all, 1, or comma-separated IDs such as 1,4,8
# 1 T-S/all/global       2 T-S/all/per_question
# 3 T-S/static/global    4 T-S/static/per_question
# 5 T-Ref/all/global     6 T-Ref/all/per_question
# 7 T-Ref/static/global  8 T-Ref/static/per_question
RUN_NAME="${RUN_NAME:-static_panel_v1}" # Change this for a different experiment configuration.
ALPHA="${ALPHA:-0.1}"                 # Slerp interpolation fraction, 0..1
LAYER="${LAYER:-26}"                  # 0-based block output; 26 = 75% of Qwen3-4B's 36 blocks
GATE_METRIC="${GATE_METRIC:-overlap}"  # overlap or P_overlap; both are always recorded
RANK_METRIC="${RANK_METRIC:-overlap}"
GATE_TARGET_RATE="${GATE_TARGET_RATE:-0.2}"
TEACHER_TP="${TEACHER_TP:-2}"         # 1 or 2; pool must contain at least this many GPUs
GPU_MEMORY="${GPU_MEMORY:-0.75}"
WAIT_SECONDS="${WAIT_SECONDS:-3600}"  # Maximum wait per GPU stage; 0 = fail if unavailable
REFERENCE_MAX_TOKENS="${REFERENCE_MAX_TOKENS:-12288}"
SOURCE_MAX_TOKENS="${SOURCE_MAX_TOKENS:-7168}"
REFERENCE_TEMPERATURE="${REFERENCE_TEMPERATURE:-0.7}"
REFERENCE_TOP_P="${REFERENCE_TOP_P:-0.95}"
CONFIRM_TOP_N="${CONFIRM_TOP_N:-2}"   # Automatically limited to selected scheme count
# Seed stays 42. No sequence-count or CUDA-graph-size limits are imposed.

set -euo pipefail
if (( $# > 1 )); then
    echo "Usage: bash scripts/run_static_panel.sh [--check|--run|--status]" >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
exec "$STEER_OPD_PYTHON" scripts/launch_static_panel.py "${1:---check}" \
    --gpu-ids "$GPU_IDS" --schemes "$SCHEME_IDS" --run-name "$RUN_NAME" \
    --alpha "$ALPHA" --layer "$LAYER" --gate-metric "$GATE_METRIC" --rank-metric "$RANK_METRIC" \
    --gate-target-rate "$GATE_TARGET_RATE" --teacher-tp "$TEACHER_TP" --gpu-memory "$GPU_MEMORY" \
    --wait-seconds "$WAIT_SECONDS" --reference-max-tokens "$REFERENCE_MAX_TOKENS" \
    --source-max-tokens "$SOURCE_MAX_TOKENS" --reference-temperature "$REFERENCE_TEMPERATURE" \
    --reference-top-p "$REFERENCE_TOP_P" --confirm-top-n "$CONFIRM_TOP_N"
