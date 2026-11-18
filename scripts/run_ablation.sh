#!/usr/bin/env bash
# Edit this block. GPU IDs are physical nvidia-smi indices; busy GPUs are never used.
GPU_IDS="${GPU_IDS:-0,1}"
SCHEME_IDS="${SCHEME_IDS:-all}"       # all, 1, or comma-separated IDs such as 1,3,4
TEACHER_ABLATION="${TEACHER_ABLATION:-0}" # 1 adds teacher-reader S-T schemes 5..8; default preserves the original four.
STUDENT_BASELINE_RUN="${STUDENT_BASELINE_RUN:-}" # Existing step3 parameter root for teacher-only supplements.
TEACHER_LAYER="${TEACHER_LAYER:-29}" # 0-based, 75% of the teacher's 40 blocks.
# 1 T-S/all/global       2 T-S/all/per_question
# 3 T-S/static/global    4 T-S/static/per_question
STRENGTH_POLICIES="${STRENGTH_POLICIES:-fixed}" # Overlap uses four fixed schemes; annealing is a future training ablation.
RUN_NAME="${RUN_NAME:-ablation}"       # Experiment category: ablation / param / compar.
PARAM_NAME="${PARAM_NAME:-}"            # Empty = automatic key-parameter label; never a run category.
QUESTION_COUNT="${QUESTION_COUNT:-1000}"
INPUT_DATA_DIR="${INPUT_DATA_DIR:-outputs/step_0_data/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n${QUESTION_COUNT}_slerp0.1_layer26_seed42_v1}"
if [[ "$TEACHER_ABLATION" == 1 ]]; then
    CONFIRM_TOP_N="${CONFIRM_TOP_N:-8}"
else
    CONFIRM_TOP_N="${CONFIRM_TOP_N:-4}"
fi
STOP_AFTER="${STOP_AFTER:-panel}" # Four zero-update overlap comparisons; source_overlap skips SV.
ABLATION_PHASE="${ABLATION_PHASE:-overlap}" # Only overlap is implemented; param/baseline are planned.
OPERATOR="${OPERATOR:-slerp}"          # slerp or raw additive (teacher scheme 5 only)
ALPHA="${ALPHA:-0.1}"                 # Fixed Slerp alpha for the later SV panel
LAYER="${LAYER:-26}"                  # 0-based block output; 26 = 75% of Qwen3-4B's 36 blocks
GATE_METRIC="${GATE_METRIC:-overlap}"  # overlap or P_overlap; both are always recorded
RANK_METRIC="${RANK_METRIC:-overlap}"
GATE_TARGET_RATE="${GATE_TARGET_RATE:-0.2}"
TEACHER_TP="${TEACHER_TP:-2}"         # 1 or 2; pool must contain at least this many GPUs
GPU_MEMORY="${GPU_MEMORY:-0.75}"
WAIT_SECONDS="${WAIT_SECONDS:-3600}"  # Maximum wait per GPU stage; 0 = fail if unavailable
REFERENCE_MAX_TOKENS="${REFERENCE_MAX_TOKENS:-16384}"
SOURCE_MAX_TOKENS="${SOURCE_MAX_TOKENS:-16384}"
SV_MAX_CONTEXT_LEN="${SV_MAX_CONTEXT_LEN:-2048}" # SV-training: read prompt+answer up to this limit, pool all retained answer tokens.
REFERENCE_TEMPERATURE="${REFERENCE_TEMPERATURE:-1.0}"
REFERENCE_TOP_P="${REFERENCE_TOP_P:-1.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-2.0}" # Applies to teacher/student generation, including SV source answers.
# CONFIRM_TOP_N defaults to every selected family; both splits compare all eight when enabled.
CANDIDATE_TENSOR_FORMAT="${CANDIDATE_TENSOR_FORMAT:-${CANDIDATE_TENSOR_COMPRESSION:-parquet_zstd}}" # parquet_zstd, npz_compressed, npz, pt, pt_gzip; old none/gzip accepted.
HF_PARITY="${HF_PARITY:-0}"           # 0 = EasySteer-only startup; 1 = optional HF numerical comparison
# Both metrics use raw Top16; P_overlap never renormalizes the retained mass.
# Seed stays 42. No sequence-count or CUDA-graph-size limits are imposed.

set -euo pipefail
if (( $# > 1 )); then
    echo "Usage: bash scripts/run_ablation.sh [--check|--run|--status]" >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
exec "$STEER_OPD_PYTHON" scripts/launch_ablation.py "${1:---check}" \
    --gpu-ids "$GPU_IDS" --schemes "$SCHEME_IDS" --run-name "$RUN_NAME" \
    --param-name "$PARAM_NAME" --question-count "$QUESTION_COUNT" --input-data-dir "$INPUT_DATA_DIR" \
    --teacher-ablation "$TEACHER_ABLATION" --teacher-layer "$TEACHER_LAYER" --student-baseline-run "$STUDENT_BASELINE_RUN" \
    --phase "$ABLATION_PHASE" \
    --stop-after "$STOP_AFTER" \
    --strength-policies "$STRENGTH_POLICIES" \
    --operator "$OPERATOR" --alpha "$ALPHA" --layer "$LAYER" --gate-metric "$GATE_METRIC" --rank-metric "$RANK_METRIC" \
    --gate-target-rate "$GATE_TARGET_RATE" --teacher-tp "$TEACHER_TP" --gpu-memory "$GPU_MEMORY" \
    --wait-seconds "$WAIT_SECONDS" --reference-max-tokens "$REFERENCE_MAX_TOKENS" \
    --source-max-tokens "$SOURCE_MAX_TOKENS" --reference-temperature "$REFERENCE_TEMPERATURE" \
    --sv-max-context-len "$SV_MAX_CONTEXT_LEN" \
    --reference-top-p "$REFERENCE_TOP_P" --presence-penalty "$PRESENCE_PENALTY" \
    --candidate-tensor-format "$CANDIDATE_TENSOR_FORMAT" --confirm-top-n "$CONFIRM_TOP_N" --hf-parity "$HF_PARITY"
