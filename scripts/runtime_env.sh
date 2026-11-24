#!/usr/bin/env bash
# Resolve paths relative to this script so a moved checkout works from any directory.
STEER_OPD_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || return 1
export STEER_OPD_PYTHON="${STEER_OPD_PYTHON:-${STEER_OPD_ROOT}/.venv-opd/bin/python}"
if [[ ! -x "$STEER_OPD_PYTHON" ]]; then
    printf 'Cannot use Python at %s. Repair .venv-opd or set STEER_OPD_PYTHON to a working environment.\n' "$STEER_OPD_PYTHON" >&2
    return 1
fi
STEER_OPD_RUNTIME_PREFIX=$(cd -- "$(dirname -- "$STEER_OPD_PYTHON")/.." && pwd) || return 1
export PYTHONPATH="${STEER_OPD_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export HF_HOME="${STEER_OPD_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${STEER_OPD_ROOT}/.cache/datasets"
export XDG_CACHE_HOME="${STEER_OPD_ROOT}/.cache"
export TORCH_HOME="${STEER_OPD_ROOT}/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${STEER_OPD_ROOT}/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="${STEER_OPD_ROOT}/.cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="${STEER_OPD_ROOT}"
export TRITON_CACHE_DIR="${STEER_OPD_ROOT}/.cache/triton"
export CUDA_CACHE_PATH="${STEER_OPD_ROOT}/.cache/cuda"
# Match the pinned torch/vLLM cu130 build when the host has a different toolkit.
export CUDA_HOME="${STEER_OPD_RUNTIME_PREFIX}/lib/python3.12/site-packages/nvidia/cu13"
if [[ -n "${STEER_OPD_CUDA_HOME:-}" ]]; then
    CUDA_HOME="$STEER_OPD_CUDA_HOME"
elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda-13.0
elif [[ -x "${STEER_OPD_ROOT}/.runtime/cuda-13.0/bin/nvcc" ]]; then
    export STEER_OPD_CUDA_HOME="${STEER_OPD_ROOT}/.runtime/cuda-13.0"
    CUDA_HOME="$STEER_OPD_CUDA_HOME"
fi
export PATH="${STEER_OPD_RUNTIME_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export PIP_CACHE_DIR="${STEER_OPD_ROOT}/.cache/pip"
export UV_CACHE_DIR="${STEER_OPD_ROOT}/.cache/uv"
export TMPDIR="${STEER_OPD_ROOT}/.runtime/tmp"
export RAY_TMPDIR="${STEER_OPD_ROOT}/r"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export STEER_OPD_METRIC_WORKERS="${STEER_OPD_METRIC_WORKERS:-16}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# Full-pipeline requests bind the launcher's own bytes. Select the training source
# here for those launchers and interactive use. Other scripted stages keep their
# own environment; a custom training wrapper can source this file with --training.
case "${BASH_SOURCE[1]:-}" in
    ''|run_full_pipeline.sh|*/run_full_pipeline.sh|run_verl.sh|*/run_verl.sh) steer_select_training=1 ;;
    *) if [[ "${1:-}" == --training ]]; then steer_select_training=1; else steer_select_training=0; fi ;;
esac
if [[ "$steer_select_training" == 1 ]]; then
    steer_training_src=$("$STEER_OPD_PYTHON" "$STEER_OPD_ROOT/scripts/prepare_training_runtime.py") || return 1
    export PYTHONPATH="$steer_training_src:$PYTHONPATH"
    unset steer_training_src
fi
unset steer_select_training
