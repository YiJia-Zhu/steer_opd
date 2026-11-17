#!/usr/bin/env bash
# Source from the project root. Do not alter shared asset directories.
STEER_OPD_ROOT=/home/zhenning/data/working_tmp/steer_opd
# Isolate the validated experiment dependencies from the shared Conda environment.
if [[ -x "${STEER_OPD_ROOT}/.venv-opd/bin/python" ]]; then
    STEER_OPD_RUNTIME_PREFIX="${STEER_OPD_ROOT}/.venv-opd"
else
    STEER_OPD_RUNTIME_PREFIX=/home/zhenning/data/miniconda3/envs/verl
fi
export STEER_OPD_PYTHON="${STEER_OPD_RUNTIME_PREFIX}/bin/python"
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
# Match the pinned torch/vLLM cu130 build; the shell's shared default is CUDA12.8.
export CUDA_HOME="${STEER_OPD_RUNTIME_PREFIX}/lib/python3.12/site-packages/nvidia/cu13"
if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then export CUDA_HOME=/usr/local/cuda-13.0; fi
export PATH="${STEER_OPD_RUNTIME_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export PIP_CACHE_DIR="${STEER_OPD_ROOT}/.cache/pip"
export UV_CACHE_DIR="${STEER_OPD_ROOT}/.cache/uv"
export TMPDIR="${STEER_OPD_ROOT}/.runtime/tmp"
export RAY_TMPDIR="${STEER_OPD_ROOT}/.runtime/ray"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export STEER_OPD_METRIC_WORKERS="${STEER_OPD_METRIC_WORKERS:-16}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
