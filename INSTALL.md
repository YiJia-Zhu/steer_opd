# Steering-assisted OPD 环境安装

## 系统和 Python 版本

参考环境为 Linux x86_64、glibc 2.35、CPython 3.12.11。安装 Bash、GNU patch 和 Python 的 `venv` 模块，使 `python3.12` 可以从终端调用。GPU 运行需要支持 CUDA 13.0 的 NVIDIA 驱动以及版本匹配的 `nvcc` 和 CUDA 头文件。

| 虚拟环境 | 用途 | 核心版本 |
| --- | --- | --- |
| `.venv-opd` | veRL 训练、学生 rollout、CPU 编排 | PyTorch 2.13.0+cu130、vLLM 0.32.0、Transformers 5.16.1、Ray 2.55.1 |
| `.venv-easysteer` | 激活提取、干预评分和独立教师引擎 | PyTorch 2.13.0+cu130、vLLM 0.29.0、Transformers 5.16.1、项目固定的 EasySteer 前端与 fork |

## 全新安装

从项目根目录执行。EasySteer 的安装自检会核对 CUDA 工具链。`/usr/local/cuda-13.0` 是默认搜索位置，其他安装位置可用 `STEER_OPD_CUDA_HOME` 指定，例如 `export STEER_OPD_CUDA_HOME="$PWD/.runtime/cuda-13.0"`。

```bash
mkdir -p .cache/pip .runtime/tmp
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.runtime/tmp"

for env_dir in .venv-opd .venv-easysteer; do
    python3.12 -m venv "$env_dir"
    "$env_dir/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' 'torchaudio==2.11.0+cu130'
done

.venv-opd/bin/python -m pip install \
    'vllm==0.32.0' 'transformers==5.16.1' 'ray[default]==2.55.1' \
    'numpy==2.4.6' 'datasets==5.0.0' 'pandas==3.0.3' 'pyarrow==24.0.0' \
    'tensordict==0.10.0' 'hydra-core==1.3.3' 'omegaconf==2.3.1' \
    'accelerate==1.14.0' 'flash-linear-attention==0.5.2' \
    'math-verify==0.9.0' 'matplotlib==3.10.9' 'pytest==9.1.0'

PYTHONPATH="$PWD/src" .venv-opd/bin/python - <<'PY'
from steer_opd.training.verl_entry import bootstrap_source

bootstrap_source()
PY
.venv-opd/bin/python -m pip install -e \
    .runtime/verl_training/verl-61adb0c7eb90730306e7794c2bf248331211e657

python3.12 scripts/bootstrap_easysteer.py --install
```

veRL 安装保留固定提交的源码和校验归档。EasySteer 安装脚本将固定 fork 的 Python 源码覆盖到 vLLM 0.29.0 并注册项目插件。[veRL 源码准备入口](src/steer_opd/training/verl_entry.py#L20) · [EasySteer 安装脚本](scripts/bootstrap_easysteer.py) · [EasySteer 源码清单](ref/easysteer_runtime/manifest.json)

## 复用已有环境

迁移时用 `rsync -a` 保留软链接和可执行权限。如果两个虚拟环境的 Python 指向 `.runtime/python-3.12/`，一起复制该目录。veRL 需要 `.runtime/verl_training/`，EasySteer 需要 `.runtime/easysteer-plugin/`。使用项目内 CUDA 工具链时，同步 `.runtime/cuda-13.0/`。

目标机器采用相同的绝对项目路径，可复用旧命令入口。路径改变时，核对两套环境的 `bin/python*` 链接、`pyvenv.cfg`、激活脚本和 `ray`、`torchrun` 等命令脚本首行。也可以在目标路径按全新安装步骤生成这些文件。迁移二进制包需要目标系统的架构、glibc 和 CUDA 驱动与所用 wheel 兼容。

## 接入模型和数据

将目标机器上现有的模型和数据目录链接到 `assets/models/` 和 `assets/datasets/`。下面的命令以新目录为例，先将两个变量改为本机实际路径。

```bash
steer_models=../huggingface_models
steer_datasets=../huggingface_datasets/huggingface_datasets
mkdir -p assets/models assets/datasets

for model_name in Qwen3.5-35B-A3B Qwen3-4B; do
    ln -srT "$steer_models/$model_name" "assets/models/$model_name"
done
for dataset_name in OpenThoughts3-1.2M MATH-500 AIME2025; do
    ln -srT "$steer_datasets/$dataset_name" "assets/datasets/$dataset_name"
done
```

使用 CompassOPD 时还需接入 `assets/models/Qwen3.5-0.8B`。使用已有题集或静态 SV bank 时，同步运行配置引用的 `outputs/step_0_data/` 或对应的 SV 文件。[默认配置](configs/ablation.json)

## CPU 验收

从项目根目录检查两个 Python 环境。导入检查会核验 veRL 源码路径。挂好资源后可运行静态消融的配置预览。

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" - <<'PY'
from steer_opd.training.verl_entry import activate_verl
import ray
import torch
import vllm

verl_source = activate_verl()
print(torch.__version__, torch.version.cuda, vllm.__version__, ray.__version__)
print(verl_source)
PY
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD/.runtime/easysteer-plugin" \
    .venv-easysteer/bin/python scripts/check_easysteer_runtime.py
bash scripts/run_ablation.sh --check
```
