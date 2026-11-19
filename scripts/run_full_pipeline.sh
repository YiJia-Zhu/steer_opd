#!/usr/bin/env bash
# 修改下面的参数，或用同名环境变量覆盖。无参数 / --check 仅 CPU 预览；--run 执行全流程。
# 流程：师生准备回答 → SV 提取 → 单组 overlap 核验 → OPD 训练及裸学生评测。

# ==================== SV 与注入超参数 ====================
SV_GRANULARITY="${SV_GRANULARITY:-global}" # global：全部有效题目的差分均值。当前 OPD 仅支持 global；per_question 仅在 overlap 入口可用。
INJECTION_MODE="${INJECTION_MODE:-all}"   # all：教师的全部回答预测位置。当前 OPD 仅支持 all；static_overlap 仅在 overlap 入口可用。
OPERATOR="${OPERATOR:-slerp}"             # slerp：保持 hidden 范数的球面插值；additive：h + alpha × 原始全局均值差分，无额外范数缩放。
ALPHA="${ALPHA:-0.1}"                    # slerp 可选 [0,1]；additive 可选非负有限数。0 为零干预对照；全程固定强度，不退火。
TEACHER_LAYER="${TEACHER_LAYER:-29}"      # 教师 0-based block：0..39，例如 19 / 29 / 39；提取与注入同层，39 在 final norm 之前。
SV_MAX_CONTEXT_LEN="${SV_MAX_CONTEXT_LEN:-2048}" # 提取时 prompt+回答的总 token 上限，整数 >=2；只平均其中保留的回答 token。
# 方向固定为教师空间 S−T：H_T(学生回答) − H_T(教师回答)；教师冻结，训练只更新学生。

# ==================== 训练与评测超参数 ====================
QUESTION_COUNT="${QUESTION_COUNT:-1024}"  # 当前统一训练池固定1024题；保留原1000题并同源补充24题，无Select/Confirm划分。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" # 当前正式训练每step32题，每题4条回答，共128条；1024题一轮32个step。
TOTAL_STEPS="${TOTAL_STEPS:-64}"          # 正整数，最多64；64步正好完整遍历两轮，改小可提前停止。
EVAL_EVERY="${EVAL_EVERY:-16}"            # 正整数；默认step0/16/32/48/64评测，提前结束时最后一步也评测。
SAVE_EVERY="${SAVE_EVERY:-$EVAL_EVERY}"    # 正整数；默认与评测对齐，每16步及最终步保存模型。
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all_models_latest_state_v1}" # all_models_latest_state_v1：各保存点的模型都保留，恢复状态仅留最新；latest_full_v1：只留最新完整ckpt。

# ==================== GPU、采样与监控 ====================
GPU_IDS="${GPU_IDS:-2,3}"                 # 恰好两张物理卡，例如 0,1 或 2,3；准备和训练共用此池，繁忙时排队。
WAIT_SECONDS="${WAIT_SECONDS:-3600}"      # 每次 GPU 准入最多等待的秒数，>=0；0 为不等待。
PREP_GPU_MEMORY="${PREP_GPU_MEMORY:-0.75}" # 准备阶段引擎的显存比例，(0,1)；训练沿用既有独立显存配置。
PRESENCE_PENALTY="${PRESENCE_PENALTY:-2.0}" # 准备生成、训练生成及评测统一使用，[-2,2]；0 为无惩罚，评分/loss 仍用原始概率。
REFERENCE_MAX_TOKENS="${REFERENCE_MAX_TOKENS:-16384}" # SV 准备时教师回答长度上限，正整数。
SOURCE_MAX_TOKENS="${SOURCE_MAX_TOKENS:-16384}"       # SV 准备时初始学生回答长度上限，正整数。
OVERLAP_METRICS="${OVERLAP_METRICS:-1}"   # 1：每个训练 step 记录 overlap/P_overlap 标量；0：关闭训练内 overlap，准备阶段仍计算双指标。
HF_PARITY="${HF_PARITY:-0}"               # 0：EasySteer 师生 GPU 验收；1：额外做 HF 数值对照。
CANDIDATE_TENSOR_FORMAT="${CANDIDATE_TENSOR_FORMAT:-parquet_zstd}" # overlap 产物格式：parquet_zstd / npz_compressed / npz / pt / pt_gzip。
# 当前正式训练配方：最多64 steps，32题/step，每题4条（seed42/43/44/45），AdamW lr=1e-6。
# 训练回答上限16384；temperature=1、top_p=1、top_k=-1。准备阶段每请求 seed42。
# 自动在 step0、每16步及最后一步评测裸学生：MATH-500 500题 + AIME2025 30题，每题1条。

# ==================== 数据、身份与保存位置 ====================
INPUT_DATA_DIR="${INPUT_DATA_DIR:-}"     # 留空：使用本地 n1024_all_train_seed42_v1；首次 --run 会先在CPU冻结题集，也可 --prepare-data 单独准备。
BASE_QUESTION_SOURCE="${BASE_QUESTION_SOURCE:-}" # 留空：优先本地原1000题 questions_source.json，缺失时只读 output_remote 原1000题。
# 全部1024题同时用于SV、overlap和OPD；MATH-500/AIME2025用于独立评测，新增24题也排除这两个评测集。
PARAM_NAME="${PARAM_NAME:-}"             # 留空：按准备超参数自动命名；改参数须用新身份。同参数重跑 --run 自动复用/恢复。
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-}"     # 留空：opd_<PARAM_NAME>；更换训练监控/保存位置等设置时使用新名字。
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs}" # 训练输出根：默认项目 outputs，即 /home/zhenning/data/working_tmp/steer_opd/outputs；可用同名环境变量覆盖。
# minimal_v1 清理过程数据；默认保留step16/32/48/64全部模型，优化器/scheduler/RNG/dataloader只保留最新完整一份。
# overlap状态实时覆盖overlap/latest.json；loss/overlap/准确率等标量曲线继续保留。

set -euo pipefail
if (( $# > 1 )) || [[ "${1:---check}" != --check && "${1:---check}" != --run && "${1:---check}" != --prepare-data && "${1:---check}" != --help ]]; then
    echo "用法：bash scripts/run_full_pipeline.sh [--check|--run|--prepare-data|--help]" >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
exec "$STEER_OPD_PYTHON" scripts/run_full_pipeline.py "${1:---check}" \
    --sv-granularity "$SV_GRANULARITY" --injection-mode "$INJECTION_MODE" \
    --operator "$OPERATOR" --alpha "$ALPHA" --teacher-layer "$TEACHER_LAYER" \
    --sv-max-context-len "$SV_MAX_CONTEXT_LEN" --gpu-ids "$GPU_IDS" \
    --question-count "$QUESTION_COUNT" --batch-size "$TRAIN_BATCH_SIZE" \
    --total-steps "$TOTAL_STEPS" --evaluation-freq "$EVAL_EVERY" \
    --save-freq "$SAVE_EVERY" --checkpoint-retention "$CHECKPOINT_RETENTION" \
    --wait-seconds "$WAIT_SECONDS" --prep-gpu-memory "$PREP_GPU_MEMORY" \
    --presence-penalty "$PRESENCE_PENALTY" --reference-max-tokens "$REFERENCE_MAX_TOKENS" \
    --source-max-tokens "$SOURCE_MAX_TOKENS" --overlap-metrics "$OVERLAP_METRICS" \
    --hf-parity "$HF_PARITY" --candidate-tensor-format "$CANDIDATE_TENSOR_FORMAT" \
    --input-data-dir "$INPUT_DATA_DIR" --base-question-source "$BASE_QUESTION_SOURCE" --param-name "$PARAM_NAME" \
    --train-run-name "$TRAIN_RUN_NAME" --train-output-root "$TRAIN_OUTPUT_ROOT"
