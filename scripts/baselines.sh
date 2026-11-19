#!/usr/bin/env bash
# 按列表从左到右串行执行。只跑一种就保留一个名字，也可用环境变量 METHODS 覆盖。
# 示例  METHODS=compassopd bash scripts/baselines.sh --check
BASELINES=(vanilla compassopd)

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"  # 每步32题，每题4条
TOTAL_STEPS="${TOTAL_STEPS:-64}"          # 1..64，默认两轮
EVAL_EVERY="${EVAL_EVERY:-16}"            # step 0、每16步及末步评测裸学生
SAVE_EVERY="${SAVE_EVERY:-16}"            # 每16步及末步保存模型
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all_models_latest_state_v1}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-2.0}"
OVERLAP_METRICS="${OVERLAP_METRICS:-1}"   # 1记录每批 overlap 和 P_overlap，0关闭

GPU_IDS="${GPU_IDS:-2,3}"                 # 所选方法依次使用同一对物理卡
WAIT_SECONDS="${WAIT_SECONDS:-3600}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs}"
VANILLA_RUN_NAME="${VANILLA_RUN_NAME:-${TRAIN_RUN_NAME:-opd_n1024_vanilla_seed42_v1}}"
COMPASSOPD_RUN_NAME="${COMPASSOPD_RUN_NAME:-opd_n1024_compassopd_seed42_v1}"
REFERENCE_MODEL="${REFERENCE_MODEL:-assets/models/Qwen3.5-0.8B}" # CompassOPD的教师同家族参考
ANCHOR_COEF="${ANCHOR_COEF:-0.5}"         # CompassOPD的冻结初始学生锚定系数
# 默认模板中的 training_input_data_dir 指向冻结的1024题。也兼容显式指定旧的准备配置。
if [[ -n "${SOURCE_PARAM_NAME:-}" ]]; then
    PREPARED_CONFIG="${PREPARED_CONFIG:-outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/${SOURCE_PARAM_NAME}/metadata/launch_config.json}"
fi
PREPARED_CONFIG="${PREPARED_CONFIG:-configs/ablation.json}"

set -euo pipefail
usage() {
    cat <<'HELP'
用法  bash scripts/baselines.sh [--check|--prepare|--run|--help]
默认 --check 在 CPU 上预览。--prepare 在 CPU 上冻结各自的训练计划。--run 依次启动或恢复训练。
脚本开头的 BASELINES 决定方法和顺序。临时覆盖示例
  METHODS=compassopd bash scripts/baselines.sh --check
  METHODS="compassopd vanilla" GPU_IDS=0,4 bash scripts/baselines.sh --run
HELP
}

mode="${1:---check}"
if (( $# > 1 )); then usage >&2; exit 2; fi
case "$mode" in
    --check|--prepare|--run) ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if [[ -v METHODS ]]; then
    IFS=$' ,\t' read -r -a BASELINES <<< "${METHODS//$'\n'/ }"
fi
if (( ${#BASELINES[@]} == 0 )); then
    echo '请至少选择 vanilla 或 compassopd 中的一种。' >&2
    exit 2
fi

declare -A selected_runs=()
for method in "${BASELINES[@]}"; do
    case "$method" in
        vanilla) run_name="$VANILLA_RUN_NAME" ;;
        compassopd) run_name="$COMPASSOPD_RUN_NAME" ;;
        *) printf '未知方法 %s，可选 vanilla 或 compassopd。\n' "$method" >&2; exit 2 ;;
    esac
    if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        printf '无效运行名 %s。\n' "$run_name" >&2
        exit 2
    fi
    if [[ -v selected_runs[$run_name] ]]; then
        printf '运行名 %s 重复，请检查方法列表和各方法的运行名。\n' "$run_name" >&2
        exit 2
    fi
    selected_runs[$run_name]="$method"
done
case "$OVERLAP_METRICS" in
    1) monitor_args=(--overlap-metrics) ;;
    0) monitor_args=() ;;
    *) echo 'OVERLAP_METRICS 只能为 0 或 1。' >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/runtime_env.sh
if [[ "$mode" != --run ]]; then export CUDA_VISIBLE_DEVICES=''; fi
baseline_src=$("$STEER_OPD_PYTHON" scripts/prepare_baseline_runtime.py)
export PYTHONPATH="$baseline_src:$PYTHONPATH"
for method in "${BASELINES[@]}"; do
    case "$method" in
        vanilla) method_args=(--loss traditional_opd --run-name "$VANILLA_RUN_NAME") ;;
        compassopd) method_args=(--loss compass_opd --run-name "$COMPASSOPD_RUN_NAME"
            --reference-model "$REFERENCE_MODEL" --anchor-coef "$ANCHOR_COEF") ;;
    esac
    printf '\n[baselines] %s %s\n' "$method" "$mode" >&2
    "$STEER_OPD_PYTHON" -m steer_opd.training.verl_entry "$mode" \
        "${method_args[@]}" --config "$PREPARED_CONFIG" --operator slerp --alpha 0 \
        --train-output-root "$TRAIN_OUTPUT_ROOT" --gpu-ids "$GPU_IDS" --wait-seconds "$WAIT_SECONDS" \
        --batch-size "$TRAIN_BATCH_SIZE" --total-steps "$TOTAL_STEPS" \
        --evaluation-freq "$EVAL_EVERY" --save-freq "$SAVE_EVERY" \
        --checkpoint-retention "$CHECKPOINT_RETENTION" --presence-penalty "$PRESENCE_PENALTY" "${monitor_args[@]}"
done
