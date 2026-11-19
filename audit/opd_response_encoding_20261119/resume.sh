#!/usr/bin/env bash
# 从原第16步完整checkpoint续跑；--prepare仅CPU准备，--run启动并恢复。
# 恢复参数必须与源checkpoint一致，旧运行配置和日志保持原身份。
TRAIN_RUN_NAME="opd_n1024_project_outputs_verbatim_v3"
GPU_IDS="2,3"                 # 物理GPU池；繁忙时由准入器排队。
TRAIN_BATCH_SIZE=32            # 每步32题，每题4条，seed42/43/44/45。
TOTAL_STEPS=64                 # 包含已完成的16步，总预算仍为64步。
EVAL_EVERY=16                  # 每16步及末步评测裸学生。
SAVE_EVERY=16                  # 每16步及末步保留模型，恢复状态仅保留最新。
TEACHER_LAYER=29
OPERATOR=slerp
ALPHA=0.1
PRESENCE_PENALTY=2.0

PROJECT_ROOT=/local_nvme/data/zhenning/working_tmp/steer_opd
SOURCE_PARAM=n1024_slerp0.1_S26_T29_seed42_trainall_Tgen16384_Sgen16384_Ttemp1_Tp1_pp2_v1_st_global_all_full_v3
PAIR_DATA=T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared
SOURCE_RUN=opd_n1024_project_outputs_utf8_v2
RESUME_FROM="outputs/step_4_train/ablation/${PAIR_DATA}/${SOURCE_RUN}/checkpoints/global_step_16"
CONFIG="outputs/step_3_overlap/ablation/${PAIR_DATA}/${SOURCE_PARAM}/metadata/launch_config.json"
SV_BANK="outputs/step_2_sv/ablation/${PAIR_DATA}/${SOURCE_PARAM}/reader_teacher/banks/S-T_global.pt"

set -euo pipefail
cd "$PROJECT_ROOT"
source scripts/runtime_env.sh
MODE="${1:---run}"
case "$MODE" in --run|--prepare|--check) ;; *) echo '用法：resume.sh [--prepare|--check|--run]' >&2; exit 2 ;; esac
COMMAND=( "$STEER_OPD_PYTHON" -m steer_opd.training.verl_entry "$MODE" \
    --config "$CONFIG" --sv-bank "$SV_BANK" \
    --run-name "$TRAIN_RUN_NAME" --train-output-root outputs \
    --gpu-ids "$GPU_IDS" --batch-size "$TRAIN_BATCH_SIZE" --total-steps "$TOTAL_STEPS" \
    --evaluation-freq "$EVAL_EVERY" --save-freq "$SAVE_EVERY" \
    --checkpoint-retention all_models_latest_state_v1 \
    --teacher-layer "$TEACHER_LAYER" --operator "$OPERATOR" --alpha "$ALPHA" \
    --presence-penalty "$PRESENCE_PENALTY" --overlap-metrics --resume-from "$RESUME_FROM" )
if [[ "$MODE" == --run ]]; then
    # 子进程原本只写console.log；实时转发到ste，结束后tail随此脚本退出。
    CONSOLE="outputs/step_4_train/ablation/${PAIR_DATA}/${TRAIN_RUN_NAME}/logs/console.log"
    tail --pid="$$" -n 60 -F "$CONSOLE" 2>/dev/null &
    "${COMMAND[@]}"
else
    exec "${COMMAND[@]}"
fi
