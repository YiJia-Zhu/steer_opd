#!/usr/bin/env bash
# UTF-8对齐修复后的独立训练身份；旧运行无checkpoint，从初始学生重跑。
TRAIN_RUN_NAME=opd_n1024_project_outputs_utf8_v2
GPU_IDS=2,3
TRAIN_BATCH_SIZE=32
TOTAL_STEPS=64
EVAL_EVERY=16
SAVE_EVERY=16
TEACHER_LAYER=29
OPERATOR=slerp
ALPHA=0.1
PRESENCE_PENALTY=2.0
TRAIN_OUTPUT_ROOT=outputs
CHECKPOINT_RETENTION=all_models_latest_state_v1

set -euo pipefail
cd /local_nvme/data/zhenning/working_tmp/steer_opd
source scripts/runtime_env.sh
exec "$STEER_OPD_PYTHON" -m steer_opd.training.verl_entry "${1:---run}" \
    --config outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1024_slerp0.1_S26_T29_seed42_trainall_Tgen16384_Sgen16384_Ttemp1_Tp1_pp2_v1_st_global_all_full_v3/metadata/launch_config.json \
    --sv-bank outputs/step_2_sv/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1024_slerp0.1_S26_T29_seed42_trainall_Tgen16384_Sgen16384_Ttemp1_Tp1_pp2_v1_st_global_all_full_v3/reader_teacher/banks/S-T_global.pt \
    --run-name "$TRAIN_RUN_NAME" --train-output-root "$TRAIN_OUTPUT_ROOT" \
    --gpu-ids "$GPU_IDS" --batch-size "$TRAIN_BATCH_SIZE" --total-steps "$TOTAL_STEPS" \
    --evaluation-freq "$EVAL_EVERY" --save-freq "$SAVE_EVERY" \
    --checkpoint-retention "$CHECKPOINT_RETENTION" --teacher-layer "$TEACHER_LAYER" \
    --operator "$OPERATOR" --alpha "$ALPHA" --presence-penalty "$PRESENCE_PENALTY" \
    --overlap-metrics
