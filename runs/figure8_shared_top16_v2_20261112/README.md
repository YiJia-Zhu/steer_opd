# 同题Top16 v2：已准备，实验暂停

2026-11-12用户要求停止后，仅完成代码/数据和CPU检查，没有启动GPU。
新训练两组均0/50步；旧实验两组10步和部分评测保留在`runs/figure8_first_run_20261112/`。

配置`configs/figure8_shared_top16_v2.json`：原100道OPD题同时提取SV，dev128/AIME30独立且不变。
T=Qwen3-4B Non-thinking，S=Qwen3-1.7B-Base；S0全局固定T−S，layer13/rho0.03，生成与学习注入。
rollout保存逐token完整Top16，直接教师评分/训练轨迹overlap，再学生带梯度学习，无单独HF学生评分。
同题校准文本和新SV尚未生成；两组新曲线从原Base开始，不继承旧10步权重。

原Base AIME480条（5/480）及固定dev前缀/教师评分已在协议核对后复制，1031文件SHA256一致。
第0步无hook overlap=59.29321554%，两组共享；来源见`initial_evaluation_reuse.json`。
没有重新做GPU生成或将Base结果当成新训练收益。

48项CPU检查通过；新Top16接口GPU完整支持分布/真实更新和长序列吞吐待用户恢复后验收。
`campaign.json`为paused_by_user，配置execution_hold阻止默认启动。
详细实现及证据：`audit/rollout_top16_shared_data_2026-11-12/README.md`。
