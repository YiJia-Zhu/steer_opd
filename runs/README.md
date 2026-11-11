# 运行记录与历史产物

当前所有实验及模型/数据下载暂停，先审定[完整方案](../docs/01_research_plan.md)。
新配对与验收见[论文设置](../docs/02_paper_settings.md)和[执行清单](../docs/03_execution_plan.md)。
旧命令/队列不代表当前授权，不自动续跑Phi/Smol或R1历史排程。

| 记录 | 已有事实与边界 |
|---|---|
| [pilot_20261108_01](pilot_20261108_01/campaign.json) | 初始R1/JustRL评测及早期工程记录；旧队列曾因环境漂移失败 |
| [pilot_20261110_01](pilot_20261110_01/campaign.json) | E0/E3各100步工程实验完成，队列被后续优先级替代；未形成训练后效果结论 |
| [cross_family_overlap_20261110_02](cross_family_overlap_20261110_02/summary.json) | 历史Phi/Smol/Qwen探索的144状态基线；修复上下文空格后复用原轨迹重算 |
| [cross_family_steering_probe_20261110_02](cross_family_steering_probe_20261110_02/campaign.json) | 历史Phi/Smol方向诊断执行结束；无参数更新，不替代新论文配对 |
| [用户停止记录](cross_family_steering_probe_20261110_02/user_stop_20261110.json) | 停止核查时已自然结束，无需发送信号；暂停后续运行 |

所有原始产物保留。需要具体结果时读取逐题文件、summary与process退出码，不从目录名
或队列启动判断完成。历史文档名的迁移去向在[清理清单](../audit/docs_cleanup_2026-11-10/manifest.json)。

恢复后仍逐阶段重新检查nvidia-smi，仅使用空闲GPU；不终止其他项目，不运行全局
ray stop、pkill或killall。新计划中的异构候选训练和方案2尚无已验收启动器。
