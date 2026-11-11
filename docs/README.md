# 文档入口

更新：2026-11-12。**以用户最新完整稿为逻辑来源，本文档导航已同步；不启动实验或下载。**
主候选为T−R方向＋overlap单阈值门控，熵差独立对照；生成和学习使用一致受控策略，
主评测无steer/无门控。先在兼容词表验证，再决定跨家族/跨tokenizer扩展。

| 顺序 | 文档 | 用途 |
|---|---|---|
| 1 | [主计划摘要](01_research_plan.md) | 当前方法、三乘三矩阵、100题试验与证据边界 |
| 2 | [用户最新完整稿](Steering_OPD_完整研究方案.md) | 本轮来源与完整推导，保留原文 |
| 3 | [论文设置与证据](02_paper_settings.md) | 固定来源、原论文配方、阅读范围和资产 |
| 4 | [执行与验收](03_execution_plan.md) | 受控rollout、单步probe、缓存隔离和训练重放 |
| 5 | [新颖性历史审计](09_novelty_audit_cross_family_steering_opd.md) | 原2026-11-07范围及新版本提示，不能视为本轮重新完成审计 |
| 6 | [启动准备检查](../audit/steering_full_plan_readiness_2026-11-12/README.md) | 2026-11-12本地CPU证据、词表差异、代码缺口和最小闭环建议 |

[AGENTS](../AGENTS.md)记录当前暂停和恢复条件；[运行模板](../experiments/run_record_template.md)
统一记录方向、门控、行为策略、目标、无干预评测及总成本。
[同步审计](../audit/plan_integration_2026-11-12/manifest.json)保存修改前快照和输入版本hash。
旧JSON/CSV及历史启动器并未随本次Markdown改写为新方法，不构成运行配置。
