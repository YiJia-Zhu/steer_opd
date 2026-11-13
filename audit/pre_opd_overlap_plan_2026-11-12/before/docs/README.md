# 文档入口

更新：2026-11-12。以[用户完整稿](Steering_OPD_完整研究方案.md)新增的第三个问题为准：
**什么时候提取steer，以及全局/逐题与固定/刷新如何分别比较。**
主线2–2–2为T−R、overlap单门控、每轮rollout前用当前学生提取当前题方向。
方向在本轮生成和学习重放中固定；最终主评测无steer、无门控、无教师。

| 顺序 | 文档 | 用途 |
|---|---|---|
| 1 | [主计划摘要](01_research_plan.md) | 三个设计问题、同题数据、粒度/刷新对照与运行范围 |
| 2 | [用户完整稿](Steering_OPD_完整研究方案.md) | 本轮逻辑来源；本次未改写 |
| 3 | [论文设置与证据](02_paper_settings.md) | 原配方、逐题固定先例、周期刷新结果与阅读边界 |
| 4 | [执行与验收](03_execution_plan.md) | 每题文本/向量保存、checkpoint版本、因果生成与重放 |
| 5 | [新颖性历史审计](09_novelty_audit_cross_family_steering_opd.md) | 旧方案审计与新版待核对边界 |
| 6 | [启动准备历史检查](../audit/steering_full_plan_readiness_2026-11-12/README.md) | 此前CPU证据、词表差异与当时实现缺口 |

[AGENTS](../AGENTS.md)记录授权及当前范围；[运行模板](../experiments/run_record_template.md)
统一记录题目ID、参考文本、提取者版本、本轮向量、mask、学习与无干预评测。
本轮[同步审计](../audit/extraction_timing_sync_2026-11-12/manifest.json)保存输入原文及修改前快照。

**最新执行状态：用户已停止实验，当前仅做代码/数据准备和CPU验收，不自动续跑。**
旧100/256/128首轮两组均10/50步，断点与所有已完成结果保留。
[同题Top16 v2](../configs/figure8_shared_top16_v2.json)改为SV/OPD共用原100题，
独立dev128与AIME30保持不变；rollout保存逐token原生Top16，直接交教师评分并计算训练轨迹overlap。
无hook固定dev overlap仍是主学习指标，两者分开记录。方向仍为S0全局固定T−S，
新版2–2–2的T−R、动态门控、在线逐题没有自动加入。
[代码/数据/验收记录](../audit/rollout_top16_shared_data_2026-11-12/README.md)列出恢复边界与待做GPU检查。
[原仓库对齐](../audit/rethinking_alignment_2026-11-12/README.md)保留此前79项训练/6项评测默认的证据；
旧JSON/CSV及原fork预览入口尚未迁移为新版逐题面板，不能据此启动新版实验。
