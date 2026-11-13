# 文档入口

**2026-11-13当前任务：** [8格overlap参数搜索候选](10_overlap_parameter_search.md)，等待用户挑选强度/层集合；本次没有启动新搜索。
逐token注入前/后对比图现为默认，每个正式rollout批次完成后自动画前3条。现有首步生成已完成，但训练因重放一致性检查失败退出，正式更新0；最新事实见[执行状态](08_full_scheme_execution.md#6-当前容量与运行状态)。
下文保留较早任务说明，以本条和最新执行状态为准。

当前恢复任务入口：[完整队列实施与恢复](08_full_scheme_execution.md)。包含新门控批量实现、选参、对照和实际资源限制。

**2026-11-12最新：用户已授权完整方案实现与实验，覆盖下文历史暂停及诊断后再确认限制。**
新任务使用`runs/full_scheme_20261112/`；先完成原定零更新诊断，再分阶段运行，旧实验不混入。
第一轮查卡全部被占用，precheck已进入等待；当前没有新增GPU结果。

**当前第一步：[训练前overlap调参](04_pre_opd_overlap_tuning.md)。** 用户已停止训练，
要求先在原始Base、相同固定前缀上调层位置/层数/强度，OPD更新数为0；
展示无hook基线与带hook即时差值，停在结果供用户检查，之后才考虑OPD。
最新授权已完成方法分支与独立诊断入口的代码/CPU准备；旧v2配置、数据、断点保持原样，不启动GPU。
下文三个设计问题是后续完整研究，不覆盖这一停止点。

更新：2026-11-12。以[用户完整稿](Steering_OPD_完整研究方案.md)新增的第三个问题为准：
**什么时候提取steer，以及全局/逐题与固定/刷新如何分别比较。**
主线2–2–2为T−R、overlap单门控、每轮rollout前用当前学生提取当前题方向。
方向在本轮生成和学习重放中固定；最终主评测无steer、无门控、无教师。

| 顺序 | 文档 | 用途 |
|---|---|---|
| 0 | [当前阶段：训练前overlap调参](04_pre_opd_overlap_tuning.md) | 同题方向、固定Base、网格/逐题数据与用户检查点 |
| 0c | [完整方案实现清单](07_control_implementation_matrix.md) | 核心分支、部分实现与缺失对照的实际证据 |
| 0b | [逐token留档与复用](06_token_level_records.md) | 注入位置对照、激活/分布/mask保存与必要重算边界 |
| 0a | [方法分支与第一步入口](05_method_branches.md) | 实际代码选项、配置生成、零更新边界与待做GPU验收 |
| 1 | [主计划摘要](01_research_plan.md) | 三个设计问题、同题数据、粒度/刷新对照与运行范围 |
| 2 | [用户完整稿](Steering_OPD_完整研究方案.md) | 本轮逻辑来源；本次未改写 |
| 3 | [论文设置与证据](02_paper_settings.md) | 原配方、逐题固定先例、周期刷新结果与阅读边界 |
| 4 | [执行与验收](03_execution_plan.md) | 每题文本/向量保存、checkpoint版本、因果生成与重放 |
| 5 | [新颖性历史审计](09_novelty_audit_cross_family_steering_opd.md) | 旧方案审计与新版待核对边界 |
| 6 | [启动准备历史检查](../audit/steering_full_plan_readiness_2026-11-12/README.md) | 此前CPU证据、词表差异与当时实现缺口 |

[AGENTS](../AGENTS.md)记录授权及当前范围；[运行模板](../experiments/run_record_template.md)
统一记录题目ID、参考文本、提取者版本、本轮向量、mask、学习与无干预评测。
本轮[同步审计](../audit/extraction_timing_sync_2026-11-12/manifest.json)保存输入原文及修改前快照。

**最新执行状态：已按用户要求实现可选代码分支；第一步仍T−S/all/S0-global，实验继续暂停。**
旧100/256/128首轮两组均10/50步，断点与所有已完成结果保留。
[同题Top16 v2](../configs/figure8_shared_top16_v2.json)改为SV/OPD共用原100题，
独立dev128与AIME30保持不变；rollout保存逐token原生Top16，直接交教师评分并计算训练轨迹overlap。
训练后无hook固定dev overlap仍是后续学习指标；当前先测S0带hook的即时overlap，三者分开记录。
此前配置方向仍为S0全局固定T−S，
新版T−R、门控和当前学生逐题已作为可选分支接入，未启用或运行GPU，见[实现说明](05_method_branches.md)。
[代码/数据/验收记录](../audit/rollout_top16_shared_data_2026-11-12/README.md)列出恢复边界与待做GPU检查。
[原仓库对齐](../audit/rethinking_alignment_2026-11-12/README.md)保留此前79项训练/6项评测默认的证据；
旧JSON/CSV及原fork预览入口尚未迁移为新版逐题面板，不能据此启动新版实验。
