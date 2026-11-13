# 文档入口

最新[Compass式跨tokenizer实现](16_compass_alignment_eight_way.md)已覆盖数据/bank/源评分/静态mask/8方案编排，按论文A.4独立实现，官方仓库待URL核对。
overlap是明确命名的项目字节续写指标，T/Ref/S各用原生支持。CPU准备100题，缺35B完整权重，新8组GPU未运行。

**下一配对已确定：Qwen3.5-35B-A3B（T）/Qwen3.5-0.8B（Ref）→Qwen3-4B（S）。**
**本轮已选定Slerp0.1、75%单层（0-based26），完整比较8方案＋共同基线**，见[本轮规格](14_slerp_eight_scheme_screen.md)。
先做0更新同前缀选方案；40配置搜索仍是历史建议。静态mask/Slerp及EasySteer API适配已接入，见[实现状态](15_static_mask_slerp_implementation.md)。
新版EasySteer引擎和新配对接口的GPU路径仍未验收，本轮尚未运行。
新接口已通过CPU验收；真实GPU路径仍待验收。
下文192为完整候选池，旧28层/55题缓存属于上一配对，不能直接复用为新学生缓存。

**2026-11-13最新决定：改用静态overlap。** 普通学生rollout完成后，教师在同一前缀批量评分，计算并冻结需要注入的idx；
后续按该位置表注入，注入后不动态重判。[静态协议](12_static_overlap_protocol.md)是当前门控规则。

[下一步实验](10_overlap_parameter_search.md)保留8方案：T−S/T−R × 全位置/静态overlap位置 × 全局/逐题SV。
两种算子加法/Slerp均取0.01/0.1/0.3/0.5，全部单层50%/75%/100%，共192个非零候选及无干预基线。
当前先做0更新、同源回答前缀的有/无SV重放比较，默认画前3条配对曲线、实际mask散点和差值。

EasySteer支持已知位置selector，后续若选择按静态mask再生成可使用该接口；本地迁移、Slerp和静态适配尚未完成验收。
已完成源回答的学习前向注入与二次自由生成分别列明，不能混作相同阶段；本次没有自动选择后续训练阶段。
本轮只改文档和规格，没有启动GPU、新搜索或恢复旧失败训练。

旧动态首步256条生成耗时2小时48分30秒，后续学习重放失败、正式更新0。
[在线门控耗时与引擎改造](11_online_gate_acceleration.md)现为历史路线，不再要求完成动态引擎迁移才能做当前搜索。
旧run、配置、数据、权重、mask、动态比较图与实际失败证据保留原义。

| 文档 | 当前用途 |
|---|---|
| [Compass式跨tokenizer与8方案](16_compass_alignment_eight_way.md) | span似然、项目overlap、完整编排、来源和实际停点 |
| [静态代码与EasySteer适配](15_static_mask_slerp_implementation.md) | 已实现代码、入口、CPU证据及本地GPU/跨tokenizer缺口 |
| [固定Slerp的8方案筛选](14_slerp_eight_scheme_screen.md) | 本轮已选配置、完整8方案、静态规则与源码状态 |
| [下一配对与搜索建议](13_next_pair_greedy_search_proposal.md) | 已确定模型、36层映射、40配置顺序建议与指标接口 |
| [静态overlap协议](12_static_overlap_protocol.md) | 源rollout/批量教师评分、冻结mask、EasySteer和应用阶段边界 |
| [192候选搜索](10_overlap_parameter_search.md) | 8方案、两种算子、强度/层位置、排名与默认图 |
| [训练前零更新诊断](04_pre_opd_overlap_tuning.md) | 同前缀DeltaOverlap、数据划分、结果检查 |
| [主计划摘要](01_research_plan.md) | 研究问题、方向粒度/刷新与学习收益边界 |
| [完整研究稿](Steering_OPD_完整研究方案.md) | 当前静态方案、后续研究和文献关系 |
| [执行与验收](03_execution_plan.md) | 数据/方向/模型版本保存和恢复要求 |
| [完整队列与历史执行](08_full_scheme_execution.md) | 当前静态流程，以及旧队列实际运行/失败状态 |
| [代码分支说明](05_method_branches.md) | 已有动态代码和计划静态适配的实现区别 |
| [逐token留档](06_token_level_records.md) | 静态mask来源、位置映射、概率/图及复用 |
| [实施清单](07_control_implementation_matrix.md) | 当前静态缺口与历史实现审计 |
| [论文设置与阅读边界](02_paper_settings.md) | 原配方和论文证据，不将SFT结果外推为OPD收益 |
| [新颖性历史审计](09_novelty_audit_cross_family_steering_opd.md) | 近邻工作与当前静态方案的待验证边界 |
| [历史动态性能路线](11_online_gate_acceleration.md) | 耗时事实与已降为非当前前置项的引擎改造 |

[AGENTS](../AGENTS.md)记录最新范围；[运行记录模板](../experiments/run_record_template.md)记录源回答、静态mask、方向、阶段及无干预评测。
[本次静态同步审计](../audit/static_overlap_protocol_20261113/README.md)保留修改前文档、规格和核验结果。
