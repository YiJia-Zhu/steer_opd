# 面向低相容师生的门控 Steering OPD

研究：强教师难以被学生通过普通OPD吸收时，能否通过学生空间中的参考差分与训练期干预，
改善撤去steering后的独立学生表现？不预设方法有效。

**当前完整方案以2–2–2为主线：T−R差分、overlap单阈值门控、当前学生逐题提取。**
每轮rollout前，当前学生关闭steering和梯度，读取当前OPD题的教师T与小参考R回答，
得到该题方向。同题在本轮的多个rollout及学习重放中使用同一向量；学生更新后再次遇题时重提。
教师/小参考文本可以逐题缓存；文本缓存、初始学生逐题向量缓存和当前学生重提是不同设置。
最终主评测完全关闭steering、门控和教师调用，熵差门控只作独立对照。

| 文档 | 内容 |
|---|---|
| [用户完整稿](docs/Steering_OPD_完整研究方案.md) | 三个设计问题、2–2–2主线与全部备选 |
| [主计划摘要](docs/01_research_plan.md) | 逐题提取流程、数据复用、粒度与刷新对照 |
| [论文与资产证据](docs/02_paper_settings.md) | 固定来源、SV全局/逐题/刷新先例及原配方 |
| [执行与验收](docs/03_execution_plan.md) | 逐题持久化、提取版本、生成/重放一致性与实施边界 |
| [新颖性历史审计](docs/09_novelty_audit_cross_family_steering_opd.md) | 历史检索及新版适用范围；不视为新审计 |
| [AGENTS](AGENTS.md) | 授权、当前运行与方法边界 |

提取与OPD题目在逐题主方案中相同；dev与测试保持独立。独立题集全局方向是迁移对照，
同时保留同数据全局均值与初始学生逐题缓存，分别检验粒度和随学生刷新是否有收益。
以主方案为中心逐项改变差分、门控与提取策略，按需要补交叉，不把完整组合矩阵作为起步条件。

**当前实验已按用户要求停止，正在准备同题Top16 v2，暂不运行GPU。**
旧首轮普通OPD/T−S全位置全局固定方向的两组各10/50步保留，使用
[旧配置](configs/figure8_first_run.json)与[旧记录](runs/figure8_first_run_20261112/README.md)。
新配置[同题Top16 v2](configs/figure8_shared_top16_v2.json)让SV提取与OPD共用原100题，
保留原独立dev128与AIME2025 30题；rollout逐token保存原始Top16候选和log-prob，
教师一次前向评分并计算训练轨迹overlap，跳过HF学生无梯度重放。
仍用S0提取的全局固定T−S（layer13/rho0.03），生成与学习注入，评测无hook；
不自动加入T−R、逐题刷新或动态门控。两组新训练从原Base开始，旧10步不混入新曲线。
代码与CPU验收见[修改记录](audit/rollout_top16_shared_data_2026-11-12/README.md)，
新接口GPU验收须待用户恢复。入口拒绝自动重启paused_by_user任务。

基础配方仍沿用Student Top-K16反向KL系列surrogate，冻结student_p奖励、PPO与token-mean的
原代码口径见[对照报告](audit/rethinking_alignment_2026-11-12/README.md)。原fork默认由
[参考配置](configs/rethinking_opd.json)记录；当前单GPU首轮的实际配置与支持集修正在其运行记录中明示。
新版逐题重提与门控尚待实现验收，不因文档更新而视为已有能力。
修改前快照、输入原文与本轮校验见[同步记录](audit/extraction_timing_sync_2026-11-12/manifest.json)。
