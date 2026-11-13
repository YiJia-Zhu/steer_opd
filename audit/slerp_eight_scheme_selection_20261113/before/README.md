# 面向低相容师生的门控 Steering OPD

**2026-11-13下一配对已确定：T=Qwen3.5-35B-A3B、Ref=Qwen3.5-0.8B、S=Qwen3-4B。**
新学生36层，50%/75%/100%对应0-based17/26/35。[分阶段搜索建议](docs/13_next_pair_greedy_search_proposal.md)
将原192候选池按约40个非零配置筛选，保留两种算子和8方案回查；顺序尚未最终冻结或运行。
新配对须先验收跨tokenizer的可比概率事件；下述28层/55有效题及旧缓存说明属于上一配对。

**当前阶段：静态 overlap 的零更新参数搜索准备。** 以[静态协议](docs/12_static_overlap_protocol.md)和
[下一步搜索](docs/10_overlap_parameter_search.md)为准：普通无SV学生rollout保存原生Top16，
教师在同一前缀批量评分后冻结低overlap位置；所有候选在同源回答前缀上应用固定mask，注入后不重判位置。

搜索为T−S/T−R × 全位置/静态overlap × 全局/逐题共8方案；加法rho和Slerp alpha均取
`0.01 / 0.1 / 0.3 / 0.5`，单层深度为`50% / 75% / 100%`，当前28层模型对应0-based `13 / 20 / 27`。
共192个非零候选及共同无干预基线；全部0次参数更新，默认保存前3条同前缀有/无SV曲线、实际注入散点和差值。
结果交用户检查，后续learning-only或静态mask二次生成的训练阶段尚未选定。

**实施状态（2026-11-13续接核对）：** 静态mask适配、Slerp和新搜索入口尚待实现验收。
当前源码`gate=overlap`仍是历史动态门控，旧入口不能直接执行新的192候选规格。
已有55题公共T/R/S激活包含第13、20层，第27层需补提取；这55题与旧32题dev前缀没有交集，
旧学生参考回答也未保存原生Top16。新逐题搜索的数据准备边界见[续接核对](audit/static_overlap_continuation_20261113/README.md)。

旧动态候选首步rollout已完成256条，后续学习重放失败，正式更新0；候选和面板为`failed_resumable`。
核对时未发现其生成、训练或监督进程，顶层`campaign.json`的`running`是旧记录。
本次续接只核对源码/缓存并补文档，未改生产代码、旧配置或产物，也未启动GPU和新搜索。

| 当前入口 | 用途 |
|---|---|
| [新配对与贪婪搜索建议](docs/13_next_pair_greedy_search_proposal.md) | 已确定模型、36层映射、40配置建议及跨tokenizer前置接口 |
| [静态overlap协议](docs/12_static_overlap_protocol.md) | mask来源、固定前缀比较、后续应用阶段与EasySteer边界 |
| [192候选搜索](docs/10_overlap_parameter_search.md) | 已固定参数、共同数据、排名、绘图与实施顺序 |
| [文档导航](docs/README.md) | 当前协议和历史记录索引 |
| [仓库约定](AGENTS.md) | 最新范围、资源与研究记录要求 |

以下保留此前动态方案和旧实验的说明，反映各自记录时点；当前状态与范围以上文为准。

## 历史队列与研究说明

历史完整队列说明：[完整方案实施与恢复](docs/08_full_scheme_execution.md)。旧实验状态在`runs/full_scheme_20261112/`，
旧入口为`scripts/run_full_scheme.py --execute`，包含后续学习阶段；不能用它启动当前静态零更新搜索。

**2026-11-12最新授权：恢复完整方案的实现、验收和分阶段实验。** 新任务在`runs/full_scheme_20261112/`，
与下文旧暂停记录分开。优先一张空闲GPU，首次检查所有卡均被占用，precheck已排队等待。
先执行原定T−S/all/S0-global零更新诊断，再推进现有配对的主线和对照；实际状态以新process/campaign记录为准。

**当前目标已调整：暂停OPD，先在原始Base上调steering，做0次参数更新的固定前缀overlap诊断。**
先检验同题提取的T−S全局方向，再扫描注入层位置、注入层数与强度；完整结果交用户检查，
收到明确下一步指令后才进入短程OPD。最新已授权实现方法分支及独立零更新诊断入口；当前仅做代码/CPU准备，不运行GPU。
具体网格、指标与停止点见[训练前overlap调参](docs/04_pre_opd_overlap_tuning.md)，
代码开关、默认第一步与验收边界见[方法分支](docs/05_method_branches.md)。
旧第10步无hook overlap发生在OPD之后，不是这次要测的初始Base带hook即时overlap。
方向提取与未来OPD直接固定共用原100题；当前直接进行层位置、层数和强度调参。

下文2–2–2为后续完整研究设计，当前不启动该面板。

研究：强教师难以被学生通过普通OPD吸收时，能否通过学生空间中的参考差分与训练期干预，
改善撤去steering后的独立学生表现？不预设方法有效。

**当前完整方案以2–2–2为主线：T−R差分、overlap单阈值门控、当前学生逐题提取。**
每轮rollout前，当前学生关闭steering和梯度，读取当前OPD题的教师T与小参考R回答，
得到该题方向。同题在本轮的多个rollout及学习重放中使用同一向量；学生更新后再次遇题时重提。
教师/小参考文本可以逐题缓存；文本缓存、初始学生逐题向量缓存和当前学生重提是不同设置。
最终主评测完全关闭steering、门控和教师调用，熵差门控只作独立对照。

| 文档 | 内容 |
|---|---|
| [当前第一步：训练前overlap调参](docs/04_pre_opd_overlap_tuning.md) | 冻结Base、层/强度扫描、0次OPD、结果后人工检查 |
| [用户完整稿](docs/Steering_OPD_完整研究方案.md) | 三个设计问题、2–2–2主线与全部备选 |
| [主计划摘要](docs/01_research_plan.md) | 逐题提取流程、数据复用、粒度与刷新对照 |
| [论文与资产证据](docs/02_paper_settings.md) | 固定来源、SV全局/逐题/刷新先例及原配方 |
| [执行与验收](docs/03_execution_plan.md) | 逐题持久化、提取版本、生成/重放一致性与实施边界 |
| [新颖性历史审计](docs/09_novelty_audit_cross_family_steering_opd.md) | 历史检索及新版适用范围；不视为新审计 |
| [AGENTS](AGENTS.md) | 授权、当前运行与方法边界 |

提取与OPD题目在逐题主方案中相同；dev与测试保持独立。独立题集全局方向是迁移对照，
同时保留同数据全局均值与初始学生逐题缓存，分别检验粒度和随学生刷新是否有收益。
以主方案为中心逐项改变差分、门控与提取策略，按需要补交叉，不把完整组合矩阵作为起步条件。

**实验保持停止；同题Top16 v2是此前已完成的代码/数据准备，本轮不继续修改或启动。**
旧首轮普通OPD/T−S全位置全局固定方向的两组各10/50步保留，使用
[旧配置](configs/figure8_first_run.json)与[旧记录](runs/figure8_first_run_20261112/README.md)。
新配置[同题Top16 v2](configs/figure8_shared_top16_v2.json)让SV提取与OPD共用原100题，
保留原独立dev128与AIME2025 30题；rollout逐token保存原始Top16候选和log-prob，
教师一次前向评分并计算训练轨迹overlap，跳过HF学生无梯度重放。
其保存值仍为S0全局固定T−S（layer13/rho0.03）；这是旧锚点，不是调参结论。
原配置计划生成与学习注入，训练后评测无hook；
不自动加入T−R、逐题刷新或动态门控。两组新训练从原Base开始，旧10步不混入新曲线。
代码与CPU验收见[修改记录](audit/rollout_top16_shared_data_2026-11-12/README.md)，
新接口GPU验收须待后续授权。现有campaign会先做参数更新，不可用它直接恢复本轮纯overlap阶段。
入口仍拒绝自动重启paused_by_user任务；新诊断入口尚未实现。

基础配方仍沿用Student Top-K16反向KL系列surrogate，冻结student_p奖励、PPO与token-mean的
原代码口径见[对照报告](audit/rethinking_alignment_2026-11-12/README.md)。原fork默认由
[参考配置](configs/rethinking_opd.json)记录；当前单GPU首轮的实际配置与支持集修正在其运行记录中明示。
新版逐题重提与门控尚待实现验收，不因文档更新而视为已有能力。
修改前快照、输入原文与本轮校验见[同步记录](audit/extraction_timing_sync_2026-11-12/manifest.json)。
