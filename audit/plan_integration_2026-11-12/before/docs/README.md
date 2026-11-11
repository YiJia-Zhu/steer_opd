# 文档入口

更新：2026-11-11。**优先准备Rethinking配对下Base OPD与Base+steer两组验证，
本地SFT对照暂缓。OpenThoughts3取100题做OPD，
AIME2025仅评测；50步及steer细节待用户检查确认；
跨family实验暂停。** 最新核对见[执行说明](03_execution_plan.md)。

主线先试跨family OPD适配；是否继续由真实学习收益决定。若效果不好，回到
Rethinking共享词表设置研究更窄的问题，不预设steering一定有效。

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | [研究方案](01_research_plan.md) | 通俗说明方案1/2、效果判断、跨family主线与Rethinking回退路线 |
| 2 | [论文设置与证据](02_paper_settings.md) | Rethinking/Simple/Compass/SV的原设置、配对、数据、模型缺口与阅读范围 |
| 3 | [执行与验收](03_execution_plan.md) | 当前实现状态、环境、P0–P5、配置起点、检查和恢复条件 |
| 4 | [新颖性审计](09_novelty_audit_cross_family_steering_opd.md) | 已有近邻、重合风险、固定来源和证据深度，保留原审计日期 |

项目常驻约束与TODO见[AGENTS](../AGENTS.md)。论文与来源材料见
`papers/`、`ref/`和`audit/`。
