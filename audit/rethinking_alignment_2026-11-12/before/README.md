# 学生空间差分与overlap门控辅助OPD

研究：强教师有额外能力，但学生难以利用其OPD监督时，能否从学生空间提取有用方向，
在师生高概率候选不相容的位置干预，改善训练后的独立学生表现？

**2026-11-12主候选：教师同家族差分T−R＋overlap单阈值门控。**
T/R/S回答均由同一个冻结初始学生读取；方向在训练前固定。生成和学习前向使用
一致的受控策略，主评测完全去掉steering和门控。熵差作为独立对照，不与overlap组合。

| 文档 | 内容 |
|---|---|
| [用户最新完整稿](docs/Steering_OPD_完整研究方案.md) | 本轮逻辑来源，保留原文 |
| [主计划摘要](docs/01_research_plan.md) | 三种方向、三种策略、OPD闭环、实验与结论边界 |
| [论文与资产证据](docs/02_paper_settings.md) | 固定论文版本、原设置、阅读范围与资产缺口 |
| [执行与验收](docs/03_execution_plan.md) | 缓存/probe/重放、概率接口、分阶段面板及预算 |
| [新颖性审计](docs/09_novelty_audit_cross_family_steering_opd.md) | 原2026-11-07审计及新版本适用范围提示 |
| [AGENTS](AGENTS.md) | 常驻状态、权限和方法边界 |

保留方案1 T−S、主方案2 T−R、R−S对照，配合全回答位置/overlap动态/因果后段形成三乘三矩阵，
先建立最小闭环再补齐。SFT替代、成本更低和跨家族效果均须实验支持。

当前先讨论Qwen3-4B Non-thinking→Qwen3-1.7B-Base，R候选Qwen3-0.6B。
OpenThoughts3中100题仍仅指OPD题池；方向题和dev独立划分的新增预算尚未冻结。
AIME2025的30题只评测。本地SFT对照/下载暂缓，C1及其他旧排程不自动启动。

**当前只更新文档，实验和下载继续暂停。** 现有模型/数据软链接优先复用。
旧`experiments/` JSON/CSV及启动器描述此前loss-only草案，尚未迁移到新主方法，不能直接启动。
本次同步保留用户完整稿及论文/来源证据；修改前快照与记录见
[同步审计](audit/plan_integration_2026-11-12/manifest.json)。
