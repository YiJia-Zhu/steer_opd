# 学生空间差分与overlap门控辅助OPD

研究：强教师有额外能力，但学生难以利用其OPD监督时，能否从学生空间提取有用方向，
在师生高概率候选不相容的位置干预，改善训练后的独立学生表现？

**2026-11-12主候选：教师同家族差分T−R＋overlap单阈值门控。**
T/R/S回答均由同一个冻结初始学生读取；方向在训练前固定。生成和学习前向使用
一致的受控策略，主评测完全去掉steering和门控。熵差作为独立对照，不与overlap组合。
基础配方已按用户要求统一到Rethinking原fork Student Top-K16反向KL系列surrogate，
其它默认训练/评测参数见[默认配置](configs/rethinking_opd.json)和
[来源对照](audit/rethinking_alignment_2026-11-12/comparison.json)。

| 文档 | 内容 |
|---|---|
| [用户最新完整稿](docs/Steering_OPD_完整研究方案.md) | 本轮方法逻辑及对齐后的基础配方；修改前原文已留快照 |
| [主计划摘要](docs/01_research_plan.md) | 三种方向、三种策略、OPD闭环、实验与结论边界 |
| [论文与资产证据](docs/02_paper_settings.md) | 固定论文版本、原设置、阅读范围与资产缺口 |
| [执行与验收](docs/03_execution_plan.md) | 缓存/probe/重放、概率接口、分阶段面板及预算 |
| [新颖性审计](docs/09_novelty_audit_cross_family_steering_opd.md) | 原2026-11-07审计及新版本适用范围提示 |
| [AGENTS](AGENTS.md) | 常驻状态、权限和方法边界 |

保留方案1 T−S、主方案2 T−R、R−S对照，配合全回答位置/overlap动态/因果后段形成三乘三矩阵，
先建立最小闭环再补齐。SFT替代、成本更低和跨家族效果均须实验支持。

当前首轮为Qwen3-4B Non-thinking→Qwen3-1.7B-Base，比较普通OPD与T−S全位置steer。
OpenThoughts3已冻结100题OPD、256题方向及128题dev；R候选Qwen3-0.6B留待后续。
AIME2025的30题只评测。本地SFT对照/下载暂缓，C1及其他旧排程不自动启动。

**2026-11-12用户已授权实现并直接运行首轮普通OPD/T−S全位置steer两组。**
新入口为`run_figure8_campaign.py`，配置为`configs/figure8_first_run.json`，本入口
每阶段只使用一张通过空闲预检的卡，不终止外部进程。断点包括每条回答/评分及每次完整
参数更新的FP32优化器/RNG状态。实际进度见[runs记录](runs/figure8_first_run_20261112/README.md)，
`scripts/figure8_status.py`可随时查看。旧`launch_pilot.py`仍为原fork预览入口。
模型与数据均复用本地输入；SFT/C1和其它扩展保持暂停。
文档同步前快照与记录见
[同步审计](audit/plan_integration_2026-11-12/manifest.json)。
