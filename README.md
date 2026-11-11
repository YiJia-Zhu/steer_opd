# Steering能否改善OPD适配

先研究跨family教师—学生的OPD适配：从少量教师示范提取学生自己的方向，
检验它能否让学生学得更好，关掉方向后仍有收益。
如果跨family没有可靠收益，就回到Rethinking的共享词表设置，研究更窄的学习困难
与冷启动成本；两边都无收益时接受负结果。“同一教师、多个学生”只是后续可选对照。

**最新优先任务：Rethinking Figure 8 + steer。** 固定Qwen3-4B Non-thinking教师，
对比Base→OPD、作者SFT→OPD与Base+steer→OPD的accuracy/overlap曲线。
AIME2025的使用范围待澄清，尚未启动GPU或权重下载；跨family及历史队列继续暂停。
具体设置见[执行说明](docs/03_execution_plan.md)。

## 阅读入口

| 文件 | 内容 |
|---|---|
| [完整研究方案](docs/01_research_plan.md) | 方案1教师−学生方向、方案2教师−弱参考方向、分词/概率目标、强基线、等规模控制与统计 |
| [论文设置与证据](docs/02_paper_settings.md) | Rethinking、Simple、Compass、SV的实际配对/数据/初始化，阅读边界与资产缺口 |
| [执行与验收](docs/03_execution_plan.md) | 当前实现、环境、阶段配置、恢复条件与验证 |
| [新颖性审计](docs/09_novelty_audit_cross_family_steering_opd.md) | 近邻与重合风险，保留原固定版本和来源 |
| [AGENTS](AGENTS.md) | 常驻研究目标、约束、状态与TODO |

## 方案范围

C1主实验对应Compass的Qwen3-30B-A3B-Instruct-2507/0.6B参考→Granite4.1-3B；
C2同教师→Qwen3-4B仅作可选控制；C3的Qwen3.5-4B→Qwen3-4B在有主结果后再考虑。
回退F1对应Qwen3-4B non-thinking→Qwen3-1.7B-Base，R1教师对照可辅助验证。
Phi/Smol移出正式主线。
缺失模型/语料仅列清单，不因模型公开或容易下载另换配对。

顺序：原始普通学生前缀诊断→真实/随机/负方向→短程更新后的无hook学习收益→
完整OPD和SFT/LoRA/Compass比较。普通rollout无hook，只在早期loss前向干预。
显式候选目标与原版span采样基线分开，不能将普通采样当steered policy采样。

旧重复探索Markdown已删除，保留原论文、数据输入、run日志、checkpoint与审计证据。
清理前哈希与新版去向见[清理记录](audit/docs_cleanup_2026-11-10/manifest.json)。
现有R1工程验证不代表新配对可训练或方法已有效，历史状态见[runs记录](runs/README.md)。

材料与计划可做CPU检查：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/validate_package.py
```

该检查只验证文件、来源、链接与计划一致性，不验证模型质量或批准实验启动。
