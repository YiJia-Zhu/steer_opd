# 同问题工作与 Steering 方法近邻

2026-11-12：当前方法按[用户最新完整稿](../docs/Steering_OPD_完整研究方案.md)与
[主计划](../docs/01_research_plan.md)定义：T−R主方向、overlap单阈值门控，生成与
学习重放采用一致受控策略，熵差独立对照，主评测无干预。下面仅是已有固定文献目录，
不表示已实现新版方法或完成对应新颖性复核。Entropy-Aware OPD在新稿中提出，待
本项目核对固定版本与公式；尚不计入本目录的九篇已归档参考。

核心来源包括 [Steering Vectors as a Training Signal](../papers/2610.40998v1_steering_vectors_as_training_signal.pdf)、[Rethinking OPD](../papers/2604.13016v2_rethinking_on_policy_distillation.pdf) 与 [CompassOPD](2609.10154v1_compass_opd.pdf)。当前已归档比较包括四篇同问题工作与五篇方法近邻，共九篇参考论文。

## 解决相同或重叠问题的工作

| 论文 | 本地 PDF | 解决的问题 | 阅读部分与边界 |
|---|---|---|---|
| Simple-OPD: Demystifying Warm-up for On-policy Distillation | [PDF](2608.06802v1_simple_opd.pdf) | 初始化不适合直接 OPD，怎样通过合理 warm-up 改善后续学习 | §3–§4；最直接冷启动基线，不代表所有跨 family 差异均已解决 |
| CompassOPD: Cross-Family On-Policy Distillation via Within-Family Likelihood Shifts | [PDF](2609.10154v1_compass_opd.pdf) | 对齐和教师轨迹 SFT 后，跨 family OPD 仍停滞 | §3–§4、§5.4；直接跨 family 信号重构 |
| Meet Students Where They Are: Student-Prefix Teacher Adaptation for On-Policy Distillation（SPTA） | [PDF](2611.03133v1_spta.pdf) | 教师不擅长处理学生前缀 | §3–§5、局限；主要测试相近谱系 |
| On the Off-Policy Teacher in On-Policy Distillation（SCOUT） | [PDF](2609.38360v1_scout.pdf) | 学生变化后，教师对当前前缀的指导不可靠 | §3.1–§3.2；在线教师适配，不等于任意异构 family 已验证 |

## Steering 方法近邻

| 论文 | 本地 PDF | 解决的问题 | 最值得核对的部分 | 为什么收录 |
|---|---|---|---|---|
| Activation-Conditioned Self-Distillation（ACSD） | [PDF](2609.38342v1_acsd.pdf) | 不靠每题参考解答或教师参数更新，怎样构造有效自教师 | §3.2–§3.4 | 激活方向构造教师，再做 OPD |
| Steering the Teacher: Improving On-Policy Self-Distillation for Thinking Models（ST-OPSD） | [PDF](2610.34005v1_st_opsd.pdf) | 参考条件教师过度引用、减少检查，损害学生学习 | §3.1–§3.2 | steering 修复特权教师的监督行为 |
| DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels | [PDF](2610.31438v1_dial.pdf) | 指令遵循强度难细调，固定方向缺少上下文适应 | §3.1–§3.3 | 学生侧 steering adapter 直接用 OPD 训练 |
| OPMD: On-Policy Mechanism Distillation Beyond Output Supervision | [PDF](2610.14022v1_opmd.pdf) | 教师状态数值对齐不保证学生后续计算可用 | §3.2；跨 family 扩展 | 构造学生能使用的内部辅助，再内化并移除辅助 |
| Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards（ASE） | [PDF](2610.27012v1_ase.pdf) | 普通 RLVR 采样重复、探索不足 | §3.2–§3.3 | 临时学生 steering 帮助训练，部署移除；**属于 RLVR 近邻，不是 OPD** |

详细比较见[同问题工作、方法近邻与当前方案的区别](../docs/02_paper_settings.md)。版本、哈希与阅读范围见 [manifest.json](manifest.json)，官方正文快照见 [sources/](sources/)。

这些论文的不同点是研究定位与实现路径的区别，不代表当前方案效果更好，也不代表新颖性已经成立。没有运行这些论文的训练代码或独立复现。

不参与当前比较的历史材料另存于 [archive/](archive/README.md)。
