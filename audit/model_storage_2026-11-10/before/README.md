# Steering-assisted OPD 实验准备

整理日期：2026-11-07；实验启动计划更新于2026-11-08。研究主线围绕以下两篇论文组织材料，最近邻工作另列在 [ref/](ref/README.md)：

1. **Steering Vectors as a Training Signal in LLM Post-Training**，2610.40998v1。
2. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**，2604.13016v2。

当前研究主线：**学生正常生成；仅在早期蒸馏训练前向加 steering；短程辅助后撤去，继续普通 OPD；最终评测普通学生。**

## 从这里开始

**当前研究目标和TODO以[AGENTS.md](AGENTS.md)为准。** 11月10日用户明确优先级为：
先诊断真正跨family的overlap，再测学生steering对相同前缀的影响，最后按实际学习收益
选择方向并做OPD，与SFT/LoRA及Compass比较。实施口径见[跨family研究与执行约定](docs/12_cross_family_research_contract.md)。
第10篇的同family先行排程是历史方案。

**当前实验与资产下载暂停。模型/数据选型优先对应Rethinking、Simple和Compass原论文，
最新对应表与缺口见[论文对应方案](docs/14_paper_aligned_models_and_data.md)。**
主配对建议为Compass已有的Qwen3-30B/0.6B参考→Granite4.1-3B，同教师→Qwen3-4B作对照。
Phi/Smol已移出正式主线，已有探索结果保留，不再据旧队列自动运行。

历史探索基线共16题×3位置×3学生、144个状态，随后完成Phi/Smol真实/随机/负方向
诊断执行；记录见[跨family首轮诊断](docs/13_cross_family_diagnosis_20261110.md)。
这些结果不替代论文对应新配对，尚无跨family参数学习收益结论。

历史初始学生/教师MATH-500 avg@4分别为85.30%/92.50%，环境恢复经过见
[首轮结果与续跑记录](docs/11_pilot_status_2026-11-10.md)。当前全部实验暂停，尚无steering训练后效果结论。

历史工程验证从 [MATH-500 首轮实验与 EasySteer 计划](docs/10_math500_launch_plan.md) 开始，复用本地 R1-1.5B 学生和 JustRL-1.5B/R1-7B 教师；MATH-500 保留测试。原 DAPO 训练 Parquet 去重并排除4个精确重叠、9个词面近重复和12组答案冲突后保留17,162题，见[固定划分](data/math500_pilot_v2/split_manifest.json)。当前跨family继续复用该数据划分，状态与限制见[运行记录](runs/README.md)。

准备实验前，先看11月8日补充的[新颖度扩大调查](docs/09_novelty_audit_cross_family_steering_opd.md)：分别核对跨 family OPD、steering＋OPD，以及 LastOPD / Latent-MOPD 等早期内部辅助近邻；“未发现完全同流程”不等于已确保创新性。

| 顺序 | 文件 | 用途 |
|---|---|---|
| 1 | [完整实验章节](docs/06_experiment_chapter.md) | 模型配对、数据划分、baseline、超参、跨 family 接口与资源 |
| 2 | [论文模型核对与下载](docs/07_models_datasets_download.md) | Rethinking / Simple 原配对，HF ID、访问状态、大小、下载命令 |
| 3 | [代码基座与修改计划](docs/08_codebase_plan.md) | 一个 verl 主干，借用 SV 与 Simple 模块；固定 commit 与实施里程碑 |
| 4 | [最小实验方案](docs/02_experiment_plan.md) | 最小对照、评测与实现边界 |
| 5 | [Idea 故事与方法](docs/01_idea.md) | 问题、动机、向量计算、待验证假设 |
| 6 | [两篇论文阅读笔记](docs/03_paper_notes.md) | 各自做了什么；原始模板和阅读定位 |
| 7 | [OPD insights](docs/04_insights.md) | 仅依据这两篇整理的观点、证据和边界 |
| 8 | [同问题工作与方法近邻](docs/05_related_work.md) | Simple、Compass、SPTA 与 steering 近邻的区别 |
| 9 | [实验矩阵](experiments/experiment_matrix.csv) / [分阶段执行清单](experiments/stage_matrix.csv) | 组别定义与 P0–P3 优先级；不是启动配置 |
| 10 | [下载清单](experiments/download_manifest.json) / [运行记录模板](experiments/run_record_template.md) | 固定版本、数据、hook、预算与结果 |

## 论文

- [Steering Vectors as a Training Signal — PDF](papers/2610.40998v1_steering_vectors_as_training_signal.pdf)：先看 §4.1、§4.3、附录 C.1；§5.1 用于区分 GRPO 分支。
- [Rethinking OPD — PDF](papers/2604.13016v2_rethinking_on_policy_distillation.pdf)：先看 §3–§6，重点为 §5.1 冷启动、§5.2 模板与 §6.2 失败解释。
- [版本、来源与哈希清单](papers/manifest.json)；[官方正文快照](papers/sources/)；[材料检查结果](audit/package_validation.json)。
- [九篇参考论文 PDF 与阅读入口](ref/README.md)：四篇同问题工作、五篇方法近邻；[版本与哈希清单](ref/manifest.json)。

## 已准备与待落实

已准备两篇核心论文、九篇主参考论文、九份研究与实验文档、实验矩阵与记录模板；补充了24项模型/数据资源、九个下载 profile 和固定版本的代码修改计划。新颖度扩大调查另保留35篇官方正文及分层阅读记录，不计入原九篇参考 PDF。不参与当前比较的历史论文存于 [ref/archive/](ref/archive/README.md)。原始讨论文档仍在原目录保留。

已有R1/JustRL等模型不重复下载。旧download profiles保留追溯，当前选型和资产缺口以
第14篇为准，未明确恢复前不进行下载或GPU实验。显式top-k KL与论文原目标的差异须标注，
不将旧工程链路直接称为论文完整复现。

11月7日材料中的“尚未下载”是当时状态。11月8日已核实本地模型权重文件与数据，环境和训练语义仍需逐项验证。旧下载清单用于补缺，不重复下载已有共享资产；最新选择及运行顺序见第10篇。文档中的组别与假设不代表实验结果。

预览首批下载项（不下载文件）：

```bash
python scripts/download_assets.py --profile pilot-r1
```

准备实际下载时加 `--execute`，其他配对与操作说明见[下载文档](docs/07_models_datasets_download.md)。

可以重跑材料检查：

```bash
python scripts/validate_package.py
```
