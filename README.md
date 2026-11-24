# Steering-assisted OPD

固定Qwen3.5-35B-A3B教师与Qwen3-4B学生，研究激活方向干预、师生分布相容性与OPD学习。训练支持教师空间的静态 SV 和动态 SV。

- [代码框架逻辑](docs/代码框架逻辑.md)：先看共同参数与全流程表，再看训练内部拆分、启动／恢复、验收与日志。
- [实验计划](docs/ablation.md)：实验授权、方案、选参与执行记录。
- [文档索引](docs/README.md)：研究稿、指标算例、耗时、文献及备份指南。
- [当前配置模板](configs/ablation.json)与[输出结构](输出结构.md)：配置入口、阶段目录及SHA复用规则。
- [TODO](TODO.md)：保留的任务检查记录。

## 环境安装

项目使用 Python 3.12 和两套独立虚拟环境。`.venv-opd` 用于 veRL 训练，`.venv-easysteer` 用于激活提取和教师评分。GPU 运行使用 CUDA 13.0。全新安装、服务器迁移和 CPU 验收步骤见 [安装与迁移说明](INSTALL.md)。

## 实验入口

手动CPU预览使用`bash scripts/run_ablation.sh --check`；它不启动模型。GPU运行需使用对应的显式运行入口和独立配置，模板`execution_hold`保留。新参数不能覆盖已完成身份，既有模型／数据资产只读；当前流程和历史结果不在本页重复维护。

动态 SV 用 `bash scripts/run_dynamic_sv.sh --check` 预览。训练直接读 step 0 冻结题集。每步生成 32 题 × 4 条学生回答和 32 条教师参考，由同一个教师读取两组文本，提取 S−T 后沿原学生回答注入评分。[动态 SV 入口](scripts/run_dynamic_sv.sh) · [方向定义](docs/ablation.md#动态-sv)

全流程使用[`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh)，超参数及可选项集中在脚本开头。`--prepare-data`在CPU冻结1024题；`--check`预览；`--run`依次执行全局SV准备、all注入的全训练集overlap和OPD，默认Slerp0.1。全部1024题共用于SV、overlap和训练，无Select／Confirm；每step32题×4条回答，最多64步，每16步及末步评测裸学生。

无 steering 的训练对比共用 `scripts/baselines.sh`。脚本开头的 `BASELINES=(vanilla compassopd)` 决定串行顺序，每个方法从同一初始学生独立训练。默认 `bash scripts/baselines.sh --check` 在 CPU 预览两组。只运行纯 CompassOPD 可用 `METHODS=compassopd GPU_IDS=0,4 bash scripts/baselines.sh --run`。[启动脚本](scripts/baselines.sh) · [方法与配置](docs/ablation.md#无-steering-的训练基线)

per-sample及其机制／参数消融见[提取协议](docs/ablation.md#提取与评分分开)：学生每题先rollout 4条，冻结教师逐条读取后取均值，减去教师参考回答的池化值，每题构建一个S−T方向；四条只用于提取，overlap仍沿用原每题一条固定回答。该协议已实现，437项CPU回归及师生GPU验收通过，16组矩阵已完成。

模型也每16步及末步保存，各版本权重均保留；优化器等恢复状态只留最新完整一份，批次过程数据和当前overlap状态滚动更新，loss／overlap／准确率曲线继续保留。

16组S−T消融已于2026-11-19 04:16完成，结果统一保存在项目`outputs/`，见[汇总分析](outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1/analysis.md)及[完整CSV](outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1/comparison.csv)。4类bank和4096条提取rollout保留在对应step目录；该汇总目录的`metadata/implementation/`保留完整源码快照与补丁。隔离工作树已删除，训练使用的主目录源码保持不变。

原全流程的1024题overlap已完成。旧OPD在2次更新后因生成字节的UTF-8对齐错误退出；修复通过429项CPU回归及包检查，2026-11-19已在tmux ste中以新身份`opd_n1024_project_outputs_utf8_v2`从初始学生重跑。训练仍使用GPU2／3、原SV和相同超参数，输出位于项目`outputs/step_4_train/`；启动及验证凭据保存在audit/opd_utf8_restart_20261119。
