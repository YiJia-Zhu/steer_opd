# Steering-assisted OPD

固定Qwen3.5-35B-A3B教师与Qwen3-4B学生，研究激活方向干预、师生分布相容性与OPD学习。保留学生T−S消融及教师S−T主线，具体已实现流程、可选路线和参数统一在文档中说明。

- [代码框架逻辑](docs/代码框架逻辑.md)：先看共同参数与全流程表，再看训练内部拆分、启动／恢复、验收与日志。
- [实验计划](docs/ablation.md)：实验授权、方案、选参与执行记录。
- [文档索引](docs/README.md)：研究稿、指标算例、耗时、文献及备份指南。
- [当前配置模板](configs/ablation.json)与[输出结构](输出结构.md)：配置入口、阶段目录及SHA复用规则。
- [TODO](TODO.md)：保留的任务检查记录。

手动CPU预览使用`bash scripts/run_ablation.sh --check`；它不启动模型。GPU运行需使用对应的显式运行入口和独立配置，模板`execution_hold`保留。新参数不能覆盖已完成身份，既有模型／数据资产只读；当前流程和历史结果不在本页重复维护。

全流程使用[`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh)，超参数及可选项集中在脚本开头。`--prepare-data`在CPU冻结1024题；`--check`预览；`--run`依次执行全局SV准备、all注入的全训练集overlap和OPD，默认Slerp0.1。全部1024题共用于SV、overlap和训练，无Select／Confirm；每step32题×4条回答，最多64步，每16步及末步评测裸学生。
