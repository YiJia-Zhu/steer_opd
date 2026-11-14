# Steering-assisted OPD

当前准备Qwen3.5-35B-A3B教师、Qwen3.5-0.8B参考、Qwen3-4B学生的八方案零更新筛选。固定Slerp0.1与75%单层26，比较T−S/T−Ref × all/static_overlap × global/per_question，共用无SV基线。

同时记录Rethinking跨family Top16适配overlap与完整词表下一字节概率P_overlap，分别画逐token曲线。代码已接EasySteer生成、指定层capture、T/Ref原生评分和静态mask前向；真实GPU验收与性能测量待做。

实验保持awaiting_user_approval；这轮只改代码、CPU验证与清理，0次模型实验更新。35B权重已由用户准备，14分片结构与索引检查通过，项目链接已补齐；启动时记录完整内容哈希。当前配置为[static_dual_overlap.json](configs/static_dual_overlap.json)，用法见[实验协议](docs/experiment.md)。

手动启动用[scripts/run_static_panel.sh](scripts/run_static_panel.sh)，开头可选GPU、方案与超参数。先运行`bash scripts/run_static_panel.sh --check`；准备好后用`--run`启动验收及面板，同一命令可恢复；`--status`查看进度。

[文档索引](docs/README.md) · [完整研究方案](docs/Steering_OPD_完整研究方案.md) · [实现说明](docs/implementation.md) · [TODO](TODO.md)

仓库仅保留当前静态实验：scripts/含启动/环境安装/验收/生成/提取/评分/绘图编排，src/steer_opd/含其共享实现，configs/只有一个模板，tests/只覆盖这些路径。旧pilot、Figure8、full-scheme、训练/动态门控及模型数据下载代码已删除。

文献与固定源码仍在papers/和ref/；第三方仓库自带的原始数据和示例权重保留原位置，已移除其训练源码和旧评测输出。当前100题、划分、共享模型/数据和隔离运行环境保留。清理范围见[删除清单](audit/repository_cleanup_20261114/manifest.json)。
