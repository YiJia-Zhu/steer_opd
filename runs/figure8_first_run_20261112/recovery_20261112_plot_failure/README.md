# 2026-11-12绘图故障修复与恢复

用户11:47查询状态时，普通OPD和steer各完成1/50步。初始Base的AIME2025生成与评分
已在07:13完成：30题×16次，avg@16为5/480=1.0417%，无评分异常；这不是训练后方法收益。
随后`figure8_report.py`因缺少matplotlib退出，CPU异常传播导致supervisor停止，
旧`campaign.json`仍显示running。故障来源为绘图依赖和调度异常处理遗漏，checkpoint未损坏。

修复：

- 仅在项目`.venv-opd`安装matplotlib 3.10.9及其缺失的绘图依赖；torch/Transformers/vLLM版本不变。
- 结果JSON、CSV和README先保存；绘图失败只写`plot_status.json`，不阻断训练。
- 非必要报告子进程失败不阻断训练；评分本身失败仍会中止并明确记录。
- supervisor统一捕获CPU/GPU异常，写failed_resumable；状态命令核对PID、进程命令、cwd与启动时刻。
  缺失/复用PID及僵尸进程不再误报running。
- 新增4项回归测试，覆盖此次缺依赖路径、CPU异常、失效PID和非关键报告失败，全部通过。

11:54启动`supervisor_008_resume_20261112.log`，仍限制单张空闲GPU，实际选择GPU2。
现有训练步、256题两类校准文本、126对方向提取结果和480条基线评测均复用；
恢复从固定初始前缀/overlap诊断继续，然后进入第2步。配置、预算、损失和数据划分未改变。

`before.json`保存发现时的旧状态和基线成绩文件hash；`resume.json`保存恢复PID与日志。
后续以主目录`campaign.json`及`scripts/figure8_status.py`的当前结果为准。
