# 第10步无steer overlap先行评测

2026-11-12。用户询问初步overlap变化，既有流程将诊断排在长AIME生成后。
本次在空闲GPU7提前计算两个已完成第10步checkpoint的原定指标；正式GPU2继续原AIME任务。
最多两卡，诊断结束即释放GPU7；后续正式流程发现overlap.json存在会直接复用。

| 学生 | 步数 | 无hook Overlap@16 | 相对Base百分点 |
|---|---:|---:|---:|
| Base，共享初始化 | 0 | 59.29321554% | 0 |
| 普通OPD | 10 | 59.49203339% | +0.19881786 |
| T−S全局固定steer OPD | 10 | 59.47657943% | +0.18336389 |

steer减普通OPD为**−0.01545396个百分点**，当前点估计没有相对普通OPD的提升，差距很小。
两组均关闭hook、采用相同32个独立dev问题的初始Base固定回答前缀，累计35994个预测位置，
复用相同冻结教师评分；没有使用AIME答案或重新选择方向/层/强度。
这是训练后撤去干预的结果，未测量本次带hook即时变化。
只有一个训练seed、一个早期checkpoint，未做题目级显著性检验；不将这个微小差异称为确定劣势。
两组accuracy尚未齐，当前不能声称达到相对普通OPD同时提高overlap/accuracy的目标。

原始结果：`result.json`；每组进程/PID/GPU/退出及stdout记录在本目录对应子目录。
正式结果：`runs/figure8_first_run_20261112/evaluation/{vanilla,steer}_0010/overlap.json`。
报告脚本新增支持独立展示已完成overlap与尚待评测accuracy，不把缺失accuracy填为0。
训练、采样、评测协议和正在运行的进程保持原流程。
