# 实现再次复核 — 2026-11-14

用户要求更新TODO、等待明确同意后执行；现在只核对代码/配置/Markdown及CPU测试。
本轮未修改生产Python源码、科学配置或冻结job，不下载权重、不运行GPU、不生成/评分新模型回答、不训练。

结论：八方案结构与本轮配置一致，完整Markdown方案尚未复现。用户刚接受的全支持下一字节sum(min)指标尚未接入；当前执行指标仍是Top16完整token字节交集。
旧CPU诊断测试通过不表示这一新定义已经实现或验收。

| 核对项 | 实际结果 |
|---|---|
| 模型与注入 | T=Qwen3.5-35B-A3B，Ref=0.8B，S=Qwen3-4B；36层、Slerp0.1、75%层26 |
| 八方案 | T−S/T−Ref × all/static_overlap × global/per_question；4份bank、共同基线 |
| 静态性 | 来源mask冻结，候选不反馈重判；仍0更新 |
| 执行锁 | 配置及5份阶段job均awaiting_user_approval，前置blockers保留 |
| seq参数 | 生成/评分LLM及其局部字典没有手工序列上限；配置/job也没有，入口继续拒绝新增上限 |
| vLLM/EasySteer | 生成vLLM；T/Ref评分及SV提取HF；8组/基线选EasySteer，支持的本地fork尚未验收 |
| 字节指标 | 完整原生概率聚合、EOS/控制类别、源/候选缓存与评分集成均待补，不能以Top16替代 |
| 学习目标 | 本轮无学习；旧Student Top16保持，跨词表sampled-span目标未在此面板接训练 |
| 其它前置项 | 35B完整权重缺失；GPU重放/Slerp/selector、EasySteer扰动、权重内容指纹等未验收 |
| 完整研究 | 当前学生刷新、训练对照/多seed/撤去SV能力等未在该新配对复现，不进入自动队列 |

`review.json`记录拥有源码的hash、LLM构造参数、运行环境元数据、8种组合及实际缺口。
`cpu_tests.log`记录本轮项目tests/回归：200 passed、1 skipped、13.72秒，退出码0；结果与范围同时写入review.json。
既有依赖警告及退出时ResourceTracker异常输出保留在日志中；没有GPU测试。
[TODO](../../TODO.md)保留正式执行前的补齐清单；[逐项MD对照](../../docs/17_implementation_recheck_20261113.md)保留科学定义和历史证据。
