# 八方案实现复核审计 — 2026-11-13

本轮用户要求实验记入TODO，下一步明确同意后才执行；当前只复核代码、文档与CPU路径。

- [TODO](../../TODO.md)：固定模型/Slerp0.1/层26/8方案，以及执行前缺口。
- [逐项复核](../../docs/17_implementation_recheck_20261113.md)：实际后端、序列调度、Markdown一致性与未完成项。
- `before/`、`before_sha256.json`：本轮修改前快照。
- `backend_and_scheduler.json`：实际LLM调用参数、普通vLLM版本与EasySteer API缺失；无手工序列上限。
- `cpu_preflight.log`：新配置的CPU准备，状态awaiting_user_approval，100题候选、8组、仍缺35B权重。
- `cpu_tests.log`：相关CPU路径65 passed。
- `cpu_tests_project.log`：项目tests/全套200 passed、1 skipped，13.65秒，退出码0。
- `cpu_tests_full.log`：早期误将vendor测试纳入的命令日志，116项第三方收集错误；保留为失败记录，不作为项目验收。
- `validation.json`：最终源指纹、原科学定义保留、旧配置保留、job执行锁与各项未完成状态。

没有启动GPU、新模型生成/概率评分、训练、完整权重下载或旧任务恢复。
重点发现：Markdown§12.2要求完整分布firstByte聚合后sum(min)，当前代码Top16完整token字节交集不能代替；本轮未修改该科学定义。
新配置选择EasySteer，但T/Ref评分与SV提取仍HF，EasySteer GPU尚未验收；不能称全程使用或完整研究复现。
项目pytest仍有既有依赖警告和进程退出时ResourceTracker异常输出，命令本身返回0。
