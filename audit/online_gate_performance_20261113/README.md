# 在线overlap门控耗时与源码可行性复核

2026-11-13，仅读取历史任务、统计CPU数据及保存固定commit源码，没有新GPU推理或训练。

- timing.json：已完成256条rollout的原始来源、起止时间、生成循环、吞吐、拆批/重组批、注入率。
- fork_sources.json及fork/：EasySteer-vllm-v1固定commit的runner、worker manager、selector和Qwen3源码及SHA256。
- validation.json：耗时计算、回答token数、拆批事件和外部源码哈希复核。
- before/：本次计划文档更新前快照。

子进程墙钟2小时48分30秒，内部生成循环2小时48分26秒；337427个输出token，约33.375 token/s。
在线教师评分已包含。31次显存拆批（15次OOM、16次预防压力）、24次重新组批。
本批是旧双层T−R候选，不能直接外推新单层192候选的耗时；后续学习重放失败、正式更新0。

完整优化路线及实施边界见[docs/11_online_gate_acceleration.md](../../docs/11_online_gate_acceleration.md)。
源码修改方案尚未实现/性能验收，没有具体加速倍数或EasySteer迁移完成的结论。
