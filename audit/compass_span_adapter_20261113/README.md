# Compass式span适配审计 — 2026-11-13

用户要求参考CompassOPD代码仓库。未在论文v1、GitHub名称检索或交叉tokenizer工作清单中找到官方仓库，已请求用户提供URL。
本轮是依据论文附录A.4的独立实现，不声称官方代码复现。新增的字节续写Top16交集指标是本项目适配，未归属于Compass原方法。

实现说明：[docs/16_compass_alignment_eight_way.md](../../docs/16_compass_alignment_eight_way.md)。

- `reference_status.json`：仓库检索范围/结果、论文版本与阅读依据。
- `paper_a4_excerpt.txt`：本地固定正文附录A.4的原文摘录。
- `teacher_metadata/source_manifest.json`：35B教师官方元数据，固定revision和文件hash；不含权重。
- `before_sha256.json`及`before/`：本轮开始的源码/旧配置指纹与关键文件快照。
- `cpu_tests_final.log`：最终全套CPU回归，186 passed、1 skipped（13.31秒）；早期完整与局部复查日志保留。
- `validation.json`：最终文件指纹、旧配置保留、原生支持/模型类型、8方案与CPU预检状态。

本轮仅CPU实现、检查和job准备；没有启动GPU、生成新参考回答、做新模型8组评分或更新模型。
预检在runs/compass_static_eight_20261113/preflight.json，当前只有35B教师完整权重缺失；这一记录不代表GPU数值路径已验收。
