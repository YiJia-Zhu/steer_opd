# 用户固定的下一步overlap搜索网格 — 2026-11-13

用户固定T−S/T−R × 全位置/overlap门控 × 全局/逐题8方案；每方案比较加法和Slerp，
两者强度均0.01/0.1/0.3/0.5，全部单层50%/75%/100%。共192个非零候选，加共同无干预基线。

- `grid.json`：结构化规格与192个完整候选，不是可启动campaign配置。
- `before/`、`document_changes.json`：修改前文档和本次前后SHA256。
- `literature_sources.json`、`sources/CAA/`：CAA固定commit的源码快照，官方7B逐层扫描含0..31及最后block。
- `validation.json`：候选数量、组合完整性、层号换算及文档链接核验。

本次只改计划文档并保存规格。未改生成/训练源码、旧配置或权重，未执行GPU或训练。
当前项目仍缺Slerp算子，最后层需要新激活/尺度与验收；旧5条重放失败仍未解决、正式更新0。
