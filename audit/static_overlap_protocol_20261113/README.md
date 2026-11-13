# 静态overlap协议文档同步 — 2026-11-13

用户因动态门控成本改用静态overlap：普通rollout→同前缀教师批量评分→冻结idx。
保留8方案及加法/Slerp、4强度、3单层，共192非零候选及共同基线。

- before/：本次修改前文档/模板及先前参数规格，保留历史。
- grid.json：当前静态搜索规格与192项清单；不作为旧Method可执行配置。
- unchanged_code_config_before.json：本次前74份生产源码/配置SHA256，验证本轮未修改代码或旧配置。
- document_changes.json、validation.json：文档变更与完整性核验。

当前先做同源前缀的零更新比较。学习前向注入/二次受控生成分别列明，未自动选择或启动。
EasySteer已知mask接口可以表达固定位置，本地迁移、Slerp和静态适配尚待实现/验收。
未加载模型或运行GPU；旧5条重放失败、正式更新0的历史事实不改写。
