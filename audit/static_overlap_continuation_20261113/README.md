# 静态overlap续接核对 — 2026-11-13

读取用户指定对话`rollout-2026-11-13T10-21-36-01a1d2b5-db99-7792-962b-552e321044f9.jsonl`，
最后决定为普通源rollout→教师同前缀批量评分→固定静态mask。此前只更新文档和规格。
本次核对源码、运行状态和缓存，补齐根README的当前说明及搜索文档中的实施缺口。
没有加载模型、运行GPU、新建mask或源前缀、执行新搜索、修改生产代码/旧配置或恢复训练。

## 已核对事实

- 静态规格包含192个互不重复的候选，8方案各24项，均为0更新。
- 74份受保护生产源码/配置与静态文档同步时的SHA256全部一致。
- 公共T/R/S池有55题，保存层为6/10/13/16/20；55份池文件SHA256均匹配manifest。
- 旧固定dev的16题select及16题confirm与公共池无交集，现有逐题bank不能直接覆盖这些旧dev行。
- 公共池学生参考文本的生成配置为temperature=0.7、top_p=0.95、logprobs=None，没有保存原生Top16。
- 旧候选和面板为failed_resumable；候选无已提交学习checkpoint。进程检查未发现对应生成、训练或监督进程。
  顶层campaign.json仍为历史running，不把该字段当作活跃进程证据。

## 具体实现位置和缺口

| 位置 | 已有能力 | 新静态搜索仍需完成 |
|---|---|---|
| [method_config.py](../../src/steer_opd/method_config.py) | 历史动态overlap、相对层深换算 | 明确静态方法/算子契约，保持旧标签语义 |
| [steering.py](../../src/steer_opd/steering.py) | 固定预测mask、加法hook、首预测t−1映射 | Slerp及零方向/零强度/范数/边界验收 |
| [overlap_tuning.py](../../src/steer_opd/overlap_tuning.py) | 同前缀加法评分与配对统计 | 每题向量绑定、静态来源校验、实际扰动记录 |
| [pre_opd_overlap_stage.py](../../scripts/pre_opd_overlap_stage.py) | 独立零更新评分 | 当前入口仅接受T−S/global/all；新8方案需独立入口或显式适配 |
| [extraction.py](../../src/steer_opd/extraction.py) | 捕获block输出，保存池化和token激活 | 补27层三角色及尺度，固定全部候选公共有效交集 |
| [plot_token_overlap_gates.py](../../scripts/plot_token_overlap_gates.py) | 历史动态probe/正式分布对比 | 静态整段无SV源基线、固定mask候选、默认前3条输出 |

新源前缀应覆盖共同有效训练题并预先冻结训练侧select/confirm。旧32题独立dev及历史产物保留。
不能从旧动态轨迹的probe伪造全程无SV源基线，也不能把校准回答没有保存的概率说成原生rollout概率。
Slerp仍未接入；EasySteer已知位置接口的本地适配仍未验收，当前同前缀评分不要求先迁移引擎。

阈值精确值与比较精度须一起验收。旧校准为0.37500000000000006；图示0.375仅为显示，
在离散Overlap@16边界上不同精度可能影响mask。此核对没有替换旧阈值或改写旧结果。

## 证据文件

- `readiness.json`：源码/缓存/状态摘要及来源对话。
- `code_config_before.json`：本次开始时74份源码/配置hash。
- `before/`：本次修改前README及搜索文档快照。
- `validation.json`：文档链接、网格及受保护文件核验结果。

后续仍需实现静态mask与Slerp、CPU语义验收、GPU模型路径验收及192候选零更新评分。
当前没有新实验结果；旧5条学习重放异常也未在此次核对中得到修复。
