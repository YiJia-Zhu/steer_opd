# 默认逐token对比图与8格搜索候选

用户要求：后续默认画注入前/后overlap、实际steer散点与差值，提出T−S/T−R × 全位置/overlap门控 × 全局/逐题的8格搜索，并先给强度和层集合候选供挑选。

- 绘图脚本默认开启comparison；旧`--comparison`兼容，只有显式`--probe-only`才画历史单线图。
- campaign在正式rollout完成后、教师评分及训练之前自动调用CPU绘图；已完成rollout恢复时同样补图，原清单前3条。
- 新HF批量/参考job默认保存比较probe；全位置额外无梯度分支不写入正式KV或采样RNG。原已存在job的协议字段不改。
- 全位置/普通OPD也可画配对数据；只有overlap方法画阈值线，零向量及无SV明确处理。缺少真实probe的历史产物不伪造曲线。
- 已用不带`--comparison`的真实CLI重画现有3条到step目录，验证PNG/SVG/CSV/3页PDF、comparison默认、原回答SHA256不变。
- CPU回归67 passed、2 warnings。覆盖探针KV隔离、正式策略和采样不变、带诊断probe的32-token恢复、自动绘图的新生成/缓存复用及旧job字段保留。
- 未进行新GPU任务、未启动参数搜索，现有重放异常未解决。新增全位置GPU记录路径仍需真实模型验收。

候选与定点论文证据见[方案](../../docs/10_overlap_parameter_search.md)，机器记录见summary.json、source_manifest.json、cpu_tests.log。
