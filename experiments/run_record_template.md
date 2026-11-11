# 单次运行记录

模板更新：2026-11-12。对应[当前执行计划](../docs/03_execution_plan.md)，待运行时复制填写。
不要在结果出现前填入预期数值。主方法是T−R＋overlap单阈值，生成/更新受控定义一致，主评测无干预。

## 角色、版本与轨道

- run ID / 日期 / 当前阶段 / 明确的恢复与预算授权：
- 师生配对 / 同家族或独立家族 / 兼容tokenizer或扩展路径：
- T、R、S0完整repo ID / revision / thinking模式 / 后训练来源：
- assets链接与实际路径 / 权重或配置校验范围：
- 原始起点或共同SFT起点 / checkpoint / 冻结学生锚点（若使用）：
- 总参数 / 激活参数 / 文本主干 / 可训练参数：
- 原论文对应 / 本项目方法变更 / 源码commit / 环境版本：
- GPU / dtype / 微批与并行 / 全局batch / 实际optimizer更新次数：
- native模板 / 任务指令 / tokenizer映射 / 特殊与停止事件 / 合法输出支持：

## 数据与方向

- D_vec / D_opd / D_dev-select与confirm / D_test ID文件、hash和题目数：
- 跨split去重 / 长度过滤 / 原论文30K题集对应 / 100题循环访问规则：
- T/R/S0三文本缓存 / 完整、截断、退化、正确性及共同过滤集合：
- 正文mask / pooling / 每题权重 / 三方向共同协议：
- T−S、T−R、R−S原始范数、夹角、分解误差 / 归一化文件：
- 单层block精确位置 / 0-based层号 / 校准激活尺度s_l / rho：
- 随机或负方向 / 随机种子 / 层与强度选择预算 / 无干预dev选择依据：
- 提取是否训练前一次冻结 / 初始化改变后是否重新提取：

## 门控、受控生成与学习

- 策略：无注入 / 全位置 / overlap单阈值 / 因果后段 / 独立熵差 / 随机门控：
- 主门控k=16 / tau_O / 校准前缀 / 初始触发率 / 固定阈值记录：
- 独立熵差对照tau_H / 有符号及绝对熵差 / full-vocab或top-K熵口径：
- 后段固定b的训练侧来源 / 在线随机匹配规则 / 实际匹配误差：
- 是否无AND/OR组合 / 固定rho / 门控停止梯度 / 不裁剪loss位置：
- probe是否仅当前步关闭 / 历史受控KV / probe与正式缓存隔离：
- 预测状态t−1 / 首回答边界 / padding/packing与EOS mask：
- 真实行为策略 / rollout温度与截断 / 采样token和行为log-prob保存位置：
- g轨迹、token、prefix hash / 逐步与teacher-forcing重放一致性误差：
- loss前向是否重放历史干预 / backward重算同mask / 参数更新与新鲜rollout：
- 目标：full-vocab RKL / 原Student Top-K参考 / 统一近似 / span或byte扩展：
- loss归约（逐题/逐token）/ 支持集与tail/归一化 / 数值与梯度校验：
- 内循环次数 / 门控冻结局部近似 / 学习策略陈旧与修正：
- 若为loss-only、rollout-only、滞后门控或末尾修复：独立命名与精确定义：

## 评测、成本与证据

- 关闭T/R/steer/在线门控的主评测checkpoint / 预先冻结解码协议：
- avg@n、pass@1或pass@n定义 / 题目级配对区间 / 训练与评测seed：
- 带干预推理结果（只作机制分析，单独列出）：
- 固定相同前缀与自由轨迹的probe/受控/完全无hook分布：
- Top-K集合overlap / 共享Top-K概率质量 / 全分布相交系数（分开命名）：
- 有符号/绝对熵差、KL / 触发率与位置 / 每题注入数：
- 回答长度 / 截断 / 重复 / 有效答案率：
- 离线T/R/S生成及S0读取token / 搜索与SFT数据预算：
- 在线T查询 / R或锚点评分（若用Compass）/ 学生probe及重算 / 训练token：
- 墙钟时间 / GPU-hours / 峰值显存 / KV与批处理开销 / 评测成本：
- 固定学习预算与固定实际计算预算的比较差异：
- 日志、process/campaign退出状态、checkpoint及逐题结果路径：
- 实测结论 / 失败与限制 / 下一步或停止理由：
