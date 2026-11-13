# 训练前静态overlap调参与结果检查

2026-11-13更新：用户因在线动态门控开销过大，改用rollout后计算的静态overlap位置。
本文件替代先前仅T−S/全局/全位置、5层×3强度及多层补扫的起步提案；旧配置和产物保留，不直接复用为新网格。
完整规则见[静态协议](12_static_overlap_protocol.md)，参数见[下一步实验](10_overlap_parameter_search.md)。

## 1. 目标与更新边界

冻结真正Qwen3-1.7B-Base为S0，教师Qwen3-4B Non-thinking，R为Qwen3-0.6B。
只检验同前缀有/无SV的即时Overlap@16增量，所有候选OPD更新数=0。
参考文本生成、无梯度提取、普通源rollout、教师评分和固定mask前向不计参数更新；不做backward/optimizer.step。
源回答已经完成后才计算mask，后续注入不倒改源rollout；当前诊断使用同一回答的固定前缀重放。

## 2. 固定网格

T−S/T−R × 全回答预测位置/静态overlap位置 × 全局均值/逐题，共8方案。
每方案比较加法rho与Slerp alpha，两者均[0.01,0.1,0.3,0.5]；全部单层50%/75%/100%。
28层学生对应0-based 13/20/27；共192个非零候选，加共同无干预基线。
本轮提取者S0固定，参考文本/过滤/尺度匹配；不添加刷新频率和双层搜索轴。

## 3. 数据和静态mask

原100道OT3为方向/未来OPD候选，已有共同有效交集55题；最后层提取若改变交集，各方法共用新交集并报告数量。
逐题方向使用本题训练参考；训练侧select/confirm与源回答在扫描前固定，不在独立dev/test题上提取答案方向。
原dev128与AIME30保留独立，AIME不选阈值/层/强度；旧32题dev的S0-global诊断另作历史结果。

S0无SV rollout保存原生Top16，教师在相同token前缀批量评分并保存自身Top16。
`O_src[t]=|Top16(S0(.|c_src[t])) ∩ Top16(T(.|c_src[t]))|/16`；`g_static[t]=1[O_src[t]<tau_O]`。
阈值预先在训练侧冻结，mask附checkpoint/源token/hash/输出支持/预测位置映射。
同一源回答的不同方向、粒度、层、强度候选共用静态mask，注入后overlap不反馈修改idx。

## 4. 指标与默认图

在相同源前缀和有效位置，计算`DeltaO=100×(mean(O_steer)-mean(O_noSV))`，单位百分点。
主排名题目等权并给配对区间，token加权、静态触发率、触发位置增量、共享质量与扰动另报。
全位置组g恒为True，静态组使用保存mask；零向量的实际注入为False并保留请求mask。
默认画源清单前3条：整段无SV蓝线、固定mask注入后橙线、实际注入散点和差值，保存PNG/SVG/CSV/PDF及hash。
这是同状态即时分布比较；不把带hook效果叫作训练后无hook学习收益。

## 5. 复用、验收与输出

身份匹配时复用源rollout、教师Top16、静态mask及13/20激活；27需新提取本层三角色激活/尺度。
优先批量固定mask前向，核验首回答预测边界、阈值、padding、方向/算子、0强度及移除hook恢复。
Slerp、静态调度适配和EasySteer路径尚未GPU验收，原动态gate CPU/GPU结果不替代这些检查。
输出完整候选表、逐题/逐位置数据、八方案最优设置及confirm结果后交用户检查。
当前只改文档，未启动新搜索/训练；后续学习阶段和目标未确定前不自动再生成或切为loss-only。

## 6. 历史证据

旧第10步无hook同32题dev、35994位置：Base59.29321554%，普通OPD59.49203339%，steer OPD59.47657943%，
steer减普通−0.01545396个百分点，不支持已建立steering收益。[第10步审计](../audit/step10_overlap_2026-11-12/README.md)。
旧双层T−R动态首步256条生成耗时2小时48分30秒，后续5条重放失败、正式更新0；
历史动态probe/正式分布的图及读数不能改称当前静态无SV源基线。[耗时记录](11_online_gate_acceleration.md)。
