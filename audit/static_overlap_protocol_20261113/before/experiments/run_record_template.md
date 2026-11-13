# 单次运行记录

模板更新：2026-11-12。对应[主计划](../docs/01_research_plan.md)及[执行计划](../docs/03_execution_plan.md)。
完整主线2–2–2为T−R、overlap单门控、当前学生在线逐题；各备选按实际配置填写，不预填预期收益。
现有首轮T−S/全位置/全局固定仍按原运行记录，不用本模板改写已发生的实验条件。

## 当前阶段专用：0更新overlap调参

按[训练前调参协议](../docs/04_pre_opd_overlap_tuning.md)记录；本阶段训练字段标为不适用。

- 原始Base checkpoint/hash，OPD更新数=0；无backward/optimizer.step或真实更新验收证据：
- 固定同题100候选/有效数/过滤ID：
- 固定前缀/hash/有效位置，select/confirm ID预定划分，教师缓存/支持集：
- 差分/全局或逐题/S0，层0-based索引集合与层数，每层方向/尺度/hash：
- rho_total/各层rho，实际delta范数及delta/激活比；正/负/随机seed：
- 无hook基线与带hook即时overlap、Delta百分点；逐题等权与token加权、逐题差值/区间：
- 共享质量、位置/长度分桶、alpha0与移除恢复、mask/首回答边界/数值检查：
- select完整网格、预选候选、confirm结果；补扫是否另记预算而非继续用confirm调参：
- 逐题数据、原始候选/概率文件与向量版本；结果链接及用户检查状态：
- 完成即退出；用户明确允许下一步前未启动OPD/旧campaign/50步/新面板：

即时改善不证明参数学习；S0未更新时去hook应回到基线。下面为后续OPD获准后使用的完整模板。

## 角色、版本与范围

- run ID / 日期 / 阶段 / 明确的执行授权与预算：
- 是否完整2–2–2 / 哪些因素变化 / 原论文对应 / 新实验或适配实现：
- T、R、轨道初始S0完整repo/revision、thinking与后训练来源：
- 本轮当前学生checkpoint_id、已提交更新步、行为学生版本与checkpoint hash：
- assets链接与实际路径 / 权重校验范围 / family与tokenizer边界：
- 总参数、激活参数、文本主干及可训练参数 / GPU、dtype与并行：
- 全局batch、微批、n、更新步与实际optimizer步数 / 代码commit或文件hash / 环境：
- prompt/template、任务指令、EOS/stop、合法输出支持及排除质量：
- 基础OPD损失与有效token归约 / 相对原配方和现有首轮的明确差异：

## 数据与训练参考信息

- D_opd题数、原source/index、question_id文件、题干/编码hash、去重及长度规则：
- 提取与OPD同题 / 独立数据全局D_vec / 同数据全局使用D_opd（按实际选择）：
- D_dev-select/confirm与D_test的ID、hash、题数及互斥检查：
- 测试题是否不生成参考、不提取答案向量、不参与阈值/checkpoint选择：
- 每题T/R参考生成模型、模式、参数、seed、文本/编码及正文mask/hash：
- 完整/截断/退化/正确性/共同过滤及失败处理；实际有效数与候选题数：
- T−S/R−S是否先生成当前无hook学生bootstrap / 使用S0缓存文本则明确标变体：
- 参考文本缓存索引与信息覆盖 / 各基线能否访问同一教师及T/R两源文本：

## 提取粒度、时机与逐题方向

- 方向：T−R / T−S / R−S / 随机 / 反向：
- 粒度：全局 / 逐题；提取题：独立训练集 / 同OPD题：
- 提取者：S0 / 当前S_theta_k；刷新：训练前一次 / 每轮rollout前 / 明确周期：
- 本轮batch_id / 唯一question_id列表 / 同题n个rollout复用规则：
- 当前学生提取时eval/no_grad、无hook、参数不更新的核验：
- extraction_id、student_checkpoint_id/步数、参考hash、层、模板、池化/mask、dtype：
- 每题原始差分/范数、三方向夹角及分解误差、近零/有效状态：
- 实际归一化向量、校准尺度s_l、rho、vector_id与文件hash、存储精度：
- 全局是否平均原始差分后再归一化 / 未将单位向量均值混作相同协议：
- 本题/随机他题控制的对应表、置换seed、向量集合与尺度匹配：
- 同数据全局vsS0逐题缓存：哪些字段一致，是否仅改变粒度：
- S0逐题缓存vs当前学生重提：是否保持题目/文本/注入，刷新读取成本：
- 同题初始/当前方向漂移与固定同前缀比较 / 层、强度与dev选型预算：

## 门控、受控生成与学习

- 注入：全位置 / overlap单阈值 / 独立熵差 / 因果后段 / 随机或位置匹配：
- k=16、tau_O、训练侧校准来源、初始/实际触发率、固定阈值记录：
- 独立熵差tau_H / full-vocab或top-K条件熵 / 无overlap与熵差AND/OR组合：
- 后段固定b或在线随机匹配规则 / 是否仅使用因果已有信息：
- probe是否仅当前步关闭，历史受控KV与正式分支隔离核验：
- trajectory_id、question_id、batch_id到vector_id与提取checkpoint的绑定：
- prompt/响应token、记录mask、行为log-prob、支持候选ID及教师评分/mask：
- 生成与学习使用同一个已保存向量、同一历史mask；t−1/首回答边界/padding/packing：
- 同参数同向量下逐步生成/teacher-forcing重放误差，非零方向/alpha0/移除恢复：
- backward重算是否保留同题方向/门控，提取过程/方向/教师是否停止梯度：
- g=0仍接受监督 / loss实际受控Top16与门控probe Top16不混用：
- Student Top-K16、student_p、token_reward_direct/PPO / 其它目标单独命名：
- fresh rollout、内循环次数、门控/方向冻结局部近似及策略陈旧：
- loss-only、rollout-only、滞后门控或末尾修复：独立变体与预算定义：

## 持久化、断点与恢复

- 文本缓存、逐题向量分片/索引、本批绑定、rollout/评分文件的位置与hash：
- 文本/提取产物原子提交；rollout是否只引用完整已提交vector_id：
- checkpoint中的模型、optimizer/RNG、数据顺序/游标、已提交步及配置hash：
- 已生成而未更新轨迹恢复时是否沿用原向量/mask/行为学生checkpoint：
- 若废弃重提或重采样：明确原因、被废弃batch/轨迹ID与新的版本，未静默覆盖：
- 重复题、题目排序变化、batch/packing向量关联与缓存版本命中检查：
- 向量/参考/轨迹保留策略，是否在相应更新提交前保留全部恢复依赖：
- process/campaign退出、真实PID/启动时刻、完整产物与checkpoint原子提交证据：

## 无干预评测、成本与结论

- 主评测是否只给学生题目，无T/R、无参考文本、无向量/提取/门控：
- checkpoint与预先冻结解码/评分协议；avg@n/pass@1/pass@n、训练/评测seed及配对区间：
- 固定相同前缀与自由轨迹的无干预/即时受控/训练后分布分开记录：
- accuracy、Top-K集合overlap、共享Top-K质量、全概率相交系数、有符号/绝对熵差：
- 触发率/位置、向量版本/漂移、长度/重复/截断/有效答案与带hook诊断：
- T/R参考生成、S自身bootstrap、S0读取、当前学生每轮重读与搜索token/成本：
- 逐token教师查询、probe/重算、学习重放、更新、评测的token、墙钟/GPU-hours与峰值显存：
- 数据/信息、学习、实际总计算三类预算；便宜基线如何使用节余的同计算预算：
- 仅即时受控改善 / 关闭后收益 / 粒度收益 / 刷新增量及负结果（按证据填写）：
- 是否可收敛为便宜缓存/全局方案 / 失败与限制 / 后续范围与依据：
