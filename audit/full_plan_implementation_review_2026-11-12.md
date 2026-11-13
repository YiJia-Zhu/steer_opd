> 历史缺口审计：以下对应本轮实现前的源码hash。后续用户已授权实现可选分支，最新状态见[方法分支](../docs/05_method_branches.md)与[实现审计](method_branches_2026-11-12/README.md)；GPU仍未运行。

# 完整方案与实际代码对照（2026-11-12）

结论：完整方案第16节的2–2–2尚未实现。当前可执行首轮接口仍是T−S＋全回答预测位置注入＋S0全局固定方向。
上一轮只增加同题100数据、rollout原生Top16直交教师、训练轨迹overlap和对应恢复/CPU检查。
本次只阅读源码、核对记录并保存此审计，不修改训练逻辑、不启动GPU、不下载资产。

## 三条主轴

| 项目 | 完整方案要求 | 现有实际代码 | 状态/证据 |
|---|---|---|---|
| 差分对象 | 同一个学生读T/R文本，构造T−R；T−S/R−S可作对照 | calibration只生成teacher/student两角色，提取器读两份后求T−S | 未接入R文本与三方向选择。run_figure8_campaign.py:185；figure8_stage.py:175/227 |
| 位置门控 | 采样前由教师分布与当前步无注入probe计算overlap，低于阈值才注入 | 先生成完整回答，再教师评分；vLLM decode对所有真实回答预测状态加同一个delta | 无教师在线交互、probe或阈值分支。run_figure8_campaign.py:213；vllm_steering.py:26 |
| 提取粒度 | 每题一个方向，同批同题rollout共享 | 每题pooled表示只作为中间文件，最后对原始差分取全局均值 | 保存pooled文件不等于按题注入。figure8_stage.py:216/227；resumable_core.py:92 |
| 提取刷新 | 每轮用本轮起始学生S_theta_k重新读取T/R缓存文本 | 固定加载cfg.student即S0；已有vectors.pt则直接返回；calibration在训练循环外 | 无当前checkpoint提取/刷新。figure8_stage.py:160/166；run_figure8_campaign.py:249 |
| 生成/学习一致性 | 同题本轮向量＋真实记录的逐token gate mask，含backward重算 | 同一全局向量，学习mask从首回答预测边界起全True | 固定全位置路径已有基础；动态mask/每题向量绑定与重放未实现。resumable_core.py:202 |
| probe缓存语义 | 探测和正式前向从同一旧KV开始，探测上层KV不污染正式分支 | 没有probe/正式分支KV快照或回退逻辑 | 新门控GPU验收缺口，不能由已有全位置缓存测试代替 |

配置figure8_shared_top16_v2.json中的direction=T−S、injection=all_response_prediction_states_rollout_and_replay。
提取器没有按direction字段选择角色，当前差分逻辑硬编码T/S；只改JSON为T−R不会实现对应算法。
旧configs/rethinking_opd.json虽声明T−R/overlap_only，但同时标记controlled_backend_implemented=false；
旧fork入口execution_blockers拒绝steering启动。声明字段和方法文档不作为实现证据。

## 已有基础与验证边界

- 同题数据：原OPD100同时作direction候选池，原独立dev128/AIME30不变；prepare_figure8_first_run.py:25。
- 同一个学生空间读取、回答正文mask、先池化再作差、单层/典型范数缩放已有固定方向代码。
- rollout逐位置保存raw Top16 ID/log-prob，候选额外采样项处理、原子落盘和hash绑定已有实现。
- 教师同一前向同时计算自己的Top16与学生候选分数；阶段结束输出训练轨迹overlap和共享质量。
- 学习沿用原student_p冻结奖励/三维PPO surrogate和token-mean，候选与rollout概率停止梯度。
- 去steer的AIME评测与固定Base dev前缀overlap已有入口；两组共用第0步Base评测。
- 模型/Adam/RNG原子断点、fresh idle GPU入口、暂停保护已有；新run仍paused_by_user、训练0步。
- 上一轮48项CPU检查覆盖上述固定方向Top16接口；其验证记录没有包含2–2–2，新接口GPU验收尚未运行。

## 训练轨迹overlap的限制

当前steer组保存的是已经加SV的正式策略Top16。教师随后在这些学生前缀上评分，可计算事后训练诊断；
新门控需要当前步不注入、但保留过去干预历史的probe分布，并在下一token采样前作决定。
因此无法由完成回答后的评分结果倒推当时门控；无hook整段重放也不能替代有受控历史KV的当前步probe。
Top16损失支持集来自实际受控分布；门控probe的候选集合需要另外区分。

当前教师整批后评分的单GPU流水线，需要为门控改造生成/评分的交互调度。探测、触发后重算、
采样前教师评分都是待计量的新成本，不能将刚省去的独立学生评分当成已覆盖这些开销。

## 其他研究面板

- 同题全局T−S是现有机制先导，尚不是T−R同题全局vs逐题缓存的完整公平比较。
- 独立熵差、等注入率/位置匹配随机门控、固定后段、错误题向量、在线/缓存/全局刷新均未接入当前训练campaign。
- 历史probe_cross_family_steering.py有随机/负向固定前缀诊断，extract_pilot_vectors.py有旧固定方向工具；
  它们没有构成新计划的在线门控训练对照。
- 共同SFT轨道、CompassOPD、多seed与MATH-500/多配对完整评测未接入本轮；SFT仍按用户早先要求暂缓。
- 跨tokenizer的有限候选/首字节历史诊断存在，不是第12节完整在线字节门控＋span OPD训练。
- 新动态向量的question_id/vector_id/checkpoint/hash、逐token mask与probe指标的持久化/恢复协议未接入。

## 实现顺序建议

1. 将提取器改为T/R/S角色明确、全局/逐题可选；先完成同题T−R全局与S0逐题缓存的方向格式和CPU语义。
2. 支持当前学生checkpoint逐题重提，按本轮checkpoint与参考文本hash版本化；同题多rollout绑定相同vector_id。
3. 支持每请求向量及指定历史gate mask的生成/学习重放，先验收固定已知mask和backward checkpoint一致性。
4. 接入采样前教师/probe交互与KV隔离，阈值从训练/dev侧校准；持久化mask、probe候选与实际受控候选。
5. 在用户明确恢复后完成GPU验收及最小2–2–2/普通OPD/全位置/随机控制，记录额外成本。

以上是代码缺口评审与建议，没有据此恢复实验或自动实施全部研究面板。
