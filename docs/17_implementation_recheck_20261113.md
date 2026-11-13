# 实现与Markdown一致性复核 — 2026-11-13

2026-11-14按用户再次要求重新核对，源码状态与本次审计一致；完整字节定义已获同意但尚未实现。
最新[复核证据](../audit/implementation_recheck_20261114/README.md)记录当前8组/4bank、执行锁、无人工seq上限、实际后端与未完成项；不启动实验。

结论：八方案结构与最新选择一致，**尚未完整复现用户Markdown，也没有全程使用vLLM/EasySteer**。
本次只复核和修正执行设置，不运行GPU。实验已列入[TODO](../TODO.md)，必须等用户明确同意，且先修复前置缺口。

## 1. 当前范围与实际后端

T=Qwen3.5-35B-A3B，Ref=Qwen3.5-0.8B，S=Qwen3-4B；Slerp0.1、75%单层26；八方案加共同基线、0更新。
这覆盖此前40/192配置提案的当前执行范围，不改变完整研究稿的科学定义。

| 阶段 | 实际实现 | 核对结果 |
|---|---|---|
| T/Ref/S参考回答、普通S源回答 | figure8_generate.py，普通vLLM0.32及项目worker | 已接入；新35B未GPU验收 |
| S0读取三份文本、提取第26层激活 | prepare_compass_banks.py、HF文本decoder | 已接入；中间层逐token激活不能由普通embedding接口直接替代 |
| T/Ref同文本原生概率评分 | score_compass_sources.py、HF批量因果前向 | 已接入，仍非vLLM评分 |
| 八方案/共同无hook重放 | HF参考实现＋EasySteer可选实现 | 原默认HF；本次待执行配置改为显式EasySteer，无自动降级 |
| EasySteer运行环境 | 固定fork的API/selector CPU契约、项目Slerp插件 | 普通vLLM不具备该API；本地GPU环境与数值路径未验收 |
| 统计/排名/冻结mask/绘图 | CPU | 不需要推理引擎 |

“已接入EasySteer适配器”与“本地已运行EasySteer”分开记录；没有当前新配对的GPU结果或提速倍数。

## 2. 序列调度核对与修正

当前生成LLM与静态EasySteer LLM构造都没有max_num_seqs/max_num_active_seqs、手工CUDA Graph尺寸或max_num_batched_tokens参数。
生成一次提交全部pending请求，EasySteer固定前缀评分一次提交该候选全部rows；没有按HF batch_size拆分vLLM队列。
max_model_len是上下文长度约束，SamplingParams.n=1是每题采样数，不是引擎并发序列上限。
resolved_max_num_seqs只读取vLLM自动决定的值，便于成本审计，不向引擎回传。

新增execution_policy.require_default_scheduler_options拒绝新配置中的手工序列/图尺寸上限；HF评分batch_size保留其原本含义。
历史generation_runtime中的旧参数名仅用于剥离运行限制、验证旧断点，不影响新运行。
普通vLLM当前关闭chunked prefill和prefix caching；这不是人工seq上限，但仍是未优化项目，不能称“全部已使用默认配置”或“已经最快”。
静态EasySteer暂时关闭prefix caching以免跨向量复用KV；在版本/请求身份隔离验收前不直接开启。

## 3. 逐项对照Markdown

| 约定/正文 | 实际代码 | 判断 |
|---|---|---|
| 最新选择：T−S/T−Ref × all/static_overlap × global/per_question | compass_campaign.make_base_job构造8组；compass_banks构造4份bank | 结构一致，尚未实测 |
| 完整稿§6/§8：同一个S0读取共同T/Ref/S文本；全局先均值后归一化，逐题不回退 | prepare_compass_banks、make_common_banks、direction_bank | CPU路径一致；新有效交集未生成 |
| 完整稿§7/§9：源生成完成后冻结mask，候选不重判；0更新同前缀前向 | static_masks、static_screen、run_static_overlap_screen | CPU路径一致 |
| 完整稿§9.6：prompt末预测第一token，后续t−1；零向量不干预 | selector_for_predictions、build_spec、Slerp及相关测试 | CPU通过；GPU待验收 |
| 完整稿§12.1：实际文本动作的共同span似然 | align_response、span_scores；保留实际S采样token，T/Ref各自编码 | 参考论文A.4独立实现，官方源码对照未完成 |
| **完整稿§12.2路径B：全支持下一字节概率，sum(min)** | **aligned_metrics仅比较Top16 token完整续写字节，intersection/16并广播span** | **不一致，正式实验阻塞项** |
| 完整稿§9.7：实际mask、逐token激活、概率、原清单前3条图和扰动 | HF路径有扰动，EasySteer明确relative_perturbation_available=false | 其余CPU路径已接；EasySteer扰动缺失 |
| 完整稿§9.7：提取checkpoint/hash | 缓存绑定模型路径、tokenizer、文本/概率/激活产物hash | 模型权重内容指纹尚未完整纳入，同路径换权重需防止复用旧缓存 |
| 无SV源概率vs无hook重放一致性 | 保存source_rollout_vs_replay_delta，基线与候选共用所选后端 | 仅记录差值，不等于已有GPU验收/拒绝阈值；运行前须验收 |
| 完整稿§10/§11：学习、撤去SV、多seed与机制控制等 | 旧完整队列有部分实现，当前新配对静态入口仅零更新8组 | 不能称完整复现；本轮不启动 |

两种跨词表指标无法通过改标签等价：Top16文件缺其余支持的概率质量，完整token字节事件与firstByte分类也不同。
修复时应从各角色原生完整分布聚合到256字节及明确EOS/控制类别，核验质量守恒、共享前缀/可解码边界、span覆盖规则和sum(min)。
若继续保留当前Top16指标，应只称独立诊断变体，单独输出与缓存，不能冒充MD路径B或Compass官方指标。
本次保留完整稿§12.2原定义，没有为迁就代码改写科学要求。

## 4. 当前锁与留档

[执行配置](../configs/compass_static_eight_20261113.json)含execution_hold=awaiting_user_approval及已知execution_blockers。
锁传播到参考/源生成、bank和原生评分/八组评分job；入口在模型加载、GPU申请或执行写入之前检查。
CPU准备仍允许，无需解除锁；用户同意不是自动忽略指标与环境缺口，缺口修复后才能执行正式面板。

新目录runs/compass_static_eight_review_20261113用于本次待执行配置。
旧runs/compass_static_eight_20261113仍保留原HF配置与100题准备记录，不重写冻结job，也不把准备称实验结果。
本次没有下载完整权重、启动GPU、生成新回答/评分或进行参数更新。

CPU验证、构造器静态审计与修改前指纹见[audit/implementation_recheck_20261113](../audit/implementation_recheck_20261113/README.md)。
相关CPU回归65 passed；项目tests/全套200 passed、1 skipped（13.65秒）。CPU通过不解决上述科学定义差异，也不替代GPU验收。
