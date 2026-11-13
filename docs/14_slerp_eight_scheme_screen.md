# 第一轮：固定Slerp 0.1与75%层，完整比较8方案

最新[复核](17_implementation_recheck_20261113.md)已发现当前项目Top16字节续写指标与完整稿§12.2 firstByte全概率指标不同，正式面板须先修复。
实验已记入[TODO](../TODO.md)，等待用户明确同意；待执行配置默认EasySteer，但本地GPU引擎仍未验收。

当前可执行编排见[Compass式跨tokenizer接口](16_compass_alignment_eight_way.md)和[新配置](../configs/compass_static_eight_20261113.json)。
8项选择清单保留为规格；新配置用明确命名的项目字节续写overlap，官方仓库源码对照尚待URL，GPU评分未开始。

2026-11-13用户选定：先固定Slerp强度0.1、75%单层，全部8方案都比较，再选择方案。
本轮只搜索方案，不自动扩展到此前192项全网格或40项搜索建议。
[参数清单](../experiments/next_pair_slerp_scheme_screen_20261113.json)是已选实验规格，尚不是可执行campaign配置。

## 1. 当前确定的配置

| 项目 | 设置 |
|---|---|
| 教师T | Qwen3.5-35B-A3B |
| 参考Ref（R） | Qwen3.5-0.8B |
| 学生S | Qwen3-4B |
| 算子与强度 | Slerp，alpha=0.1 |
| 注入层 | 75%单层；学生36层，对应0-based26，即第27个block输出 |
| 提取者 | 同一个冻结的初始学生Qwen3-4B读取各角色文本 |
| 向量刷新 | 本轮固定，不随token重新提取 |
| 比较阶段 | 同源回答固定前缀，0次参数更新 |
| 共同基线 | 整段无SV/no-hook，额外1组 |

Slerp的alpha表示沿夹角旋转的比例；0.1不是按旧加法rho做数值等价替换。

## 2. 八方案全部比较

| 编号 | 方向 | 位置策略 | 向量粒度 |
|---|---|---|---|
| 1 | T−S | 全回答预测位置 | 全局均值 |
| 2 | T−S | 全回答预测位置 | 逐题 |
| 3 | T−S | 静态overlap位置 | 全局均值 |
| 4 | T−S | 静态overlap位置 | 逐题 |
| 5 | T−Ref | 全回答预测位置 | 全局均值 |
| 6 | T−Ref | 全回答预测位置 | 逐题 |
| 7 | T−Ref | 静态overlap位置 | 全局均值 |
| 8 | T−Ref | 静态overlap位置 | 逐题 |

共8个非零配置及1个无干预基线。本轮不先淘汰方向、粒度或门控中的任何一轴。
所有方案共用题目、T/R/S参考文本、有效交集、无SV源回答前缀和选型数据规则；全局先平均原始差分再归一化。
逐题只使用本题向量。旧1.7B学生的向量、尺度、源概率和旧55题有效数不迁移为新学生结果。

## 3. 静态overlap的规则与真实代码状态

方案上已明确改为静态：普通无SV学生源rollout完成后，由教师在相同可见文本前缀批量概率评分，
计算O_src并冻结`g_static[t]=1[O_src[t]<tau]`，随后所有静态方案复用这份位置表。
候选注入后的新overlap仅作结果，不据此重判idx，不做逐token在线教师回调。
本轮固定前缀比较不会改变已经生成的源回答；后续二次生成或learning-only训练阶段另按明确目标定义。

用户后续已授权修改代码：[静态mask/Slerp及EasySteer适配](15_static_mask_slerp_implementation.md)已实现并通过CPU验收。
新入口run_static_overlap_screen.py使用冻结mask及批量前向；旧gate=overlap保留动态语义，旧入口拒绝新静态/Slerp标签。
此8项规格仍需源数据、bank和已验收的跨tokenizer指标，不能直接传给旧campaign。

新配对还须验收跨tokenizer概率接口：本地Qwen3.5-0.8B与Qwen3-4B的ID映射不同，
35B教师官方tokenizer元数据已核验，T/Ref映射相同而S不同；完整权重与GPU路径尚缺验收。不能直接相交T/S整数token ID，不能将学生token ID直接喂教师，
不能无条件沿用旧151665输出支持或旧阈值。须先固定同可见文本前缀上的可比事件指标，
再在新训练侧源数据上校准并冻结阈值；本轮不额外搜索阈值。

## 4. 排名、留档与当前停点

按验收后同前缀指标的题目等权配对增量排名，另报token加权、区间、静态触发率与触发位置增量。
在候选评分前冻结训练侧select/confirm，confirm用于检查冻结结果，AIME不选型。
默认每方案画源清单前3条的整段无SV基线、候选曲线、实际mask散点和差值，保存原始数据与hash。
先报告8方案与共同基线，再决定后续选层、选强度或OPD；此选择不自动启动后续训练阶段。

当前已经固定本轮8项规格并同步文档，尚无新8组模型评分或有效性结果。
静态mask/Slerp已完成CPU验收；EasySteer GPU引擎及新配对概率接口仍需验收，已查路径尚未发现35B教师权重。
旧[40配置建议](13_next_pair_greedy_search_proposal.md)和[192候选池](10_overlap_parameter_search.md)保留为后续参考，
不作为本轮自动执行队列。核对记录见[audit/slerp_eight_scheme_selection_20261113](../audit/slerp_eight_scheme_selection_20261113/)。
