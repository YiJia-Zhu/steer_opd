# 逐token注入对照的留档与复用

2026-11-13补充：用户已授权完整队列，见[执行协议](08_full_scheme_execution.md)。新轨迹统计schema2除教师自身Top16，
还持久保存教师对学生Top16的原生log-prob和实际采样token的教师分数（若可用），避免清理训练scores后丢失这些概率。
Compass额外的R/S0候选及采样token分数另存`auxiliary_token_statistics/`，不随临时scores清理。
S0公共三文本池保留R的逐token激活，复用已有T/S激活；当前学生每轮大激活仍按预算另行开启。

更新：2026-11-12。用户要求提前记录以免后续因缺日志重跑。代码与92项CPU验证完成，
未启动GPU、未生成正式模型的新激活/SV/overlap结果；第一步仍T−S、全位置、S0全局均值、0次OPD。
提取与未来OPD固定同100题，本轮只调层位置/层数/强度，不安排旧/新题集对照。

## 1. 完整方案中的“逐token”

第7.3–7.5节比较的是注入开关g_t：全位置、overlap单阈值、因果后段、独立有符号熵差，
并要求注入率/绝对位置分布匹配的随机对照。同一轨迹/对应学习使用固定方向，记录mask供重放。
只干预回答预测状态，包括最后一个prompt状态对首回答token的预测；不对全部题干位置加向量。
这与“每个token提取一个SV”不同。随机概率分支已有；自动在线位置匹配控制器仍待实现，
不能把普通随机分支当作已完成严格匹配对照。

## 2. 提取缓存

零更新诊断的提取job默认`cache_token_activations=true`、`token_activation_dtype=bfloat16`；
不修改原v2配置字节或首轮方法。普通训练的bank分支默认不开启每轮完整激活转储，可显式配置；历史legacy提取路径保持原样。
只对实际经过读取的参考保留激活；空白/截断/严重重复等早期排除仍记录原因，不为了转储而额外读取。

每份方向bank目录包含：

| 路径 | 内容 |
|---|---|
| `token_activations/<question_id>/<role>.pt` | 每候选层的token激活、模型/文本身份、分词/offset/正文mask及位置映射 |
| `pooled/<question_id>.pt` | 两份正文均值、逐内容token范数、可用/排除原因、原始缓存路径与SHA256 |
| `vectors.pt` | 每层全局原始均值差分、每题原始差分、冻结尺度与bank版本 |
| `metadata.json` | 可用题ID、完整缓存索引、dtype、checkpoint/步数/hash与方向文件hash |

每层`states`为`[response_length+1, hidden_size]`，第0行是最后一个prompt状态。
`response_token_state_rows=1..L`对应实际回答正文状态，用于Phi池化；
`prediction_state_rows=0..L-1`对应预测相应回答token的t−1状态。
`state_input_positions`给出模型输入绝对位置，`student_response_ids`与`body_mask`长度为L，
`response_offsets`保留学生tokenizer对原回答文本的offset_mapping，避免过滤后丢失位置对应。

只有选定层会被缓存，未缓存的新层需要读取模型。T/S回答长度不同，不能按同序号直接相减。
BF16默认保留BF16学生原始层输出；源/缓存dtype均记录。需要FP32可在实际提取前指定float32。
中间文件按角色原子提交、pooled记录后提交；完整bank后缺失小元数据可CPU修复，无需模型重读。
缺失/损坏的已提交激活明确报错，不悄悄当作完整缓存。

CPU复用入口是`steer_opd.extraction.load_token_activation_cache`，验证SHA256和位置契约后以mmap读取。
例如取`states[layer][response_token_state_rows][body_mask]`即可重新池化；这些是指定checkpoint、
关闭hook、读取该份参考文本时的激活，不能代替参数更新后或换注入历史的激活。

## 3. 固定前缀逐位置统计

`<run>/teacher_token_statistics/<qid>.pt`缓存冻结教师的Top16和完整151665输出子集上的熵。
旧Top16缓存缺熵时只补一次同checkpoint前向，核对原候选分数（最大误差<=5e-4），保留旧候选/概率作为目标。
此后select/confirm各候选共用，无需为每个超参重复教师前向；不会从条件Top16伪造全支持熵。

`<run>/select|confirm/<condition>/<qid>.pt`保存响应token ID、响应序号、绝对预测位置、bank/条件/row hash、
实际mask、学生Top16原生概率、逐位置overlap与共享质量、两端完整支持熵及有符号熵差、位置/长度桶。
教师文件路径/hash明确绑定；无hook基线和带hook条件各存一份。rho=0时g_t仍可为True，但条件记录的扰动幅度为0。
记录标记`current_step_probe=false`：这是固定Base token历史下当前条件的完整重放，不能冒充受控历史上的当前步probe。

## 4. 后续训练轨迹

- `rollouts/<key>.json`显式保存逐token`steering_mask`、token序列、首预测绝对位置、方向绑定/生成job hash。
- `rollouts/topk/<key>.pt`保存原生Top16、采样logp及完整支持熵；HF门控还保存教师分数、probe候选/熵。
- `<step>/token_statistics/<key>.pt`在教师评分时保存teacher Top16、逐位置交集/共享质量、熵差与mask、
  向量/模型/前缀版本和原始学生Top16文件指针。在线probe_overlap/熵差与正式受控分布统计分别存放。
  此目录不随`student_scores/scores/full_logits`临时缓存清理而删除。

vLLM在raw logits处只计算每位置熵标量，不转储整词表logits；真实GPU验收会检查熵与捕获logits一致。
该新增GPU接口尚未运行。HF参考路径的熵直接来自同一完整分布，逐token与RNG/轨迹一起保存。
没有完整logits时不能离线重算任意全词表函数，因此只承诺已经保存的统计和Top16事件分析。

## 5. 能避免哪些重跑

可避免：重新生成同一参考文本、为同checkpoint/同文本换池化重读模型、重新计算已保存条件的逐位置统计、
重复教师评分，以及仅为得到位置分布/历史mask/向量版本而重跑旧实验。

仍需运行：新门控/位置条件的学生前向或生成、实际训练对照、新checkpoint的提取，以及未缓存层。
在固定前缀上离线匹配注入次数/位置可作为诊断；精确匹配已完成轨迹不能被说成因果在线方法。
当前结果仍停在GPU未运行状态，不把保存接口或小模型检查当作方法效果。
