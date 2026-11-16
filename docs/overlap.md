# overlap 与 P_overlap 应该怎样计算

当前实验采用：**在相同可见文本前缀上，比较两方下一原生 token 的前缀兼容性；overlap 数候选，P_overlap 数概率质量。** 两个指标使用同一套兼容规则。它们用于高效筛选 steering 方案，不表示完整单词或完整回答的生成分布已经相同。

下面用同一条回答、同一组候选概率，把分词、两项指标和 `late` 的后续分布连起来说明。例子中的分词和概率均为人为构造，不是 Qwen tokenizer 的实测结果。两项指标均只使用vLLM返回的原生Top16，不重新归一化。

## 1. 先确定在哪些位置比较

设共同问题前缀为 Q，冻结的学生源回答为 `calculate the sum`。假设两方分词为：

```text
S：  ["calculate",       " the", " sum", "<EOS>"]
T：  ["calcu", "late",   " the", " sum", "<EOS>"]
```

比较依据是已经看到的原始文本字节，不能按两方 token 序号直接配对：

| 已看到的回答正文 | S接下来预测 | T接下来预测 | 是否计算两项指标 |
|---|---|---|---|
| 空，即前缀Q | `calculate` | `calcu` | 是：两方都是Q处的下一步分布 |
| `calcu` | 没有这个原生token边界 | `late` | 不单独比较：学生没有相同文本状态 |
| `calculate` | ` the` | ` the` | 是：读取Q＋`calculate`处的新分布 |
| `calculate the` | ` sum` | ` sum` | 是：读取Q＋`calculate the`处的新分布 |

实际按S/T两方共同的UTF-8安全token边界划分最小span。本例第一个span是完整的 `calculate`，覆盖S的一个token、T的两个token。两项指标均比较S与T。

因此，**不是把S的 `calculate` 分别与T的 `calcu`、`late` 比两遍。** 两项指标在Q处算一次，下一次到Q＋`calculate`再算。内部的 `late` 条件分布如何处理，见§5。

## 2. 同一个前缀下，哪些候选可以匹配

对异构tokenizer，普通候选的规则是：**一个完整token的原始字节串，是另一个完整token字节串的前缀，即为兼容。**

| S候选 | T候选 | 是否兼容 | 理由 |
|---|---|---|---|
| `calculate` | `calculate` | 是 | 完整字节相同 |
| `calculate` | `calcu` | 是 | `calculate`以完整的 `calcu` 开头 |
| `calculate` | `continue` | 否 | 两个完整候选均不是对方的前缀 |
| `calculate` | `calendar` | 否 | 仅共享开头 `cal` 不够 |
| `calculate` | `late` | 否 | 中间或结尾包含不算前缀 |
| ` calculate` | `calcu` | 否 | 前导空格也是实际字节 |

不把所有词先拆到词表中的最短token，更不统一投影到首字节 `c`。如果某方真的给短token `c` 分配了概率，它可以凭自己的候选名额或概率容量参与匹配；这仍只说明短前缀兼容，不能说明完整词相同。

EOS只与约定的终止事件兼容；其他控制token按明确语义和拼写精确匹配，不能把所有控制token合并。普通空字节token不能作为万能前缀。保留空格、大小写和精确ByteLevel字节，不做词干化或同义词替换。

为保留同词表的原生计数口径，tokenizer映射完全相同时，两项指标都只允许相同原生ID匹配。本文讨论的S与T是异构映射，使用上述前缀规则。这是本项目的跨family适配，不称为Rethinking已经给出的跨词表定义。

## 3. Q处的完整候选表

下表所有概率都以**同一个Q**为条件。`—`表示该方没有这个token，质量为0；每方有19个token，概率总和为1。

| token字节文本 | S概率 | T概率 |
|---|---:|---:|
| `calculate` | 0.400 | — |
| `calcu` | — | 0.250 |
| `continue` | 0.200 | 0.350 |
| ` solve` | 0.040 | 0.040 |
| ` simplify` | 0.040 | 0.040 |
| ` compare` | 0.035 | 0.035 |
| ` compute` | 0.025 | 0.025 |
| ` check` | 0.025 | 0.025 |
| ` find` | 0.025 | 0.025 |
| ` add` | 0.025 | 0.025 |
| ` subtract` | 0.025 | 0.025 |
| ` multiply` | 0.025 | 0.025 |
| ` divide` | 0.025 | 0.025 |
| ` the` | 0.025 | 0.025 |
| ` sum` | 0.025 | 0.025 |
| ` result` | 0.025 | 0.025 |
| ` answer` | 0.025 | 0.025 |
| `late` | 0.003 | 0.003 |
| `<EOS>` | 0.004 | 0.004 |
| `<CTRL>` | 0.003 | 0.003 |
| 合计 | **1.000** | **1.000** |

这里 `<CTRL>` 指两方相同的一个具体控制事件，不代表所有控制token的汇总。16个高概率普通候选组成Top16；最小概率0.025高于尾部三项，因此Top16边界不存在并列选择问题。

注意：本表的 `late:0.003` 是 **p(late｜Q)**，表示直接在Q后输出 `late`。它与T先输出 `calcu` 后再输出 `late` 的概率不是同一项。

## 4. 两项指标直接从这张表怎样算

### overlap：Top16的最大一一兼容匹配数 / 16

取S、T各自原生Top16，用§2的规则连出兼容关系，寻找能组成的最多匹配对。**每个Top16候选最多使用一次**，不能让一个短token同时替多个长token贡献计数，也不能按出现顺序贪婪固定配对。

本例可以这样匹配：

| 匹配部分 | 对数 |
|---|---:|
| `calculate` ↔ `calcu` | 1 |
| `continue` ↔ `continue` | 1 |
| 从 ` solve` 到 ` answer` 的14个相同候选 | 14 |
| 合计 | **16** |

所以按当前前缀兼容定义：

$$
\mathrm{overlap}(Q)=\frac{16}{16}=1.
$$

它表示两方Top16可以全部找到互不重复的兼容对象，不要求对应概率相等。本文不再把严格完整token相等得到的15/16混作修正后的结果。

### P_overlap：Top16原始概率的最大兼容质量

仅对两方Top16候选使用同一兼容关系。每个token拥有vLLM返回的原始概率exp(logprob)，可以把质量分给兼容对象，但分出去的总量不能超过自己的概率，接收方也不能超出自己的概率。

令x、y分别属于两方Top16，m(x,y)为一对兼容token之间分配的质量：

$$
P_{\mathrm{overlap}}(Q)=\max_m\sum_{x,y}m(x,y),
$$

约束为：

$$
m(x,y)\ge0,\qquad
\sum_y m(x,y)\le p_S(x\mid Q),\qquad
\sum_x m(x,y)\le p_T(y\mid Q),
$$

不兼容时强制m(x,y)=0。这是带概率容量的最大匹配；不能对所有兼容对的min直接相加，也不能预先贪婪匹配同名项。

本例的有效匹配互不争抢容量，可以直接列出：

| 匹配部分 | 可匹配概率质量 |
|---|---:|
| `calculate` ↔ `calcu` | min(0.400, 0.250) = 0.250 |
| `continue` ↔ `continue` | min(0.200, 0.350) = 0.200 |
| 其余14个共同普通候选 | 0.390 |
| 合计 | **0.840** |

因此：

$$
\mathrm{overlap}(Q)=1,\qquad P_{\mathrm{overlap}}(Q)=0.84.
$$

S剩余0.15的 `calculate` 不能匹配T剩余0.15的 `continue`。这就是候选可以全部兼容、概率质量却没有全部兼容的原因。

本例两方Top16各保留0.99质量，Top16外的late/EOS/CTRL共0.01不参与。保留原始概率，不除以0.99。因此P_overlap=0.84，衡量的是截断后的匹配质量，不能当成完整词表的重合度。EOS或控制token只有进入Top16才参与匹配。

## 5. T预测late的表怎么办：有这一步，但不单独算overlap

T在同一源回答上做因果评分时，确实还会有 **Q＋`calcu`** 处的下一步分布。举例：

| T在Q＋`calcu`处的下一token | 条件概率 | 拼接结果 |
|---|---:|---|
| `late` | 0.80 | `calculate` |
| `lator` | 0.15 | `calculator` |
| `lation` | 0.05 | `calculation` |
| 合计 | **1.00** | |

此时学生没有Q＋`calcu`这个原生token边界。不能拿这张表与学生Q处或Q＋`calculate`处的表求交，因为已经看到的文本不同。

具体处理为：

1. **overlap/P_overlap不在Q＋`calcu`处另建比较点。** 本例学生只有一个 `calculate` token，因此没有漏掉一个可对齐的学生预测位置。
2. **保留T对源token `late` 的条件logp。** 它参与既有的共同span原生token平均logp等似然记录；教师这一步前向和源文本概率不丢弃。
3. T沿源文本消费完 `late` 后，两方都到Q＋`calculate`，读取那里的新分布，再计算下一组指标。Q处的1和0.84不能沿回答一直复用。

这也说明单步前缀兼容的限度：

```text
S选择单token calculate的概率：0.40
T先选calcu、再选late这条token路径的概率：0.25 × 0.80 = 0.20
Q处前缀匹配分给calculate↔calcu的质量：0.25
```

0.25只表示T下一步选择了与 `calculate` 相容的 `calcu`；它没有承诺后面一定接 `late`。如果第二张表的late概率变成0.02，T这条路径的概率就变成0.005，但Q处两个前缀兼容指标仍不变。

**因此，P_overlap=0.84不能解释为“84%的概率生成同一个完整词”。** 要比较完整词/文本，需要接入后续条件分布。仅把上例0.25换成0.20也不够：其他候选同样可能需要展开，还可能存在多条token路径；不同长度字符串之间也要重新定义互不重叠的比较事件。现有两张Q处分布无法给出这种完整文本重合度。

对于本项目固定前缀、固定强度的T−S／S−T消融，明确使用本文的单步前缀兼容指标，避免为所有候选额外展开后续分支。它解决“仅因首字节相同就把两个完整长token当成兼容”的问题，同时保留短token后续尚未确定这一局限。

## 6. 怎样落实到当前静态实验

- 无SV源rollout后，S、T对同一正文批量评分；在合法共同span起点取分布，不在线追问教师，不为候选额外生成分支。
- 每个span起点计算一次两项指标，并广播到该span覆盖的学生tokens。如果学生把 `calculate` 分成多个token，它们复用span起点值；内部学生位置不是各自完成了独立对齐的精确比较点。
- 源文本终止EOS目标、含控制符或不可对齐的span按既定规则记NaN及覆盖率。有效起点Top16中的EOS候选仍参与指标，未进入Top16的尾部不计。
- 原T−S含2个static_overlap方案和2个all方案，新增教师S−T使用相同的global/per_question × static_overlap/all组合，共8类候选。static_overlap按gate_metric和strict-<阈值冻结位置，all请求全部回答预测位置；两项指标都记录和画图。后续训练退火规则见[ablation计划](ablation.md#固定强度与线性退火)，零强度不实际注入。
- 用同一兼容规则计算overlap与P_overlap，并留档定义版本。新规则下重新校准阈值；既有静态mask、排名和缓存不能直接沿用。
- 直接请求EasySteer/vLLM的`logprobs=16`与`prompt_logprobs=16`。保存原生ID(int32)、原始logprob(FP32)，以及实际观察token的logprob。vLLM额外返回的观察token若不在Top16，只用于路径似然，不加入两项指标。

实现使用[prefix_matching.py](../src/steer_opd/prefix_matching.py)：增广路径求Top16计数最大匹配；最多32个候选节点组成小前缀森林，精确求概率容量匹配。相同tokenizer映射时，P_overlap退化为两方Top16 ID交集上的sum(min原始概率)。统一记录`metric_version=4`和`matching_rule_version=native_token_prefix_compatibility_v1`。

已删除完整词表概率采集、258类字节概率投影和自定义统计RPC。两项指标共用Top16缓存，源码、模型/分词/控制事件身份和概率文件hash仍校验。旧定义的阈值、mask、排名和GPU凭据不能直接复用；已提交教师文本保留，重新启动时单独核验可复用范围。

每个预测位置的Top16 ID与logprob共128字节，加实际观察token的logprob为132字节。约38万位置的这一部分约48 MiB；模型KV、文本与BF16激活另计。引擎仍整批连续调用，返回后保存，不增加候选续写或教师回调。

CPU测试覆盖`overlap=1、P_overlap=0.84`算例、未归一化的截断质量、贪婪配对反例、容量争抢、同词表退化、别名/EOS/控制/空token和独立枚举最小割。Top16协议的真实引擎验收与下述教师S−T实验已完成；尚未做专门吞吐对照，不据此声称性能提升。

## 7. 新增实测：1000题教师空间S−T消融（2026-11-16）

教师S−T四组已于2026-11-16北京时间19:03完成，Select和Confirm各四组，GPU子阶段均正常退出。此次只做固定前缀的无梯度干预与评分，参数更新数为0。执行计划仍以[ablation.md](ablation.md)为准，本节记录已完成结果。

### 7.1. 方向、数据与注入对象

记μ(A←B)为模型A读取问题与模型B回答后，对保留回答token隐状态求得的均值。原学生方向和新增教师方向分别为：

$$
d_i^{T-S,S}=\mu_i(S\leftarrow T)-\mu_i(S\leftarrow S),\qquad
d_i^{S-T,T}=\mu_i(T\leftarrow S)-\mu_i(T\leftarrow T).
$$

S−T由同一个冻结教师读取学生源回答与自身参考回答，在教师隐空间作差，并注入教师。原T−S由同一个冻结学生读取两种回答，在学生隐空间作差并注入学生。两种向量不能直接跨模型相减，S−T也不是对原学生向量简单取负。

| 项目 | 本次设置 |
|---|---|
| 模型 | T=Qwen3.5-35B-A3B，S=Qwen3-4B；不使用Ref |
| SV题池 | 原1000题，教师侧1000对文本均可用；每题读取学生／教师回答各一次，共2000份文本 |
| 划分 | 原Select700题／Confirm300题，划分不变；global方向使用全部1000题，包含Confirm |
| 提取 | 教师自身tokenizer与模板；prompt+回答右截断至2048 token，对保留回答token求均值，不按正确性、截断或重复过滤 |
| 层与强度 | 教师40层的0-based29层，Slerp α=0.1；原T−S为学生36层的0-based26层 |
| 方向粒度 | global先平均原始差分再单位化；per_question使用本题方向，零方向零干预 |
| 评分与留档 | 读取完整既有学生源推理，2048仅限制SV提取；seed42，Top16原始概率；仅保存池化均值／范数与bank，不保存逐token隐状态 |

本次复用原1000题教师参考回答、学生无SV源回答、学生capture、原生教师评分、对齐与源指标，没有重新生成师生回答或受SV控制的自由rollout。

### 7.2. 如何保持共同基线与mask

原T−S改变学生分布，新增S−T固定学生分布、改变教师分布。新增四组引用原T−S已提交的学生无SV重放概率，并复用同一份冻结教师评分；两个实验使用完全相同的无SV基线。该基线的complete SHA保存在每组common_baseline中，实际来源与产物SHA见[来源索引][st-sources]。

历史mask只允许移除direction_version／direction_versions这类方向元数据；原始prefix、概率SHA、指标、阈值和gate必须一致。原700／300两部分的1000份mask均已核验，不能因更换注入模型而重新选择有利位置。

当前gate_metric与rank_metric均为overlap。Select的目标选点率为20%，按离散分值可实现的最接近比例得到19.3292%（以有效学生位置为分母）；冻结阈值为τ=0.5625000000000001，采用FP64严格小于比较。不能将τ四舍五入成0.5625后重新判定位置。Confirm沿用同一阈值。

选点先在原学生坐标的共同span上冻结，再投影到教师原生token预测位置，实际注入索引为教师自己的prompt_len−1+t。两项指标仍广播到学生坐标；教师原生注入率和映射后的学生覆盖率分别记录。

### 7.3. 与原1000题T−S四组比较

下表均为相对上述共同无SV基线的**题目等权增量，单位pp（百分点）**：先求每题有效位置均值，再对题目平均。+1 pp表示0到1指标增加0.01，与相对增长1%不同。两种指标均按前文metric_version=4定义计算。

| ID | 方向／注入模型 | 位置／粒度 | Select Δoverlap | Select ΔP_overlap | Confirm Δoverlap | Confirm ΔP_overlap |
|---|---|---|---:|---:|---:|---:|
| 1 | T−S／学生 | all／global | -0.6302 | -0.3625 | -0.6394 | -0.3411 |
| 2 | T−S／学生 | all／per_question | -0.8491 | -0.3347 | -0.8962 | -0.3492 |
| 3 | T−S／学生 | static_overlap／global | +0.2428 | +0.0541 | +0.2405 | +0.0582 |
| 4 | T−S／学生 | static_overlap／per_question | +0.1633 | +0.0212 | +0.1506 | +0.0223 |
| 5 | S−T／教师 | all／global | +0.7368 | +0.6101 | +0.7077 | +0.6091 |
| 6 | S−T／教师 | all／per_question | +0.4709 | +0.4498 | +0.4362 | +0.4535 |
| 7 | S−T／教师 | static_overlap／global | +0.3138 | +0.1710 | +0.2989 | +0.1734 |
| 8 | S−T／教师 | static_overlap／per_question | +0.2640 | +0.1345 | +0.2467 | +0.1407 |

数值来自[原T−S Select][ts-select]、[原T−S Confirm][ts-confirm]和[新增S−T Select][st-select]、[新增S−T Confirm][st-confirm]。Select与Confirm均保留四个预先指定方案，Confirm不反向选参。

新增S−T中，方案5（all＋global）在Select两项指标均最大，Confirm保持同样排序。下面同时列出其绝对值，所有数值为百分数：

| 划分 | 无SV overlap | 方案5 overlap | 无SV P_overlap | 方案5 P_overlap |
|---|---:|---:|---:|---:|
| Select | 69.3079% | 70.0447% | 90.4429% | 91.0531% |
| Confirm | 69.3615% | 70.0692% | 90.4260% | 91.0351% |

例如Confirm的overlap从69.3615%到70.0692%，增加0.7077 pp；P_overlap从90.4260%到91.0351%，增加0.6091 pp。方案5的题目配对bootstrap 95%区间分别为[0.6824, 0.7362] pp与[0.5854, 0.6349] pp。这是当前题池、固定方向与一次运行条件下的题目重采样区间，不覆盖多随机种子或独立测试集的不确定性。

**token加权结果单列如下**，每个有效学生预测位置一票，长回答权重更大；不能与上面的题目等权均值混用：

| S−T方案 | Confirm Δoverlap（token加权，pp） | Confirm ΔP_overlap（token加权，pp） |
|---|---:|---:|
| 5 | +0.7053 | +0.5425 |
| 6 | +0.4556 | +0.3878 |
| 7 | +0.2770 | +0.1401 |
| 8 | +0.2345 | +0.1094 |

### 7.4. 注入覆盖与教师重放漂移

| 静态S−T方案7／8 | 教师原生实际注入率 | 映射到学生坐标的覆盖率 |
|---|---:|---:|
| Select | 19.3835% | 19.3234% |
| Confirm | 19.1453% | 19.0632% |

all组教师原生注入率为100%，静态组约19%；两组没有匹配注入预算。映射比例的分母为全部学生预测位置，校准比例的分母为有效学生位置，教师原生比例又使用教师token数，三者不要求相同。Select与Confirm分别有2,243,391和937,047个有效学生预测位置。

Qwen3.5教师未启用batch-invariant，因此另做无SV重放，记录相对冻结共同基线的漂移。以下为题目等权pp，**没有自动从候选增益中扣除**：

| 划分 | 无SV重放 Δoverlap | 无SV重放 ΔP_overlap |
|---|---:|---:|
| Select | +0.001222 | +0.000158 |
| Confirm | -0.002306 | +0.000066 |

本次平均漂移幅度远小于方案5的增量，原始记录见[Select漂移][st-drift-select]与[Confirm漂移][st-drift-confirm]；这不构成多次独立运行的稳定性证明。

### 7.5. 当前结论与解释边界

- 本固定前缀设置下，教师S−T四组的两项指标在Select／Confirm均为正；all＋global的增量最大。原学生T−S的all两组为负，static_overlap两组为正。
- 教师四组中all增量更大，同时使用了更多注入位置；未做同预算对照，不能据此判断静态选点没有作用。该对照同时改变读取者、隐空间、层位与注入对象，不能把差异单独归因于S−T符号，也不保证两模型的Slerp0.1对应相同扰动大小。
- S−T提高的是**教师分布对固定学生分布的相容性**。学生没有更新，尚未证明SV撤去后的学生收益、正式OPD学习收益或答案准确率提升。全部1000题参与方向构造，Confirm属于训练侧复核；AIME／MATH-500独立测试尚未开展。

运行参数名为`n1000_slerp0.1_S26_T29_seed42_st_only_v1`。原始评分与图分别位于该参数的step3/select/teacher和step3/confirm/teacher；各方案comparison目录分别保存overlap／P_overlap的PNG、SVG、CSV与PDF。汇总见[分析Markdown][st-analysis]和[CSV][st-csv]，方向提取质量见[教师文本读取统计][st-quality]。本节仅补充已完成结果，不启动新实验。

[ts-select]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_layer26_seed42_v1/select/summaries.json
[ts-confirm]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_layer26_seed42_v1/confirm/summaries.json
[st-select]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/select/teacher/summaries.json
[st-confirm]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/confirm/teacher/summaries.json
[st-drift-select]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/select/teacher/teacher_repeat_drift.json
[st-drift-confirm]: ../outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/confirm/teacher/teacher_repeat_drift.json
[st-sources]: ../outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/metadata/source_index.json
[st-analysis]: ../outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/analysis.md
[st-csv]: ../outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/comparison.csv
[st-quality]: ../outputs/step_2_sv/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/reader_teacher/activations/text_quality.json
