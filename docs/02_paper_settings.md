# 论文设置、模型与数据证据

## 最新执行限制与复核（2026-11-13）

本轮固定T=Qwen3.5-35B-A3B、Ref=Qwen3.5-0.8B、S=Qwen3-4B，Slerp0.1/75%层26，8方案＋共同基线，0更新。
实验已列入[TODO](../TODO.md)，必须等用户明确同意后才执行；现在仅代码/文档/CPU检查，不运行GPU。
[最新复核](17_implementation_recheck_20261113.md)发现跨tokenizer门控与完整稿§12.2不一致；EasySteer GPU等仍未验收，不能称完整复现。
下文192网格、旧配对、旧授权/实现状态保留为历史；不得据其覆盖本轮等待同意的执行锁。


## 2026-11-13最新：静态overlap取代动态逐步门控

当前使用普通无SV源rollout→同前缀教师批量概率评分→一次性冻结idx/mask，注入后不重新判定。
当前零更新搜索在同一源回答前缀重放比较，保留8方案、两种算子、4强度及3单层，共192非零候选和共同基线。
规则与应用阶段边界见[静态协议](12_static_overlap_protocol.md)，网格见[下一步实验](10_overlap_parameter_search.md)。
历史动态代码/结果保留，不再要求采样前probe、在线教师回调或动态引擎改造作为当前主线前置条件。
本次只同步文档：静态适配、Slerp及EasySteer迁移仍待实现/验收；未改生产代码或旧配置，未启动GPU。
后续同回答学习前向注入与静态位置二次生成分别列明，未自动选择训练阶段或新增rollout。

本次只修正项目当前流程，不改写文献的真实方法、分数或历史阅读深度；静态位置策略也不据此获得首次提出的主张。

2026-11-13补充核验：已读取Compass正文式(9)–(11)及EOPD v3正文/附录A，而非仅摘要。
本轮分别实现sampled优势/锚定、有限候选目标和教师高熵Top16 FKL；配对、初始化、数据与预算仍为本项目适配。
正文/输入SHA与CPU证据见`audit/full_scheme_2026-11-12/`；[运行边界](08_full_scheme_execution.md)不等于论文完整复现。

更新：2026-11-12。本文记录定点阅读所得的设置，**没有独立复现论文成绩**。
研究设计以[用户完整稿](Steering_OPD_完整研究方案.md)及[主计划摘要](01_research_plan.md)为准。
本轮只同步文档与定点核对已有本地正文；模型缺口不代表下载授权。

## 0. 最新方案与来源边界

最新用户已暂停训练，要求先在原始Base上做层位置/注入层数/强度的训练前overlap调参，
OPD更新数为0，结果供用户检查后才决定是否进入学习。详见[当前诊断协议](04_pre_opd_overlap_tuning.md)。
旧第10步overlap是OPD后无hook结果，不是原始Base带hook的即时结果，来源见
[第10步证据](../audit/step10_overlap_2026-11-12/README.md)。当前固定提取/未来OPD同题100题，直接进行注入参数搜索。
当前不新增文献或新颖性结论，不把即时overlap提高解释为OPD有效的充分条件。
以下2–2–2为后续研究设计；可选[代码分支](05_method_branches.md)已实现CPU语义，GPU未验收，第一步仍T−S/全位置/S0全局。

按用户修订的[完整稿](Steering_OPD_完整研究方案.md)同步为2–2–2：T−R方向、
overlap单阈值门控、每轮rollout前当前学生读取当前OPD题的T/R文本，逐题提取。
同题本轮生成与学习重放使用同一向量，更新后再遇题时重提；参考文本可固定缓存。
逐题粒度与刷新频率分开比较，熵差独立对照，不与overlap组合。
同家族低相容机制验证先于跨家族，主评测无steer/无门控。旧仅loss前向方案是单独变体。
用户随后要求统一基础配方：新方法面板采用Rethinking原fork Student Top-K16反向KL
系列surrogate，student_p/token_reward_direct/PPO及token-mean均对齐源码；全词表KL
另作消融。论文子集重归一化KL与源码冻结原生log-ratio奖励不直接宣称梯度等价。
100题、AIME单项、方向及受控门控是本项目差异，不能把新面板称为原Figure 8完全复现。
默认值与来源逐项见[配置对照](../audit/rethinking_alignment_2026-11-12/comparison.json)。

旧[首轮配置](../configs/figure8_first_run.json)100题OPD/256题独立方向/128题dev的
两组各10步现已暂停并保留。最新用户要求的[同题Top16 v2](../configs/figure8_shared_top16_v2.json)
让SV/OPD共用原100题，保留dev/test划分；方向仍为S0全局固定T−S。
rollout保存原始Top16并跳过单独HF学生评分，是本地新接口，不能归为原fork已这样实现；
原fork仍有compute_log_prob前向。新接口训练轨迹overlap与无hook固定dev主指标分开。
代码/CPU验收见[修改记录](../audit/rollout_top16_shared_data_2026-11-12/README.md)；GPU未恢复。

完整稿列出的Entropy-Aware OPD暂为候选相关工作/强基线，本项目尚未独立核对其
固定版本公式与源码，不以该稿转述建立已复现证据。原稿称未定位SCOUT的说明
不覆盖本地已有证据：固定2609.38360v1实际题名为
《On the Off-Policy Teacher in On-Policy Distillation》，§3为SCOUT方法，来源仍保留。
正文已核对范围以本文件、manifest及原审计为准，不把来源稿自述阅读深度算作本项目新增阅读。

## 1. 核心论文与阅读范围

| 工作 | 固定版本与正文 | 本轮核对范围 | 对本项目的作用 |
|---|---|---|---|
| Rethinking OPD | [2604.13016v2](https://arxiv.org/abs/2604.13016v2)、[本地正文](../papers/sources/2604.13016v2.txt) | §2–6、§8、附录A.2/B/C.1 | 共享词表机制、overlap、SFT冷启动、模板及长前缀局限 |
| Simple-OPD | [2608.06802v1](https://arxiv.org/abs/2608.06802v1)、[本地正文](../ref/sources/2608.06802v1.txt) | §3–5、附录A/B.2、固定代码启动器 | 同教师CoT、LoRA warm-up、实际merge与OPD |
| CompassOPD | [2609.10154v1](https://arxiv.org/abs/2609.10154v1)、[本地正文](../ref/sources/2609.10154v1.txt) | §2–5、式(6)–(11)、附录A.1–A.6 | 异构文本span、教师家族增量、学生锚点及原配对 |
| SV Training | [2610.40998v1](https://arxiv.org/abs/2610.40998v1)、[本地正文](../papers/sources/2610.40998v1.txt) | §4.1/4.3、§5.1、附录C.1；本轮补查C.2/表8的粒度及刷新，已有固定代码 | 全局/逐题固定、周期刷新、teacher-forced训练hook与GRPO修正分别解释，不作为在线逐题OPD已验证证据 |

以上是方法/设置定点阅读，不将下载全文算作逐公式独立验证。PDF、官方快照、版本
与哈希保持在[papers清单](../papers/manifest.json)和[ref清单](../ref/manifest.json)。

## 2. 三篇OPD论文究竟研究了什么

### 2.1 Rethinking：兼容词表下也有可教性问题

§2.1先假设共同词表。§3–4的师生连线分别在Qwen3、R1-Distill-Qwen及其RL后继
内部进行。§3.2写“across model families”是在不同谱系中分别验证组内对照，
不是Qwen3教师跨接R1学生。§8确实讨论Qwen→LLaMA的预训练数据、tokenizer和
架构混杂，但没有因此完成本文自己的异构tokenizer实验或给出对齐算法。

教师准确率更高不保证OPD更有效。作者将模式相容和可转移能力增量总结为经验条件，
不是所有模型下的必要充分定理。共享候选消融支持其重要性，未证明任何提高overlap
的干预都有效；§6.2逐token梯度抵消解释也明确尚未直接验证。

普通学生轨迹`y ~ pi_S`，相同状态`c_t=(x,y_<t)`：

`Overlap@k = mean_t(|TopK(pi_S(.|c_t)) intersect TopK(pi_T(.|c_t))| / k)`。

k默认16；共有12个候选就是75%。分母不是并集大小。它比较下一token候选，不是
完整回答、语义或hidden state相似度。共享候选97%–99%是相关实验的概率质量，
不是序列位置比例。§5.2还有overlap集合更小、共享质量更高、效果更好的反例。

### 2.2 Simple：GPT文本SFT之后明确还有OPD

主分析使用DAPO RL后的Qwen3-8B-Base→Qwen3-1.7B-Base；thinking扩展为
Qwen3-4B-Thinking-2507→Qwen3-0.6B thinking；另有JustRL→R1-1.5B。
这些OPD连线均为兼容分词器的配对。

§3.2比较GPT-5.5和原OPD教师提供的CoT。**每个warm-up checkpoint后接75步OPD**；
逐token教师仍为DAPO RL后的Qwen3-8B-Base。完整流程是GPT文本SFT→Qwen内部OPD，
并非GPT继续作为OPD概率教师。SFT学生沿用自己的分词器，因此没有1对3概率对齐。
论文观察同OPD教师的CoT更适合后续学习，错误CoT也可有益，不能把结论简化成
“只需要正确答案”或“所有warm-up都一样”。

### 2.3 Compass：先有对齐和SFT，仍需改变监督信号

§3发现跨family即使经过SFT仍可能平台化；大幅增强教师并未同比增强学生收益。
对同一对齐文本动作：`ell_T - ell_S = (ell_R - ell_S) + (ell_T - ell_R)`。
方法传递后项并以冻结初始学生锚定。A.4的多token span用平均log-likelihood；
无法形成一致文本边界的位置mask。该评分不是完整文本概率，不能简称任意跨词表精确KL。

§3.1明确用过**Qwen3.5-4B→Qwen3-4B**比较教师规模；它支持本项目的4B级模型
选择。本文采用同一对模型测试Compass/方案1/2是新实验组，不能声称论文已经报告
这些组合的数字。共享广义Qwen谱系与独立family迁移分开标注。

## 3. 原论文模型、初始化与题源

### 3.1 本轮补查：逐题与刷新分别已有怎样的证据

SV Training §4.1的逐题差分与其全局均值，可以由初始学生训练前一次提取并冻结。
附录C.2/表8在Qwen2.5-1.5B、GSM8K、layer15的SV辅助SFT中报告：

| 方法 | alpha | 准确率（%） | 能支持的比较 |
|---|---|---|---|
| Mean-SV additive | 0.50 | 61.11 | 全局固定加法基线 |
| Per-sample additive | 0.50 | 62.93 | 相同加法方式下，保留逐题方向的结果 |
| Per-sample Slerp | 0.10 | 64.14 | 同时改变注入操作/强度，不单独归因于粒度 |
| Dynamic SV | 0.50 | 60.23 | 每200个优化步用当前LoRA学生重算全局方向；未超过固定全局 |

Dynamic SV重读的是GT/学生预测相关轨迹并更新全局均值，不是本项目当前题T−R、每轮在线逐题。
这些是原论文单一SFT设置的结果，本项目未独立复现；不外推为OPD刷新必然无效或逐题必然更好。
本轮仅定点核对本地C.2正文及表8，不把完整稿的自述阅读范围继承为本项目全部新证据。

相应实验需用两组桥接对照：同题同文本同S0的全局均值vs逐题缓存检验粒度；
同题同文本的逐题缓存vs当前学生每轮重提检验刷新。独立数据全局vs在线逐题同时改变
训练信息、题目覆盖、粒度与提取者，不能隔离各自贡献。参考文本缓存不等于激活缓存。
ACSD的固定全局、自教师侧注入与SV训练的逐题固定先例均不能当成新版在线提取已获验证。

### 3.2 冷启动概念与原配对

2026-11-11针对“冷启动”概念补充：SV Training §4.1/4.3的方向由同一学生读取
参考/自身回答所得，固定后用于teacher-forced监督训练，评测不注入，正文没有
规定仅前10步启用。其§5.1 GRPO是SV-Rescue：预先提取正确−错误轨迹方向，
用投影分数对错误rollout修正advantage，包含all-wrong零方差组；不等同于在
GRPO早期向hidden states直接加教师−学生方向。Rethinking §5.1为独立SFT
阶段后再OPD，§5.2则在OPD期间改变模板/题源。可将它们联系为改善训练信号
可用性的动机，但off-policy teacher forcing、GRPO零方差和OPD师生相容性
是不同问题，不能直接转移其有效性结论。

箭头均为教师→学生。是否“原配对”与是否“完整原配方复现”是两件事。

| 原设置 | 实际连线 | 初始化与数据 | 用途 |
|---|---|---|---|
| Rethinking §4.1；Simple §5.3 | JustRL-DeepSeek-1.5B→R1-Distill-Qwen-1.5B | DAPO-Math-17K；Simple用同教师回答warm-up | 成功对照、SFT/LoRA机制 |
| Rethinking §4.1 | R1-Distill-Qwen-7B→R1-Distill-Qwen-1.5B | 同DAPO与原配方 | 论文中收益弱的对照，本地需验证 |
| Rethinking §5.1/C.1 | Qwen3-4B non-thinking→Qwen3-1.7B-Base | OpenThoughts3数学域200K教师生成SFT，另约30K题OPD | 原SFT修复机制；回退路线F1 |
| Simple §3–4 | DAPO RL后的Qwen3-8B-Base→Qwen3-1.7B-Base | DAPO题源同时用于教师CoT warm-up和OPD | LoRA主分析 |
| Simple §5.2 | Qwen3-4B-Thinking-2507→Qwen3-0.6B thinking | 按统一设置的DAPO题源 | thinking扩展，不是首批必需 |
| Compass表2/A.1 | Qwen3-30B-A3B-Instruct-2507 / R=Qwen3-0.6B→Granite4.1-3B | 教师重答OpenThoughts-114k做共同SFT；DeepScaleR-Preview OPD | C1主要异构配对 |
| Compass §3.1 | Qwen3-30B-A3B-Instruct→Qwen3-4B | DAPO-Math；同family初始化不能自动视为Granite共同SFT | C2同教师控制 |
| Compass §3.1图1(b) | Qwen3.5-4B→Qwen3-4B | DAPO-Math；原SFT具体来源/跨教师初始化对应还需核验 | C3规模匹配；R=0.8B用于本项目Compass扩展 |
| Compass主实验A.1 | Qwen3.5-35B-A3B / R=0.8B→Granite4.1-3B、Qwen3-4B、OLMo-3-7B-Instruct-SFT | OpenThoughts-114k教师SFT；Qwen学生用DAPO，Granite/OLMo用DeepScaleR | 预算允许后的多异构学生扩展 |

Simple固定启动器指定教师`RedKiKi/Qwen3-8b-base-rl-dapo-17k`，默认rank32、
175 SFT+300 OPD steps；175不是所有学生的最佳时长，论文还比较150等设置。
普通Qwen3-8B不替代这个RL checkpoint；Qwen3-1.7B Instruct不替代Base；
普通Qwen3-4B不替代4B-Thinking-2507。Granite的精确发布ID/revision尚未核定，
不猜测ID后下载。所有论文名须落实到版本和原生配置。

## 4. 原始解码、训练和模板口径

| 来源 | 主要训练口径 | 评测口径 |
|---|---|---|
| Rethinking A.2 | Student Top-K16；batch64，rollout n4，LR1e-6，temperature1/top_p1；prompt1024/response7168 | AIME24/25/AMC avg@16；temperature0.7/top_p0.95，max31744 |
| Simple A | batch128，n1，LR1e-6，prompt2048/response8192；warm-up batch16，full LR5e-6、LoRA LR5e-5 | MATH-500 avg@4、AIME avg@16；base/instruct/thinking有不同解码，最长16384/24576/32768 |
| Compass A | batch128，LR1e-6，prompt2048/response24576；beta0.5；共同SFT 1 epoch、seq32768、LR2e-5、batch32 | 七项数学均avg@16；temperature0.7/top_p1/top_k40、presence penalty2，max24576 |

Compass七项为MATH-500、AIME24/25/26、HMMT25-Feb、HMMT25-Nov、HMMT26。
只报告已有子集时不使用其七项平均分。不同协议结果分表，不把文献分数当本地验收。

Rethinking §5.2两种任务模板原文如下，各模型仍使用各自原生chat template：

```text
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. {Question} Remember to put your answer on its own line after “Answer:”.
```

```text
{Question} Please reason step by step, and put your final answer within \boxed{}.
```

原论文对应表保留其模板；项目统一控制可用boxed，但需标注模板改动，不能默称原配方。
Rethinking教师离线SFT示范生成max12288、temperature0.7/top_p0.95，SFT seq14336、
LR1e-5、1 epoch；小样本DAPO实验不等于200K OpenThoughts3复现。

## 5. 本地资产与缺口

2026-11-11 Figure 8专项复核：本地普通`Qwen3-1.7B`不能替代Base。官方API可查询
`Qwen/Qwen3-1.7B-Base`及作者发布的`lllyx/Qwen3-1.7B-SFT`，已检查共享模型目录
最初未发现这两个checkpoint；用户随后补齐软链接，最新实查Base已存在，目前仅
未找到作者SFT。教师/Base的safetensors头及数据长度检查通过，详见
[当前资产盘点](../audit/rethinking_figure8_2026-11-11/current_assets.json)。
作者README说明SFT来自Base和OpenThought3-Qwen3-4B；
复用发布初始化不能写成本地重做200K SFT。用户新指定的真实目录是
`/home/zhenning/data/huggingface_datasets/AIME2025`（双g），I/II各15题。
原Figure 8位于PDF第12页，上排分别为AIME24、AIME25、AMC23，横轴展示0–200步；
只画AIME25须注明单benchmark。源码默认1 epoch，不能将图示窗口误称为明确的
200步训练规定。来源、数据文本差异及revision见
[Figure 8审计](../audit/rethinking_figure8_2026-11-11/source_audit.json)。

共享输入只读：`/home/zhenning/data/huggingface_models`与
`/home/zhenning/data/huggingface_datasets`。新产物留在项目内。盘点依据为
[选型证据](../audit/paper_alignment_2026-11-10/selection_audit.json)与
[tokenizer规则核查](../audit/paper_alignment_2026-11-10/tokenizer_scope_review.json)，
文件存在不代表训练兼容，配置哈希不代表全部权重或论文revision一致。

| 资产 | 当前状态 |
|---|---|
| Qwen3-30B-A3B-Instruct-2507、Qwen3-0.6B、Qwen3-4B、Qwen3.5-0.8B | 共享目录已有权重文件；新面板仍需模式/接口profiling |
| R1-Distill-Qwen-1.5B/7B/14B及JustRL-1.5B | 已有；JustRL实际位于共享数据目录huggingface_datasets，来源/revision仍需对应 |
| Granite4.1-3B、Qwen3.5-4B | 已检查位置未发现；C1/C3的明确缺口，当前不下载 |
| Qwen3-1.7B-Base | 已通过assets/models软链接复用；权重、配置、tokenizer存在 |
| 作者lllyx/Qwen3-1.7B-SFT | 最近盘点未找到；用户已暂缓该对照，新主方案不因修改文档恢复下载 |
| Simple的RL后8B、4B-Thinking-2507 | 未落实，非本轮必需，不为覆盖论文一次性全补 |
| DAPO原Parquet | 源码中17,917行；新任务须重新核验去重及train/calibration/dev划分 |
| OpenThoughts3_opd.parquet | 源码已有30,000行，不是200K教师SFT示范 |
| DeepScaler/train.parquet | 源码已有40,315行，含deepscaler来源；与Compass官方题集等价性待核验 |
| DeepMath_deduped.parquet | 源码已有14,116行，仅在对应分布实验时使用 |
| OpenThoughts-114k与相应教师SFT示范/初始化 | 尚未落实；须查原checkpoint/题集，不拿OpenThoughts3替代 |
| MATH-500、AIME24/AIME25/AMC | 已有输入；源码AMC23为83题，legacy83与strict40分开，不凭文件名认定年份 |

本地R1/JustRL普通词元映射与BPE规则相同；Qwen3-4B/Base普通映射一致，但教师多四个
控制词元及EOS/模板差异，输出head也有未定义行。当前首轮统一151665个共同有效输出词元，
生成/评分/训练/评测均在此支持归一化，证据见[实现记录](../runs/figure8_first_run_20261112/IMPLEMENTATION.md)。
不能据vocab_size认定相同或不同分词器，新面板仍逐对检查。

## 6. 方法近邻与主张边界

详细审计留在[第09篇](09_novelty_audit_cross_family_steering_opd.md)，不在主方案重复
历史宽检索。下表列出已有主参考的固定版本，避免清理文档后丢失阅读入口。

| 工作 | 固定版本 | 已核对关系与边界 |
|---|---|---|
| ACSD | [2609.38342v1](https://arxiv.org/abs/2609.38342v1) | 教师侧方向+OPD；§3.2–3.4/成本，本项目主候选干预学生受控生成与学习 |
| ST-OPSD | [2610.34005v1](https://arxiv.org/abs/2610.34005v1) | 特权教师的减引用/增检查方向；§3.1–3.2 |
| DIAL | [2610.31438v1](https://arxiv.org/abs/2610.31438v1) | 学生steering adapter+OPD；主干冻结、部署保留，§3.1–3.3 |
| OPMD | [2610.14022v1](https://arxiv.org/abs/2610.14022v1) | 教师状态投影注入学生后再内化；核对§3.2，跨family已报扩展为SFT |
| ASE | [2610.27012v1](https://arxiv.org/abs/2610.27012v1) | 学生激活干预帮助RLVR，§3.2–3.3；不能算作OPD基线 |
| SPTA | [2611.03133v1](https://arxiv.org/abs/2611.03133v1) | 学生前缀教师适配；§3–5/局限，任务与谱系范围需保留 |
| SCOUT | [2609.38360v1](https://arxiv.org/abs/2609.38360v1) | 在线适配教师对学生前缀的反馈；§3.1–3.2，不是固定学生方向 |

LastOPD/Latent-MOPD等早期辅助近邻仍需作为机制对照考虑；若不读取教师hidden，
说明接口与成本限制，而不是假装这些工作不存在。论文来源快照和证据深度均保留，
没有因清理探索文档删除原始研究证据。
