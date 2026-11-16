# Ablation 实验计划

实验统一命名为`ablation`，按`overlap → param → baseline`推进。100题及1000题的原T−S四方案均已完成。2026-11-16教师空间S−T的1000题补充实验已获授权启动，师生EasySteer小样本验收已通过，当前提取教师SV。param参数实验、训练消融与baseline方法对比均为后续计划。

## 当前授权：1000题教师S−T补充实验

本轮按文末补跑设置，在GPU6、7运行教师S−T方案5–8，SV使用完整1000题，沿原700／300源回答评分；复用原始师生回答、学生capture和概率缓存，并使用原T−S已提交无SV基线。师生GPU小样本验收已通过，正在提取教师侧SV。固定Slerp0.1、教师层29、读取总长2048、0次更新，到overlap／P_overlap图停止；不进入OPD、退火、param或baseline网格。模板hold保留，原100／1000题四组结果不改。

## 已完成：1000题原四方案

以下为已完成运行的历史设置。只加载T/S两模型，使用`RUN_NAME=overlap4_1000_v1`、`STOP_AFTER=panel`、`SCHEME_IDS=all`、`CONFIRM_TOP_N=4`，GPU0、1。新数据保留原100题并从本地同源OpenThoughts3数学子集补充900道不重复、两方prompt均不超过1024 token的题，seed42冻结700/300 select/confirm。四方案在两部分均评分，不只复核前两名。原100题输入、70/30划分及`overlap4_v1`结果完整保留；新实验重新整批生成1000条教师回答和1000条学生源rollout，不混用旧批次。

SV提取沿用sv_training规则：全部题目参与，不按正确性、生成截断或重复过滤；`sv_max_context_len=2048`包含prompt和回答，右截断后对全部保留回答token求平均，仅缺少有效回答token时跳过。global使用整个预先冻结的1000题提取池，包含confirm题目，因此confirm仍为训练侧复核。仅保存池化均值/范数与SV，不保存逐token隐状态。Slerp0.1、层26、Top16原始概率、零更新均保持。模板hold保留，新运行使用独立验收凭据和冻结配置。

以下是新入口的CPU准备／预览示例，历史结果仍在原目录：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/prepare_ablation_data.py
RUN_NAME=ablation PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1 bash scripts/run_ablation.sh --check
RUN_NAME=ablation PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1 bash scripts/run_ablation.sh --status
```

该历史输出已移至step3、PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1目录，select/与confirm/均包含4组共同前缀的双指标配对图与汇总，全部只做无梯度前向。每组分别输出清单前3条的PNG/SVG/CSV/PDF及实际注入点。默认HF_PARITY=0，仅EasySteer；显式设1才加HF数值参考。

设置STOP_AFTER=source_overlap可单独只计算普通源图，该模式跳过教师回答、capture、方向bank及SV注入。此次按上面命令显式选择panel，执行教师文本、普通学生源rollout、S0方向提取、T源前缀评分和4组固定前缀比较。两模式均不更新参数；完整OPD另行实现和授权。

## Overlap阶段的四组候选

本次授权比较以下4组固定强度候选，加共同无SV基线。**退火不属于本阶段，不按题目分配退火强度。**

| ID | 方向 | 位置 | 粒度 | 强度 |
|---|---|---|---|---|
| 1 | T−S | all | global | fixed |
| 2 | T−S | all | per_question | fixed |
| 3 | T−S | static_overlap | global | fixed |
| 4 | T−S | static_overlap | per_question | fixed |

固定T=Qwen3.5-35B-A3B、S=Qwen3-4B，S0冻结、0次参数更新。后续注入面板固定Slerp alpha=0.1与学生0-based单层26（36 blocks的75%）。all两组请求全部回答预测位置；static_overlap两组使用源分布预先冻结的位置，候选不重判idx。global/per_question共享相同文本、过滤交集和学生侧尺度。

原四方案方向为T−S：同一S0读取教师回答和自身普通无SV源回答，2048-token上下文内保留的回答原始激活逐题池化后作差；SV提取遵循sv_training的全样本、2048总上下文、保留回答token均值规则；global先平均原始差分再单位化，逐题零方向精确无干预。新增教师空间方案见下一节。当前不使用Ref模型。source_overlap模式不需要这些SV步骤。

选择候选子集时，显式设置`STOP_AFTER=panel`、`SCHEME_IDS=1,3`等子集及新PARAM_NAME；原四组配置为confirm_top_n=4，两部分都比较完整四方案；本轮仅新增下述教师镜像对照，不自动扩展其他方向、层或强度网格。源缓存是否可复用，须按checkpoint、前缀、原生概率及代码身份独立校验，不能改旧run配置后直接续接。

## 新增：教师空间S−T，与原T−S镜像对照

记μ(A←B)为模型A在自身隐空间读取模型B回答后的正文均值。原方向与新增方向分别为：

$$
d_i^{T-S,S}=\mu_i(S\leftarrow T)-\mu_i(S\leftarrow S),\qquad
d_i^{S-T,T}=\mu_i(T\leftarrow S)-\mu_i(T\leftarrow T).
$$

S−T由同一个冻结教师T0分别读取学生无SV源回答和教师自身参考回答，按“学生回答均值−教师回答均值”作差。两端都是教师隐状态，宽度2048；原T−S位于学生宽度2560的隐空间。S−T不能通过直接对原学生空间向量取负得到，不进行跨模型隐状态相减或投影，也不引入Ref。

| ID | 方向／隐空间 | 读取者与注入对象 | 位置 | 粒度 | 默认层／强度 |
|---|---|---|---|---|---|
| 1 | T−S／学生 | S0 | all | global | 26／Slerp0.1 |
| 2 | T−S／学生 | S0 | all | per_question | 26／Slerp0.1 |
| 3 | T−S／学生 | S0 | static_overlap | global | 26／Slerp0.1 |
| 4 | T−S／学生 | S0 | static_overlap | per_question | 26／Slerp0.1 |
| 5 | S−T／教师 | T0 | all | global | 29／Slerp0.1 |
| 6 | S−T／教师 | T0 | all | per_question | 29／Slerp0.1 |
| 7 | S−T／教师 | T0 | static_overlap | global | 29／Slerp0.1 |
| 8 | S−T／教师 | T0 | static_overlap | per_question | 29／Slerp0.1 |

层位均按75%深度设定：学生36 blocks对应0-based26，教师40 blocks对应0-based29。两种读取者均使用自身tokenizer和聊天模板，总上下文2048，保留回答token均值；不按正确性／截断／重复过滤。启用镜像对照时，两侧bank使用两个读取池的共同有效题集；各隐空间独立计算尺度，global先平均原始差分再单位化。仅保存池化均值／范数和4份bank，不保存逐token隐状态。

新增方案固定学生无SV分布，将S−T注入教师读取完整既有学生推理的因果前向，输出教师Top16概率；额外1个生成token仅供评分前向并丢弃，不生成新的教师／学生推理。SV提取限制2048，不截短这一步沿学生完整回答的评分。

静态阈值仍只用Select的无SV分布校准、一次冻结。S/T共用相同文本span的选点：将学生侧冻结mask按共同span投影到教师原生token范围，再使用教师自己的prompt_len−1+t预测位置。不能把学生token序号直接作为教师位置。教师原生实际注入率与映射到学生坐标的覆盖率分别报告，不要求两方原生token占比恰好相同。

两边的overlap／P_overlap统一保留在原学生坐标上，支持题目等权与token加权比较。八组联合运行时，先提交学生侧无SV重放基线，教师组固定同一份学生概率并复用原教师无SV评分，构成与学生组完全相同的无SV基线。教师另做一次无SV重放，保存teacher_repeat_drift.json；Qwen3.5仍不支持batch-invariant，不自动从候选增益中扣除漂移。

这项镜像对照同时改变读取者、隐空间与干预对象，不能把结果差异单独归因于同一个向量的符号。教师组表示**教师分布向固定学生分布适配**，学生组表示学生分布受干预；两者分别标注intervention_role，教师组overlap上升不能称学生能力或OPD学习提升。教师组图中的注入点表示映射到学生坐标的教师span，真实教师mask另存native_steering_mask。

默认TEACHER_ABLATION=0保留原四组。以下仅CPU预览，不启动模型：

```bash
TEACHER_ABLATION=1 TEACHER_LAYER=29 SCHEME_IDS=all bash scripts/run_ablation.sh --check
TEACHER_ABLATION=1 SCHEME_IDS=5,7 bash scripts/run_ablation.sh --check
```

开关映射为include_teacher_ablation=true，新增配置teacher_layers=[29]及teacher_direction_version。默认RUN_NAME=ablation，PARAM_NAME=n1000_slerp0.1_S26_T29_seed42_v1；启用时默认Confirm包含全部8组（只选子集时相应缩小）。不覆盖或改写旧run配置与结果。后续运行仍须显式授权并通过新的师生GPU验收；本轮不执行--run。

新增产物在step2的reader_teacher/activations、reader_teacher/banks内，与reader_student/分开。Select／Confirm按student/与teacher/分别整阶段提交，在各split根目录合并summaries.json，不改写已提交的角色清单。底层记录绑定读取者、checkpoint、方向版本、层、文本与代码身份；旧T−S验收不能冒充教师S−T验收。教师扩展此前172项CPU测试、包检查及八组／子集CPU预览通过；本次复用改造另做验证，教师双卡实际capture／注入尚未验收。

## 数据、指标与留档

唯一模板[configs/ablation.json](../configs/ablation.json)，主编排[scripts/run_ablation.py](../scripts/run_ablation.py)，逐候选评分[scripts/run_overlap_ablation.py](../scripts/run_overlap_ablation.py)。新阶段路径见[输出结构](../输出结构.md)。原始1000题输入与700/300划分已移动到step0的n1000_slerp0.1_layer26_seed42_v1目录；由[prepare_ablation_data.py](../scripts/prepare_ablation_data.py)在CPU从本地Parquet生成并冻结；sampling_manifest.json记录seed、题目及来源分片SHA、筛选条件和原100题身份。原100题及70/30划分已移动到step0的n100_slerp0.1_layer26_seed42_v1目录，内容不改。激活/对齐后的实际有效题数另报；confirm为训练侧冻结复核，不称独立dev。

- `overlap`：共同UTF-8安全span起点的两方原生Top16最大一一前缀兼容匹配数/16。
- `P_overlap`：同一兼容关系下，仅使用vLLM返回的Top16 ID与原始logprob，最大化可匹配概率质量。取exp(logprob)，不对Top16重新归一化，忽略其余尾部。
- 同词表仅按相同ID；异构普通token完整字节互为前缀，EOS只匹配终止，其他控制精确匹配，空普通token非万能前缀。
- span起点值向学生span广播。终止EOS目标与不可对齐位置为NaN，明确覆盖率；EOS作为候选概率保留。
- metric_version=4、matching_rule_version=native_token_prefix_compatibility_v1。两指标共用一份Top16数据；删除完整词表概率、字节概率投影与自定义概率RPC。旧指标、阈值、mask、排名和GPU凭据不混接。

师生输出位于各参数的step1/teacher与step1/student，每题一个JSON，text字段为生成正文；100题和1000题分别在各自PARAM_NAME下。当前仅计算分布overlap，不做近似答案准确率评测。

完整定义和例子见[overlap.md](overlap.md)及[完整研究稿§12.2](Steering_OPD_完整研究方案.md#122-两种跨family-overlap并行记录)。固定前缀指标变化只表示即时干预，不代表学习收益。

两项指标的源值只算一次并以概率/对齐SHA绑定缓存。阈值采用分数count、排序和累计计数，支持overlap与P_overlap，保留原严格比较和最接近20%的规则。已用255484个select位置验证：两项阈值及100题实际mask与原实现完全一致。static_overlap组只在select上校准阈值，按gate_metric冻结strict-< mask，confirm沿用；rank_metric显式决定排名，默认均overlap，两指标始终分别记录。source_overlap模式仅输出原始分布与图；当前panel模式按预先固定的select/confirm规则评分，结果交用户检查。

每请求seed统一42，启动前固定PYTHONHASHSEED。全量pending由EasySteer/vLLM自动调度，不设置max_num_seqs或人工CUDA Graph尺寸。学生启用batch-invariant，T保留默认且一次评分SHA冻结，重复漂移单独诊断。每位置Top16 ID(int32)、logprob(FP32)及观察token logprob共132字节，约38万位置为48 MiB，另有文本、文件元数据和小型池化结果；逐token隐状态不落盘。教师权重只核验，不下载。

每次公开generate/capture调用连续完成后才处理并保存；恢复粒度为完整engine_call，StageCheckpoint以SHA清单及原子complete.json提交。未提交整批重算，已提交整批先CPU校验后复用，无引擎内逐请求落盘或KV/RNG恢复。console.log、events.jsonl、progress.json与startup.log保留北京时区时间、心跳、阶段、前向完成和提交数；未知ETA不伪造。

终端不显示外层步骤/阶段tqdm，只打印当前阶段、开始/结束及耗时、缓存复用或失败。EasySteer/vLLM的模型加载、CUDA Graph、生成、评分和capture原生tqdm继续实时显示并写入console/startup日志。15秒心跳和完整JSON事件只写events.jsonl/progress.json，不向终端刷屏；--status/--watch仍可查看计数与ETA。原四组面板已完成；当前新增S−T仅做代码、文档和CPU验证。

## 固定强度与线性退火

这项消融归入后续有真实训练进度的param/OPD实验，本轮overlap使用固定强度，教师镜像组也不交叉退火。保留退火规则与强度计算接口，不运行题级退火诊断，也不生成题目退火idx表。

比较固定强度a与100个预先确定的逻辑训练批次：

$$
a_k=2a\max(1-k/80,0),\qquad k=0,\ldots,99.
$$

| 训练批次idx | 0 | 20 | 40 | 60 | 79 | 80–99 |
|---|---:|---:|---:|---:|---:|---:|
| 倍数 | 2 | 1.5 | 1 | 0.5 | 0.025 | 0 |
| a=0.1示例 | 0.2 | 0.15 | 0.1 | 0.05 | 0.0025 | 0 |

同批题目及对应轨迹的学习前向沿用绑定强度。训练批次idx按全局已提交进度推进；原始题目ID、请求完成先后、回答token、microbatch、同批重复epoch和重试均不单独推进或重置时钟。最后20批保留普通rollout、教师监督和参数更新，仅撤去SV及其专用提取成本。

Slerp会单位化方向，正系数预乘SV会抵消；预先算alpha，通过已有`VectorSpec.scale`传入，保持原payload，以整批一次EasySteer调用提交，不按强度拆调用、不增额外模型前向。alpha=0直接`steering=False`，在方向查询和selector构造前跳过。方向SHA缓存和scale分开，配置指纹仍含scale，不同强度不能复用受干预KV。

Slerp alpha限制[0,1]，2倍起点要求a≤0.5；a=0.75对应的起点1.5不能直接运行或静默截断。若改算子或定义外推，另行明确并验收。100个离散批次的名义平均系数为0.81a，固定组为a，不宣称剂量相同。最终均撤去SV评测裸学生，并记录实际扰动、token、注入率和总成本。真实OPD调度与训练恢复尚未实现，不把此计划写成已完成训练实验。

用户最新缓存要求：以后capture得到的逐token隐状态只用于内存池化，不写token_activations文件；保留pooled.pt、两份SV bank、源/教师Top16概率、mask和结果。现有原始隐状态经清单校验后按hidden_state_cleanup.json记录清理，原pool与bank字节保持，artifact_index同步更新。教师原生成长度及SV读取2048、注入评分参数均保持。

## 分阶段输入输出与复用

`RUN_NAME`表示ablation／param／compar实验类型，当前仍只执行overlap流程；`PARAM_NAME`表示具体设置，改参数或候选子集时显式使用新名字。默认四组为n1000_slerp0.1_layer26_seed42_v1，教师八组为n1000_slerp0.1_S26_T29_seed42_v1。不自动运行参数网格或baseline。

step0保存数据；step1分别保存teacher／student；step2按reader保存capture和bank；step3保存原生概率、对齐、源指标、calibration和Select／Confirm；step6保存对照CSV、分析Markdown和来源索引。step4／5仅预留。配置与job位于metadata，调度器与console位于logs。

改alpha复用回答、capture、bank、对齐和原生评分；改层位／SV长度保留回答及原生概率，只重算受影响的SV和下游；改门控指标或比例复用SV及概率。原生教师评分覆盖全部源问题，不依赖SV筛选，以便换层后仍可使用。实际使用的每个上游由合同、完成标记、文件SHA及固定引用核验；完整规则和CPU历史注册命令见[输出结构](../输出结构.md#7-参数变更与复用规则)。

100／1000题已完成产物现已真实移动到新结构，原始字节与提交清单不改，旧位置已删除。1000题复用引用已同步新owner及descriptor SHA，100题旧协议单独标记历史结果；当前代码下的新候选评分仍须新的GPU验收与明确授权。

本次分阶段复用改造通过191项CPU测试及包检查，并用真实1000题缓存验证alpha从0.1改为0.2时，上游七类组件全部命中；无GPU调用。随后按用户要求完成同盘实际迁移，仍未复制原回答和大张量。

本次文件迁移保留100题Select四组／Confirm两组和1000题两部分各四组。step6增加各批comparison.csv／analysis.md及两批合并表；精确位置见[输出结构](../输出结构.md#8-已完成100题1000题的实际迁移)。已完成参数名禁止重新执行，新实验须新PARAM_NAME。其它五个旧版／未完成实验目录及无用启动记录已删；本次无模型调用。

## 本轮补跑：1000题教师S−T（2026-11-16）

用户已明确授权启动教师S−T补充实验，使用GPU6、7（0、1当前繁忙），仅方案5–8，Select700／Confirm300各四组，0次更新，到双指标图和汇总停止。参数目录为n1000_slerp0.1_S26_T29_seed42_st_only_v1。SV使用完整1000题，T层29、Slerp0.1、prompt+回答总长2048，保持sv_training全样本池化及pooled_only。

复用原1000题师生回答、学生capture／bank（共同有效集相同时）、原生教师评分、对齐和源指标，不重新生成回答。STUDENT_BASELINE_RUN指向原1000题step3参数根，复用其Select／Confirm已提交学生baseline，确保与原T−S比较使用完全相同的无SV分布。历史mask仅允许去除direction_version／direction_versions字段；旧mask须按当前源prefix完整重建核验，概率SHA、指标定义、阈值、位置和gate不得改变。教师重放漂移单独报告。

入口参数：TEACHER_ABLATION=1、SCHEME_IDS=5,6,7,8、QUESTION_COUNT=1000、CONFIRM_TOP_N=4、GPU_IDS=6,7；STUDENT_BASELINE_RUN显式选择已完成1000题根。新教师capture／注入须先通过本次EasySteer GPU验收；模板execution_hold不变，仅独立运行配置承载本次授权。
