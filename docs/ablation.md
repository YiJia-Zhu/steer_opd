# Ablation 实验计划

当前全流程使用同一1024题训练池：保留原1000题并按seed42同源补充24题，不再划分Select／Confirm。1024题共同用于SV提取、overlap及OPD；固定global／all／Slerp0.1、教师第29层，每步32题×4条回答，最多64步，每16步及末步评测裸学生，step0也评测。入口及参数见[代码框架§8.1](代码框架逻辑.md#81-环境和手动入口)。1024题overlap已完成；旧OPD在2次更新后因UTF-8文本对齐错误退出。修复通过429项CPU回归和包检查，2026-11-19已在tmux ste中以`opd_n1024_project_outputs_utf8_v2`从初始学生重跑，继续使用GPU2／3和原SV，训练输出位于项目`outputs/step_4_train/`。原冻结配置和结果保留。下文Select／Confirm及参数搜索设置属于原冻结实验，不用于新的全流程。

当前新配置使用`minimal_v1`清理过程数据，并使用`all_models_latest_state_v1`保留各保存点的模型：每16步及末步的模型都保存，优化器／scheduler／RNG／dataloader状态只保留最新完整点。当前批过程数据及overlap状态滚动更新，标量曲线长期保留。历史实验输出保持原字节，模板hold不解除；细则见[代码框架§8.5](代码框架逻辑.md#85-最小保留策略)。

从 2026-11-20 的新训练运行开始，step 0、每次定期评测及末步评测的 530 条回答写到训练目录的 `evaluation/responses/step_<步>.jsonl`。每行仅含冻结题号 `question_id`、原题面 `question` 和完整回答 `text`，与该步的准确率来自同一次生成。已冻结的 `verbatim_v3` 继续使用原评测设置。

## 动态 SV

静态 SV 先在 step 1–3 准备师生回答、教师空间方向和 overlap 面板，后续 OPD 复用这个方向。动态 SV 从 step 0 的同一冻结 1024 题直接进入 step 4，沿用静态 SV 的教师空间 S−T 定义，每个训练步重算一个全局方向。

1. 当前学生为当批 32 道题各生成 4 条回答，使用每题 seed 42／43／44／45。冻结教师关闭注入，为同一批 32 道题各生成 1 条参考回答，默认生成上限为 2048 token。教师采样沿用静态 step 1 的 seed 42、temperature=1、top_p=1、top_k=-1、presence_penalty=2 和原生终止 token。
2. 同一个冻结教师关闭注入，读取两组回答。提取沿用静态协议，将题目 prompt 和去掉首尾空白的回答一起分词，默认保留前 2048 token，取保留的回答 token 在教师第 29 层的 FP32 平均激活。每题先等权平均各学生回答的池化值，再减去该题教师参考的池化值。对有效题的原始差分等权平均，得到当步全局 S−T。
3. 将该方向以默认 Slerp 0.1 注入教师，沿同一批完整学生回答评分。OPD 的学生旧概率继续使用 rollout 返回的原始 logprob。教师评分与梯度更新沿用原训练链路。

每步生成 128 条学生 rollout 和 32 条教师参考回答。提取上下文用 `--sv-max-context-len` 设置，教师生成上限用 `--reference-max-tokens` 设置。教师参考或任一学生回答没有保留回答 token 时，该题不参与方向估计。全批无有效题或平均方向退化时，本步使用无注入评分并记录状态。参考回答、方向和逐 token 激活在批次内使用，日志保留有效题数、方向范数与注入状态。运行使用独立身份，从本动态运行的 checkpoint 恢复后按批重新计算。

```bash
bash scripts/run_dynamic_sv.sh --check
# 训练由显式 --run 启动
bash scripts/run_dynamic_sv.sh --run --gpu-ids 4,5 --run-name opd_dynamic_st_seed42_v1
```

已完成 CPU 验证，实际双卡生成、捕获和训练吞吐待 GPU 验收。[训练入口](../scripts/run_dynamic_sv.sh)

## 无 steering 的训练基线

`scripts/baselines.sh` 开头的 Bash 数组 `BASELINES=(vanilla compassopd)` 决定默认顺序，也可以用环境变量 `METHODS` 选择方法。一次调用共用同一对 GPU，前一项成功退出才会进入下一项。Vanilla、CompassOPD 和 SFT 从初始 Qwen3-4B 开始，SFT+OPD 从已保存的 SFT 模型开始。各自保存和恢复 checkpoint。[启动脚本](../scripts/baselines.sh)

| 方法 | 每批训练信号 | 默认运行名 |
|---|---|---|
| `vanilla` | 当前学生采样，原生 Qwen3.5-35B-A3B 教师评分，使用传统 OPD loss | `opd_n1024_vanilla_seed42_v1` |
| `compassopd` | 同一原生教师和教师同家族参考 Qwen3.5-0.8B 对学生回答评分，使用教师与参考的 span 平均 logp 差，加上相对冻结初始学生的逐 token 锚定项，默认系数 0.5 | `opd_n1024_compassopd_seed42_v1` |
| `sft` | 复用 step 1 正常结束且非空的教师回答，从初始 Qwen3-4B 全参数训练，只在教师回答与结尾 EOS 上计算交叉熵 | `sft_n1024_seed42_v1` |
| `sft_opd` | 从 SFT step 32 开始，用当前学生采样和原生教师评分做标准 OPD | `opd_from_sft32_seed42_v1` |

OPD 基线的评分请求均传入 `steering=False`。CompassOPD 复用项目按论文实现的 loss 和文本对齐，在本项目共同的初始学生上比较。参考模型和锚定系数分别由 `REFERENCE_MODEL`、`ANCHOR_COEF` 设置。[目标函数](Steering_OPD_完整研究方案.md#94-后续训练loss默认传统opd可选steercompassopd)

```bash
bash scripts/baselines.sh --check                            # CPU 预览两组
METHODS=vanilla bash scripts/baselines.sh --prepare          # CPU 冻结单组计划
METHODS=compassopd GPU_IDS=0,4 bash scripts/baselines.sh --run
METHODS="compassopd vanilla" GPU_IDS=0,4 bash scripts/baselines.sh --run
METHODS=sft bash scripts/baselines.sh --check
```

默认 GPU 池为 2、3，每个方法为指定两张卡最多等待 3600 秒。两组沿用冻结的 1024 题、AdamW lr=1e-6、每步 32 题 × 4 条、每题 seed 42／43／44／45、最多 64 步。temperature=1、top_p=1、top_k=-1、presence_penalty=2、回答上限 16384。step 0、每 16 步和末步评测裸学生，每 16 步和末步保存模型。两组的 overlap 和 P_overlap 都按学生与教师的共同 span 统计。训练沿用 TensorBoard 和 minimal_v1 保存策略。

输出位于 `outputs/step_4_train/ablation/<模型组合>/<训练数据集>/shared/<运行名>/`。可用 `VANILLA_RUN_NAME`、`COMPASSOPD_RUN_NAME` 分别设置运行名。重跑同一配置时各方法只恢复自己的 checkpoint，已完成最终保存和评测的方法核验后跳过。变更实验配置使用新的运行名。

当前 steer 作业按 SHA 绑定训练源码。基线入口把训练模块及差异发布到项目内 `.runtime/baselines/` 的独立目录，plan 记录实际使用的实现。准入、Ray 和评分子进程都使用这份代码，两组仍共用原有的 GPU 准入和配额检查。

### 教师轨迹 SFT

CompassOPD 论文先在 OpenThoughts-114k 上用强教师替换原答案，只保留正常结束且非空的生成。SFT checkpoint 单独接受评测，也作为标准 OPD 和 CompassOPD 的共同起点。后续蒸馏用另一套题源，Qwen3-4B 用 DAPO-Math，Granite4.1-3B 和 OLMo-3-7B 用 DeepScaleR-Preview。[论文 §5.1、附录 A.2–A.3](../ref/2609.10154v1_compass_opd.pdf)

仓库里的 `METHODS=sft` 是独立的本地对照，题源仍为 step 0 的 OpenThoughts3-1.2M 数学 1024 题，监督文本来自已提交的 step 1 冻结教师回答。默认 974 条符合论文过滤条件，另 50 条达到生成上限。学生使用与 rollout 相同的非 thinking 题面模板，教师回答用学生 tokenizer 编码，保留 `</think>` 前后的原始回答内容。可通过 `SFT_TEACHER_DIR` 选择同一冻结题池的其他已提交教师生成，通过 `SFT_RUN_NAME` 设置运行名。

默认用 VERL FSDP 全参数训练一轮，AdamW 学习率 `2e-5`、weight decay `1e-6`、cosine 调度和 10% warmup，总序列上限 32768 token。每步最多 32 条，默认末批 14 条，共 31 次更新。指定 `SFT_TOTAL_STEPS=32` 会把这 14 条分成 8 条和 6 条两个批次，974 条仍各使用一次。先评测初始学生，训练后逐个评测已保存 checkpoint。默认评测 step 0、每 16 步及末步，每次生成的 530 条完整回答与准确率一并保存。SFT 的保存及评测间隔用 `SAVE_EVERY` 设置。

### SFT 后的标准 OPD

`METHODS=sft_opd` 默认使用 `sft_n974_32steps_seed42_v1/checkpoints/global_step_32`。准备时验证源模型分片，再用与 SFT 评测相同的 VERL 合并方式在 CPU 导出 BF16 初始权重。导出按源 checkpoint 身份缓存。OPD 重新初始化优化器、scheduler、数据加载器和步数，step 0 评测从 SFT 权重开始。训练继续用冻结 1024 题、vanilla 的 loss 和 EOS 监督。

`--check` 预览源 checkpoint 和 OPD 计划，`--prepare` 会实际生成导出和冻结计划。`SFT_OPD_CHECKPOINT` 可指定另一个已提交的 SFT checkpoint，`SFT_OPD_RUN_NAME` 设置独立运行名。`TOTAL_STEPS` 表示 SFT 之后追加的 OPD 步数。

```bash
METHODS=sft_opd TOTAL_STEPS=32 bash scripts/baselines.sh --check
```

## 当前消融：S−T完整16组矩阵（2026-11-19，已完成并归档）

1024题OPD的GPU池固定为2／3，其代码、配置和输出身份保持不变。独立消融固定教师空间S−T、教师0-based29层和alpha=0.1，已完成overlap／P_overlap、图表和汇总，0次更新。原学生T−S仅保留作历史对比。此前建议的额外反向／随机方向及24点层位／强度网格不纳入本次执行。

| 轴 | 取值 | 实际含义 |
|---|---|---|
| SV粒度 | global／per_sample | global沿用当前单条学生回答的提取协议；per_sample先由S0每题rollout四条，再由教师读取并形成每题一个方向 |
| SV提取长度 | 2048／full | 2048限制每份prompt＋回答的总token数；full读取完整推理，不右截断。只改变提取，评分前缀始终完整 |
| 教师注入位置 | all／bottom_half | all注入教师所有回答预测位置；bottom_half逐题按原无SV overlap排序，选择最低50%的合法教师原生token位置 |
| 注入算子 | additive／Slerp | additive为h＋0.1×对应原始S−T差分，无额外范数缩放；Slerp使用单位方向、alpha=0.1 |

共2×2×2×2＝16个SV条件，另有一份共同无SV基线。当前全流程的global／2048／all／Slerp0.1即主参考，完成提交与SHA核验后直接纳入矩阵；无SV概率、指标和教师重复评分证据也直接复用，均不再GPU重算。因此主参考可复用时仅新增15个候选评分。尚未提交的依赖显示待就绪，不用进度条完成数冒充提交。

### 提取与评分分开

所有候选沿当前全流程原每题一条学生回答评分，复用其真实token与学生Top16，教师参考回答也保持同一份。四条提取rollout不替换评分回答，不增加overlap的每题评分样本数。两个长度、两个位置和两个算子共享同一批提取文本。

per_sample固定顺序：S0关闭SV，每题rollout四条 → 整批／分片提交并SHA核验 → T0分别读取四条及同题教师参考 → 每条独立做回答token的FP32均值池化 → 四个学生回答池化值等权平均减教师参考池化值 → 每题一个S−T方向，并只注入对应题目。不能拼接后按长度加权，也不能先逐条单位化。零方向精确无干预，无对应题目向量时不回退到global。

随机数与训练相同：每题固定slot 0→42、1→43、2→44、3→45，跨题和批次复用，不按整批继续递增；分发worker前给四个n=1请求分别设置seed。重试／resume保持(question_id,slot,seed)不变，与GPU／worker调度无关。记录多样性；整批所有题四条都相同时报错排查，少量自然重复保留，不挑选或重采样。任一输入无有效回答token时记录缺失，不静默减少四条数量。

用户已明确global保留当前单条提取协议，per_sample使用四条提取。这一轴同时改变方向粒度与学生提取采样量，报告中须披露，不能将差异全部解释为纯粒度效应。

### 排序取最低50%的静态mask

只使用原单条评分回答上的无SV overlap。将共同合法span起点的overlap映射到教师原生预测位置，逐题按(overlap值,教师位置)稳定排序，选前floor(N_valid／2)个位置；并列按位置顺序打破，不再用strict-<阈值近似20%或50%。EOS目标及无法形成合法共同span的位置不参与排序；报告合法位置覆盖与实际注入比例。

真实教师mask与映射到学生坐标的展示mask分别保存；一个span内只选部分教师位置时，学生展示mask标明“该span至少一个教师位置被选中”，不能据此声称每个学生token都实际注入。mask在无SV分布上冻结，候选不重新排序；整体指标统计所有共同有效位置，不能只统计选中位置。

### bank、缓存与结果复用

四种bank组合为global／2048、global／full、per_sample／2048、per_sample／full。第一种直接核验复用当前bank；只需新建其余三种，不为15个候选重复提取。per_sample的四条回答只生成一批，2048和full都读取同一批；各bank供all／bottom_half及两种算子共用，原始差分及尺度、有效题集、文本／seed／模型／上下文／协议身份保留。alpha不参与bank提取依赖。

原单条评分回答、教师原生概率、对齐和源双指标必须来自当前运行已提交的同一来源，不混入旧1000题或不同采样协议。原无SV和主参考不重跑；新候选仅获取受控教师概率。教师原无SV重放漂移复用现有凭据，单独展示，不自动扣除。

消融实现和执行时使用项目内隔离工作树，避免改变全流程的源码指纹。完成后按用户要求统一迁至主`outputs/`，保持相同的项目相对路径及原文件字节，19个产物描述文件、bank退休凭据和上游SHA全部核验通过，原owner不改写。新产物使用新的PARAM_NAME，遵守step1／2／3／6结构；minimal_v1只退休新bank完成校验后的本运行pool，所有bank保留可复用。原工作树已删除；完整实现、启动脚本和补丁保存在结果汇总目录的`metadata/implementation/`，其71个冻结Python文件与原源码指纹一致。

### 并行执行与汇总

本次用户已授权修改代码／文档、完成检查后立即启动16组消融。物理GPU2／3永久排除，包括当前暂时空闲的3；其余0／1／4／5／6／7持续按显存、利用率、实际计算进程及租约检测。繁忙卡不使用，不停止其他项目进程；GPU6当前有其他作业，只在以后实际空闲并通过准入时才可用。

每个学生引擎用一张卡，教师引擎用两张；允许多个独立引擎并行，总准入上限为其余六张卡，统一锁与租约避免本消融内部抢占。先完成新协议GPU验收，再并行生成每题四条的题目分片；分片完整提交后汇合。bank按依赖只构建一次，四种bank对应的候选组在依赖就绪后并行评分；同一组共享模型加载和bank。未就绪的无SV／主参考等待原全流程提交，不另行重跑。

已通过437项CPU回归、包检查、学生四请求seed验收及教师加法／Slerp验收。矩阵`matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1`于2026-11-19 04:16全部完成：4096条提取rollout、4类bank、1024道共同有效题，新增15组评分并复用1组主参考。

归档后的[完整CSV](../outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1/comparison.csv)、[汇总分析](../outputs/step_6_summary/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1/analysis.md)和[迁移凭据](../audit/matrix16_20261119/relocation.json)均位于主项目。所有20783个源文件SHA核验通过，其中13483个新文件复制入outputs，7300个已有文件保持原字节。冻结配置中的原工作树绝对路径属于历史执行记录；产物读取统一使用相同的项目相对outputs路径。

只清理已完成工作树、旧测试目录和闲置缓存／日志。`.runtime/verl_training`、EasySteer环境、源身份缓存、当前tmp／ray、GPU锁及租约和远端挂载程序继续保留；迁移及清理前后，OPD源码、plan、准备配置及SV的SHA均保持一致，未重启或停止训练。

每个候选提交后更新CSV／图／来源索引，16组全部留档，不设置Select／Confirm。分别报告overlap和P_overlap的绝对值及相对共同基线的百分点差、题目等权和token加权、1024题覆盖、合法位置数与实际注入率；P_overlap使用Top16原始概率，不重新归一化。按题配对区间只表示固定bank／固定回答下的不确定性，overlap改善不直接证明学生学习收益。

## 历史参数搜索授权：固定S−T／all／global

按用户最新指令，下一轮只保留教师空间S−T的all／global方案，搜索以下三轴；完整待实现项同步于[TODO](../TODO.md)。2026-11-17用户明确要求使用物理GPU2／3启动并复用已有结果；网格已接入入口，219项CPU测试及包检查通过，两组旧结果和源缓存已核验。旧三档配置与结果保持冻结。

| 参数轴 | 搜索范围 |
|---|---|
| 提取／注入层 | 教师40 blocks的50%／75%／100%，0-based19／29／39；同层提取与注入 |
| 注入方式 | Slerp、直接加法 |
| 强度 | 两种方式均为0.05／0.1／0.3／0.5 |
| 固定项 | S−T、教师注入、all预测位置、global方向、1000题、700／300划分、2048提取上下文、seed42、0次更新 |

共3×2×4=24组配置，各做Select／Confirm；层29的Slerp .05／.1两组优先核验复用，预计新增22组。已有.2保留作历史对照，不重复计算。师生回答、原生概率与对齐复用；新增层19／39的SV捕获与bank，每层在不同方式／强度间共用。

保持已有预测位置、FP32正文池化及Slerp实现，避免混入额外协议变化；第39层取final norm之前的block输出并在同处注入。加法沿原始global均值差分d，h'=h+alpha*d；Slerp沿其单位方向，alpha为角度比例，两者各自解释强度，不额外加入激活范数缩放。新层位与加法调用链须先完成CPU核对及相应GPU验收；范围止于双指标图和汇总。官方mask／数值复现对照、OPD及独立评测另列。

2026-11-17 14:48:51已启动，监督器PID2522901；两组旧结果已纳入汇总，首个新增点（层19、Slerp .05）于14:50:48通过师生GPU验收，现进入教师层19的SV提取。实时阶段以search_progress.json和日志为准。

当前入口：`scripts/run_param_search.py --run --gpu-ids 2,3`。搜索名`st_all_global_L19_L29_L39_ops_n1000_seed42_v2`，新点命名`n1000_<operator><alpha>_S26_T<layer>_seed42_st_all_global_v2`；step6同名搜索目录保存`search_progress.json`、`logs/search.log`、`comparison.csv`和来源索引。GPU验收绑定注入方式／强度／层号，通过后执行；每点串行、最多两卡、只用指定空闲卡。旧两点只选取scheme_05参与新汇总，原四组文件保持。所有点报告700题Select和300题Confirm，按Select题目等权Δoverlap选参。空间按每点约4.32GiB保守准入预算检查，当前约114GiB；不是最终实测占用承诺，空间不足可保留已提交阶段后恢复。

## 已完成参数实验：S−T强度扫描

该扫描启动时未记录完整参数范围，因此从已完成alpha=.1、教师层29出发执行了下列最小强度扫描。2026-11-17用户明确的后续完整范围已补入[TODO](../TODO.md)：同层提取／注入50%／75%／100%深度，Slerp／直接加法，两种方式均搜索0.05／0.1／0.3／0.5。本节保留已完成小范围实验的实际设置，完整网格按上节的新身份执行，旧三档保持冻结。

| 轴 | 本轮范围 |
|---|---|
| 方案 | S−T方案5–8：all／static_overlap × global／per_question |
| Slerp alpha | 0.05、0.1、0.2；0.1直接引用已完成ablation，不重复运行 |
| 教师层 | 固定0-based29；层位扫描暂不自动加入 |
| 静态门控 | 固定overlap、目标20%，只在原700题Select校准；原300题Confirm沿用阈值 |
| 数据／SV | 原1000题；SV仍在本题／全1000题池提取，总上下文2048，seed42 |
| 选择 | 每个方案按Select题目等权Δoverlap选择参数；全部参数同时留档Confirm，Confirm不参与排名 |
| 基线 | 原1000题T−S已提交学生无SVbaseline及冻结教师评分，共同基线不变 |
| 执行／停止 | GPU6、7，最多两卡，参数点串行；只做overlap，不执行OPD、退火或独立测试 |

旧扫描定义保存在其冻结search_request.json中；唯一模板parameter_search现定义上节的新网格。模板execution_hold保持，显式运行入口生成各参数独立执行配置。旧RUN_NAME=param，各参数目录使用n1000_slerp<alpha>_S26_T29_seed42_st_only_v1；总表在step6/param对应的st_strength_L29_n1000_seed42_v1目录。取值是本轮补充的实施默认范围，不能称原MD已经规定。

按Select题目等权Δoverlap选择的结果如下，Confirm不参与选择。数值均为相对共同无SV基线的百分点变化；三档表的历史参数身份为`st_strength_L29_n1000_seed42_v1`；下表保留当时的择优记录，原始文件按备份清单追溯。

跨参数comparison.csv同时记录value／delta_pp及token_value／token_delta_pp：前一对按题目平均，后一对按所有有效学生token位置平均。value和token_value均为注入后的绝对指标，取值0–1；delta列为相对共同无SV基线的百分点变化。单参数CSV保留原题目平均列名question_weighted_value／question_weighted_delta_pp和token_weighted_delta_pp，并同样增加token_value。两种指标分别占一行，整体统计包含未注入位置。已有CSV从已冻结summaries读取token平均值补齐，不重新评分或改变排名。

| 方案 | 选定alpha | Select Δoverlap（pp） | Confirm Δoverlap（pp） | Confirm ΔP_overlap（pp） |
|---|---:|---:|---:|---:|
| 5 all／global | .2 | +.7656 | +.7277 | +.6936 |
| 6 all／per_question | .1 | +.4709 | +.4362 | +.4535 |
| 7 static_overlap／global | .2 | +.5682 | +.5426 | +.3217 |
| 8 static_overlap／per_question | .2 | +.4715 | +.4460 | +.2509 |

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_param_search.py --check
"$STEER_OPD_PYTHON" scripts/run_param_search.py --status
```

以上搜索身份已完成并冻结。CPU优化后源码身份已改变，后续新实验须另行定义新搜索／参数身份，不能对本轮原身份再次--run。

数据、师生回答、两侧池化／bank、对齐、原生概率和源指标按科学合同与SHA复用ablation。改变alpha只新增候选评分、漂移诊断与图；单独核验原alpha=.1参考结果，不改名冒充重算。新教师baseline／重放／候选使用无损gzip的.pt.gz，保留原始tensor位模式、dtype及全部诊断字段，StageCheckpoint按提交清单自动读取；不删除原未压缩ablation结果。alpha=.2曾因空间预检停止，2026-11-17用户清理后可用约79 GiB，在原身份下恢复并完成，未重算已完成的.05／.1。

教师S−T的CPU指标按题使用spawn多进程，默认STEER_OPD_METRIC_WORKERS=16，不超过进程CPU亲和性可用核数；设为1可使用串行参考。每个worker仅接收CPU概率及tokenizer字节映射，每worker固定1个Torch线程，同一评分阶段复用进程池，最多提交2×worker数的题目。结果按原题目顺序返回，匹配、概率、NaN、逐span诊断及完整候选提交规则保持。执行进程数和spawn方式记录在events.jsonl的cpu_metric_execution事件，helper源码纳入候选复用身份。原生概率采集、GPU前向及gzip写盘方式保持。

128道真实Select题／447283个有效span的CPU验证：单进程首次53.27秒、重复54.52秒；8进程首次25.50秒、复用池10.49秒；16进程首次43.51秒、复用池4.48秒。所有返回字段及tensor位模式与串行参考一致；计时包含对应批次的进程通信，首次另含进程启动，未测正式全量端到端加速。证据audit/cpu_parallel_benchmark_20261117.json。本优化已在alpha=.2按原冻结代码完整结束后启用；安装后的17项针对性CPU测试及包检查通过。本轮三档结果均使用原代码，后续新实验使用新参数／搜索身份。

Confirm仍使用本题per_question SV；global仍使用全部1000题，所以该划分是内部复核，不是独立未见验证集。参数结果只说明教师分布适配，后续学生学习收益须独立验证。

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

## 历史对比：T−S overlap四组候选

以下为历史学生空间T−S对比的4组固定强度候选，加共同无SV基线，不是后续S−T主线的默认设置。**退火不属于该阶段，不按题目分配退火强度。**

| ID | 方向 | 位置 | 粒度 | 强度 |
|---|---|---|---|---|
| 1 | T−S | all | global | fixed |
| 2 | T−S | all | per_question | fixed |
| 3 | T−S | static_overlap | global | fixed |
| 4 | T−S | static_overlap | per_question | fixed |

固定T=Qwen3.5-35B-A3B、S=Qwen3-4B，S0冻结、0次参数更新。后续注入面板固定Slerp alpha=0.1与学生0-based单层26（36 blocks的75%）。all两组请求全部回答预测位置；static_overlap两组使用源分布预先冻结的位置，候选不重判idx。global/per_question共享相同文本、过滤交集和学生侧尺度。

原四方案方向为T−S：同一S0读取教师回答和自身普通无SV源回答，2048-token上下文内保留的回答原始激活逐题池化后作差；SV提取遵循sv_training的全样本、2048总上下文、保留回答token均值规则；global先平均原始差分再单位化，逐题零方向精确无干预。新增教师空间方案见下一节。当前不使用Ref模型。source_overlap模式不需要这些SV步骤。

选择候选子集时，显式设置`STOP_AFTER=panel`、`SCHEME_IDS=1,3`等子集及新PARAM_NAME；原四组配置为confirm_top_n=4，两部分都比较完整四方案；本轮仅新增下述教师镜像对照，不自动扩展其他方向、层或强度网格。源缓存是否可复用，须按checkpoint、前缀、原生概率及代码身份独立校验，不能改旧run配置后直接续接。

## 历史已完成：教师空间S−T与原T−S镜像对照

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


## OPD训练路线与执行记录

默认traditional OPD，可选CompassOPD；方法公式见[研究方案§9.4](Steering_OPD_完整研究方案.md#94-后续训练loss默认传统opd可选steercompassopd)，当前共同参数、两卡分时布局、每step生成／评分／overlap／更新／评测和操作命令统一见[代码框架逻辑](代码框架逻辑.md)。此处仅记录执行范围与历史身份。

用户已授权并执行过`verl_traditional_opd_st_T29_slerp0.1_seed42_shared2_formal_v5_seed42to45`：初始Qwen3-4B、传统OPD、教师S−T／global／all、层29／Slerp.1、两张空闲卡共享。该次执行采用每题四个seed42／43／44／45，跨题／批次复用；不按回答正确性筛选，不开启baseline比较、退火或其他超参数训练。种子策略与重跑边界见[重跑审计](../audit/training_rollout_seed_slots_restart_20261117.json)。

该次训练于2026-11-18按用户要求停止，已完成68次更新，停止时最近完整checkpoint为step60，见[停止记录](../audit/opd_user_stop_20261118.json)。v3重复采样标为invalid_sampling，v4整批递增seed被替换且0次更新；这些来源不能静默接续为有效新训练。当前生成默认和概率来源已在后续代码中调整，旧formal_v5实测不冒充新配置结果；本次文档整理不恢复训练，模板hold保持。

训练内accuracy使用独立带gold的MATH-500／AIME2025，Confirm不作独立测试；与1000题SV池仅核对过完全相同题目，未声称语义去重。每步overlap的监控开关、更新前口径、TensorBoard和原文／资源日志只在[代码框架§3](代码框架逻辑.md#3-opd训练内部逐阶段拆开)及[§8.4](代码框架逻辑.md#84-日志tensorboard与恢复排查)维护。耗时与停止前预算见[运行时间](目前代码估计运行时间.md)。
