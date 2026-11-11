# 执行、验收与当前待办

更新：2026-11-11。**用户要求优先准备Rethinking Figure 8 + steer，数据范围待澄清。**
下文跨family和历史排程仍暂停，不自动续跑Phi/Smol或补其权重。
方法定义见[完整方案](01_research_plan.md)，论文设置见[证据表](02_paper_settings.md)。

## 最新任务：Figure 8原设置与steer比较

最新任务先于下文C1/P0–P5。待定配置见
[Figure 8配置](../experiments/rethinking_figure8_20261111.json)，证据见
[来源审计](../audit/rethinking_figure8_2026-11-11/source_audit.json)。

- 固定教师Qwen3-4B Non-thinking；三条主曲线为Base→OPD、作者SFT→OPD、
  Base+steer→OPD。真正Base及作者SFT仍缺失，普通1.7B不能替代。
- AIME2025共30题。已提问澄清仅用于评测，还是SFT/校准/OPD也全部用它；
  答复前不执行依赖该选择的下载与GPU任务，不自行把评测题分配给训练。
- 原设置为200K OpenThoughts3教师示范和另30K OPD题。可核验复用作者SFT初始化；
  若改为AIME训练，必须标为小数据改版，不能声称原配方完全一致。
- 论文表2和原fork：batch64、mini-batch64、rollout n4、Student Top-K16、
  student_p权重、训练temperature1/top_p1、prompt1024/response7168、LR1e-6、
  1 epoch、无额外KL、token-mean。源码shuffle=False、save_freq20。
- 官方离线评测：avg@16、temperature0.7/top_p0.95、response31744、non-thinking；
  保留原生停止token。源码关闭verl内置验证，单独评估checkpoint。
- 原图展示0–200步，源码总训练为1 epoch；图示窗口与总步数分别记录。
- Overlap为普通学生前缀上的Top16交集/16，在有效回答位置上按token汇总。
  原fork主指标来自训练轨迹；不能静默改为校准文本或带hook模型的overlap。
- 原fork采用Student Top-K、student_p加权的冻结reward及PPO surrogate。现有R1
  launcher的teacher-top-K forward KL tail目标不同，不能只改模型名复用。
  原版基线先验收，加hook后的surrogate语义单独核对；alpha0必须恢复相同loss/梯度。
  不将普通rollout称为steered policy采样，也不静默改成另一种candidate KL。
- steer校准数据、层、强度、辅助时长是新增项，需要预先记录；不据AIME测试曲线
  调参。原forkremove-padding路径的预测mask、packing边界与重算仍待实际验收。

当前只完成来源及配置准备，未下载Base/SFT权重，未启动训练或生成结果曲线。

## 1. 当前事实

- R1/JustRL工程链路已有无hook、预测mask、梯度checkpoint重算及短程更新验证。
  `runs/pilot_20261110_01`的E0/E3各100步完成，未据此形成训练后效果结论。
- Phi/Smol探索队列已自然结束，历史基线、向量和诊断保留在`runs/`与`audit/`，
  不再作为正式主配对。旧3072上限教师dev高截断，不用于冻结正式选型。
- C1/C3尚有模型/初始化缺口；异构候选训练、Compass原版及方案2没有正式实现验收。
  旧共享ID KL启动器不是这些实验的现成入口。
- 用户停止核查证据在
  [stop记录](../runs/cross_family_steering_probe_20261110_02/user_stop_20261110.json)。

## 2. 环境、源码与产物边界

从项目根`/home/zhenning/data/working_tmp/steer_opd`操作，等价实际路径为
`/local_nvme/data/zhenning/working_tmp/steer_opd`。模型/数据共享目录只读；所有新
checkpoint、日志、缓存、checkout与处理数据留在本项目，不改邻近项目。

已有可用工程环境为项目`.venv-opd`，`scripts/runtime_env.sh`设置本地缓存和路径。
已验证组合：torch2.13.0+cu130、Transformers5.16.1、vLLM0.32.0、verl0.10.0.dev0、
math-verify0.9.0。`src/steer_opd/runtime.py`逐阶段校验版本；共享Conda环境曾漂移，
不能因为`verl`环境存在就覆盖安装或认为新架构可训练。

下载工具支持共享模型目录和项目软链接；当前仍不执行下载或移动已有权重。

| 源码 | 固定commit | 当前用途 |
|---|---|---|
| verl主干 | b17218ba58a6fbcdfb0de32772e3c2a79f88529d | 项目vendor中已有；原生KL与actor后端扩展 |
| Rethinking-OPD | ac26e38d6f1572eb027597b48a9f4e01f6915ef8 | 已有源码/数据快照；原Student Top-K配方对照 |
| Simple-OPD | 48234d0b9cabf9d0e01a4e8f0e5b965ab751cde6 | 已核对固定关键源码；warm-up/merge参考，不混装整个fork |
| sv_training | 198fdbd917e9d7234f734aa8eb754faa408bd2e6 | 激活/方向/hook参考 |

证据入口：[源码快照清单](../audit/experiment_planning_2026-11-07/code_sources.json)。
Compass已核对公式及附录，尚无已验证的官方可运行代码入口，原式实现应标清来源。
Qwen3.5是新架构，不能默认由旧AutoModelForCausalLM和相同`model.model.layers`
访问方式加载；Granite也需核对真实block、输出head、缓存与梯度接口。

## 3. 恢复后按顺序验收

### P0：资产、数据与论文对应

确认C1/C2/C3的角色、repo ID、revision、模式、总/激活参数和初始checkpoint。
Granite与Qwen3.5-4B只在确定清单获准后补缺。核验本地DeepScaler原题集合、
OpenThoughts-114k与OT3区分、去重/污染和dev边界；保留原数据只读。
teacher/reference/student/template/stop token实际样例归档，不能靠模型名判断兼容。

### P1：无干预诊断与概率接口

先固定普通学生前缀，在原生shared-vocab控制上恢复Overlap@16定义，再验证异构
共同文本span。单独验证候选接口：一对一、一对多、多对一、空白、中文UTF-8、
EOS、重复解码、prefix重分词、无法对齐及低覆盖。教师log-prob仅计候选后缀，
不能把prompt似然或未来token泄漏进候选。T/R训练评分temperature1，生成解码
temperature另记。sum和mean用小例子分清，不互换名称。

先完成无辅助Candidate-RKL训练，确认参数和梯度真实变化、目标有限，并检查dev响应。
用可控小例子验证目标数值/梯度，不要求真实难迁移配对的Vanilla先有准确率增益才
允许研究辅助效果。候选归一化只约束保留集内相对概率，不约束其总质量；持续质量下降时先修
接口，不能只凭条件KL变低宣布学会。原版span采样基线也单独验证。

### P2：向量和hook

目标模块：学生原生block输出，不能用最后norm后的hidden冒充指定中间block。
只缓存所需层和池化位置，记录内容mask、采样位置和每题权重。对于方案2，用同
三文本协议确认`v_total_triplet = v_cap + v_offset`；不向teacher或reference挂hook。

必须通过：alpha0输出/梯度一致，向量宽度正确，首回答预测位置t−1，batch/packing
不越界；hook贯穿backward重算；撤去未更新模型的hook恢复原输出；真实新模型上
至少一个非零方向forward/backward成功。已有R1检查不能替代Granite/Qwen3.5检查。

### P3：短程学习与选择

先用相同训练prompt顺序和seed、各自实时生成的轨迹比较E0/真实/随机；模型更新
后轨迹自然分化，不能强行共享旧rollout后仍称实时OPD。最多6个真实候选×10步，
有信号者到50步；每次关闭hook评dev-select。选定后dev-confirm和移除后普通
更新曲线确认，避免只选带hook overlap最高。候选预算和随机种子成本计入。
正向候选另做同层同强度负方向的短程学习控制，不能只看其即时诊断。

### P4–P5：强基线与正式结果

面板A原始初始化：full SFT、LoRA、SV-SFT、方案1/2与Vanilla；面板B共同SFT：
Vanilla/Compass及方向组合。主方法面板共用相同Candidate目标；原版span基线
另表。先看C1在关掉方向后是否有可靠收益；有依据才增加种子、方案2、C2/C3或多方向。
不能把所有计划行当作必须依次执行的固定队列。

### 跨family效果不好时：回到Rethinking设置

先排查分词对齐、长度、训练更新和基线是否正常；在约定预算内仍没有额外收益，
将问题缩小到共享词表下的OPD学习困难。优先F1：Qwen3-4B non-thinking→
Qwen3-1.7B-Base；另可用Rethinking的JustRL/R1-7B→R1-1.5B成败对照。
F1的Base模型和数据缺口仍须核实，不能自动以现有Instruct替代或启动下载。

回退先复核原论文无方向OPD和SFT基线，再比较方向、随机和负方向。共享词表时
可以直接在共同top-K词元上计算显式loss，不再为主方法强加异构span接口、弱参考
或多学生矩阵；原论文采样目标与显式方向消融目标的差异仍要标清。方案2不是回退必需项。

只有这一设置有收益时，结论就限定为共享词表下的训练辅助/冷启动效果。跨family
负结果保留，路线选择不使用MATH-500。若回退也没有收益，停止扩大搜索并记录负结果。

## 4. 建议的配置起点与限制

| 项目 | 工程/筛选起点 | 正式起点或对应要求 |
|---|---|---|
| 学生batch / rollout n | 8 / 1，微批1，梯度累积 | 16 / 1作为项目预算；原论文复现保留其batch64/128 |
| 学生生成上限 | 8192；短前缀工程检查另标，不报告其正式准确率 | 按模式/论文及实测截断统一审定，不因显存静默缩短 |
| OPD LR | 1e-6，先零warmup | 所有比较组共同调参预算 |
| Candidate K / 位置 | 8 / 4固定分位位置 | 通过后16 / 16，score mean主分析、sum稳健性 |
| 方向校准题数 | 128，同信息示范 | 32/128/512成本曲线 |
| 方向层 / rho | 中层 / 0.01、0.03、0.1诊断 | 依据真实学习选择，层扫描后置 |
| 辅助步数H / 总更新 | 筛选前10步，确认50步 | 300总更新中前40辅助，H20/40/80是后续选择，不与论文最佳值等同 |
| full SFT / LoRA LR | 5e-6 / 5e-5，LoRA rank32 | 搜索rank16/32/64与饱和时长，实际merge |
| Compass beta | 0.5，与direction alpha分开命名 | 记录原版或candidate-adapted |
| 训练seed | 42筛选 | 42/43/44正式，另有3个方向随机种子 |

上述均未启动。新结构峰值显存、每候选评分成本、长prefill吞吐与有效batch须在
恢复后实测，不根据BF16权重大小承诺训练可放入单卡。正式预算应同时给出更新数、
GPU-hours、教师评分token与评测费用，而非只锁定一个step数。

## 5. 测试、资源与结果记录

允许的CPU材料检查：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/validate_package.py
```

已有语义测试位于`tests/test_training_semantics.py`和`tests/test_cross_family_overlap.py`。
新的loss/边界/hook检查必须对应实际错误风险，不写只复述配置的测试。

GPU任务经`run_on_idle_gpus.py`或等价的新鲜nvidia-smi显存/利用率/进程检查入场。
无空闲卡等待；禁止pkill/killall、全局ray stop及终止外部项目。只在用户授权停止
本项目时，验证PID/cwd/完整命令后终止。日志、进程记录和退出码为完成证据。

EasySteer是可选capture加速；先核对HF逐样本池化、方向余弦/范数与开关输出，
再计时。训练主干梯度/重算仍由HF/verl负责，不能用推理加速宣称OPD训练加速。
已有定点源码依据见[EasySteer来源清单](../audit/math500_launch_2026-11-08/easysteer_sources.json)，
不因清理旧启动方案丢掉已读取的证据。

每次运行填写[记录模板](../experiments/run_record_template.md)，主组与阶段以
[实验矩阵](../experiments/experiment_matrix.csv)、[阶段表](../experiments/stage_matrix.csv)
为准。它们是当前计划，不代表启动器已支持或实验已经完成。

## 6. 当前TODO

- [x] 收敛文档，删除重复探索稿，保留原论文及研究证据。
- [x] 完整定义方案1、方案2、分词/目标边界、4B级对照与强基线。
- [x] 将“同教师、多学生”降为可选对照，增加按结果回退Rethinking的路线。
- [ ] 讨论并冻结最小面板、完整SFT或小预算面板、开销上限和主要终点。
- [ ] 落实缺失模型的精确发布映射/初始化与数据等价性；当前只核查，不下载。
- [ ] 用户明确恢复后才做资源预检与P0–P5；旧队列不能自动续跑。

## 附录：实现时再看这些技术细节

下面保留方向、mask、分词和训练目标的精确定义，供实现与复核使用。
是否进入跨family路线或回退路线，以总方案的结果判断为准；这里列出某个目标，
不代表必须同时运行全部方法。共享词表回退不需要异构span接口或弱参考。

### 5. 方案1：教师—学生总差方向

#### 5.1 方向定义

记`Phi_j,l(x,y)`为冻结初始学生`S0_j`在block l读取回答y后的内容激活池化。
每题先池化，跨题等权平均：

`v_total[j,l] = mean_i(Phi_j,l(x_i,y_T_i) - Phi_j,l(x_i,y_Sj_i))`。

不是教师激活减学生激活；两项维度和坐标都来自同一个学生。方向固定于S0，不
随训练刷新，刷新另作消融。若共享SFT初始化改变，必须从新的S0重新提取。

主池化规则：只取回答内容，排除prompt、padding、role/EOS及结构标记；每条完整
文本一次prefill。在每对文本中取相同数量、最多512个内容状态，按各自回答的
相对位置均匀抽取后平均。再比较完整内容均值与等长前512 token池化，检验长度/
位置影响。不同长回答的prefix上下文仍可能不同，等数采样不等于完全消除长度混杂。

第一轮只用中间block，0-based层号按模型真实层数记录；之后仅在有信号时比较
约1/3、1/2、2/3相对层深，不在全模型逐层扫到一个好结果。

#### 5.2 注入与作用域

对选中block输出，在回答**预测状态**加`alpha * v`。若回答token位于输入位置t，
预测它的是状态t−1，首个回答预测状态包含prompt的最后一个位置。

`alpha = rho * median_calibration(||h_l||) / ||v||`，诊断rho取0.01/0.03/0.1。
记录目标扰动比与BF16舍入后的实际扰动；向量近零时不强行归一化。随机向量与真
方向匹配层、范数、mask、强度和启用时长，至少3个随机方向种子。

**学生rollout始终无hook。** 方向只在早期loss前向启用并维持到backward/gradient
checkpoint重算结束。教师、弱参考、学生锚点、常规诊断和最终评测都关闭hook。
仅前H次更新带方向，随后移除并继续普通蒸馏；保存H、H+1、H+20及终点。

### 6. 参考Compass处理分词与评分

#### 6.1 原论文的共同文本span

普通学生生成文本，T和R分别用自己的tokenizer重新编码，按共同文本边界单调
对齐。一个span可以对应双方不同数量的token。一对一直接用token log-prob；
多token span按Compass A.4取平均token log-prob，未对齐位置mask并报告覆盖率。

例如学生的`unbelievable`为1 token、教师为`un / believ / able`三token：
教师span评分为`ell_T = (log p1 + log p2 + log p3) / 3`，每个p条件于此前文本。
教师家族增量`Delta_T = ell_T - ell_R`分配给span内学生token，学生锚点变化在
自己的token位置计算。这是平均似然评分；指定token路径的整段文本log-prob应取
和，不能将平均值称为整词概率或精确跨词表KL。

原版Compass采样优势为`sg[Delta_T - beta*(log pi_theta - log pi_0)]`，论文beta=0.5。
T/R及pi_0冻结，pi_0是共同SFT后、OPD开始时的学生。原版无hook基线按此复核。

#### 6.2 为什么不能直接给原采样公式加hook

轨迹来自普通学生，带hook模型的条件分布不同。将普通学生采到的token直接当作
steered policy样本，套用未修正的reverse-KL策略梯度，不是正确的估计。
仅detach advantage或加入PPO旧策略比率也不自动消除这项分布差异。

因此方法消融第一版采用下一节的**显式有限候选分布目标**，在普通学生前缀上枚举
候选，直接求可微KL，不对被采样单个动作套用错误估计。原论文采样基线另列。
这是一个明确的接口适配，不能声称原版Compass加hook无需任何目标变化。

#### 6.3 方法面板的可执行数学目标

在本次更新的无hook学生前缀c上，取普通学生top-K个next-token候选，将其解码为
当前上下文之后的可见续写文本。原始student token IDs保留；候选在该次更新内
固定，不对top-K选择求导。教师/参考在各自native prompt后读取相同可见前缀。

首版只保留：有效UTF-8、非控制角色、可稳定拼接，并且对T/R有稳定编码前缀边界
的候选；至少2个候选才计算loss。重复解码文本合并并对学生概率求和；EOS是单独
终止事件，概率按本模型的合法停止token聚合。不能用strip抹掉空格，不能把token
ID相等视为异构模型语义相等。原前缀也必须完整encode/decode round-trip。

每个候选可由教师多个token表示，按同样的`sum/mean`显式配置取其条件log-likelihood。
主方法面板使用Compass式mean，sum作稳健性消融；所有方法共用同一评分口径。
如果编码跨越prefix边界，首版跳过并统计；扩大为更长公共span属于后续接口版本，
不能默默改写学生采样历史。

记`b_j^v`为带hook学生对保留候选j的原始概率，`p_j^v = b_j^v / sum_k b_k^v`；
无hook时v=0。记教师候选评分为`ell_T,j`、弱参考为`ell_R,j`，冻结学生在同一候选
集上的归一化概率为`p_0,j`。所有目标都在本次固定的有限候选集C上定义：

`q_V,j = softmax_j(ell_T,j)`；

`q_C,j = softmax_j(log p_0,j + (ell_T,j - ell_R,j) / beta)`；

`L_V = sum_j p_j^v * (log p_j^v - log q_V,j)`；

`L_C = beta * sum_j p_j^v * (log p_j^v - log q_C,j)`。

`q_C`来自Compass式(11)的有限候选目标，beta起点0.5。T、R、p_0、候选集及方向
均停止梯度；只有学生theta更新。每步普通学生重新rollout；固定前缀诊断缓存不
拿来代替整个训练的实时轨迹。关闭方向后使用同一目标继续训练。

这一定义解决了本方法实验的采样/前向不一致：在固定访问状态上直接枚举候选并
求导，不需要假装动作来自带hook策略。它是有限候选reverse KL，不是完整词表/
完整文本分布KL，也不包含对离散rollout状态采样过程求导。

不同候选文本可能互为前缀，教师mean分数也不是互斥事件概率。因此`q_V/q_C`是
明确定义的归一化偏好目标，不把教师候选分数之和称为覆盖率。共享词表、一token
且完整保留原生事件时可对应Student Top-K归一化目标；跨tokenizer不能直接如此宣称。

#### 6.4 原版与适配版分开比较

所有方向消融共用`L_V`或`L_C`、候选和位置规则，以隔离方向贡献。报告名称为
Vanilla-Candidate / Compass-Candidate adapted。另保留原版无hook的Vanilla-span /
Compass-span采样目标，复核论文趋势并作为成本匹配强基线。
若只有candidate面板成立，主张限定在这个接口；不能据此声称已改进原版所有OPD。

首轮工程目标K=8、每回答4个固定分位位置；通过后比较K=16、16个位置。所有方法
一致；记录位置覆盖、有效候选数、学生保留质量、丢弃原因和每候选教师token数。
这部分教师评分可能很贵，必须独立计时，不能把“一次学生prefill”当整个训练成本。

### 7. 方案2：教师家族增量方向

#### 7.1 从Compass的分解得到一个受限表示假设

Compass在相同动作上的log-likelihood分解为“弱参考相对学生的偏移”加“强教师相对
弱参考的变化”。我们检验这个思路能否形成学生空间的更有用方向。

在相同校准题上取得T、R、S0三种回答，三者均由同一个S0读取。为保证下面分解的
代数意义，**三者共享同一个池化协议和共同样本权重**：每题内容采样数取三者长度
的最小值、最多512；同一来源的Phi在不同方向中不可重新换长度或mask。

这里的三种回答分别由大教师T、小参考R、初始学生S0生成。单独方案2只需要T/R；
下面的三文本协议用于与方案1和“弱参考−学生”方向进行公平比较。

`v_cap = mean_i(Phi(x_i,y_T_i) - Phi(x_i,y_R_i))`；

`v_offset = mean_i(Phi(x_i,y_R_i) - Phi(x_i,y_S_i))`；

`v_total_triplet = v_cap + v_offset`。

真实比较使用这个三文本协议重新计算的`v_total_triplet`，不能把方案1另一池化
规则所得向量直接带入并宣称严格分解。若额外加权，所有项用相同权重才保留恒等式。

#### 7.2 需要验证的内容

主候选是`v_cap`：尝试保留教师同家族内的增量，减少把弱参考也有的跨family偏移
全部传给学生。`v_offset`单独注入是关键反证对照；还要比较总差、随机与负方向。

名称cap只表示设计意图。T与R的自由生成回答在内容、长度、正确性和风格上都可能
不同，表示差不自动是纯能力方向；同family也不保证风格完全相同。这不是Compass
log-likelihood恒等式在隐藏空间中的证明。应加入格式/长度匹配、正确性分组，
必要时让T/R从同一个校准前缀继续生成，检验内容差异是否解释结果。

单独方案2需要T/R文本及两次学生prefill；完整三方向比较读取T/R/S三类文本。
R对训练轨迹的概率评分只在使用Compass目标时增加。教师端的log-ratio是否改善
监督，与学生方向是否改善计算是两个因素，必须分开消融：

| 学生方向 | 普通教师候选目标L_V | Compass候选目标L_C |
|---|---|---|
| 无 | E0 | C0 |
| 总差方向 | E3：方案1 | C1_TOTAL |
| 教师家族增量方向 | E5_CAP：方案2 | C2_CAP |
| 同范数随机方向 | E4 | C3_RANDOM |

相同信息面板中，即使某组不使用R，它也可以使用同一批已生成文本做合理SFT/LoRA
候选；另保留严格只用T文本的标准baseline。分别报告固定信息与实际各方法所需
信息的成本，不能通过独占额外R示范把收益全部归给方向构造。

若C2_CAP最好，也要先确认它优于C0、C1_TOTAL及C3_RANDOM，才能将收益归于方向
构造和组合价值。不能只比较“方案2+Compass”与“方案1+Vanilla”。

