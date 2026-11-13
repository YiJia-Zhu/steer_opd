# 新颖度扩大调查：跨 family OPD × Steering

2026-11-12适用范围提示：下文保留2026-11-07的检索与当时“普通rollout、仅早期loss
前向注入”草案，属于历史审计，不是新版方法定义。本次按用户[最新完整稿](Steering_OPD_完整研究方案.md)
修改的主候选为T−R＋overlap单阈值、生成/更新一致的受控学生；熵差独立对照，
方向/策略三乘三矩阵及无干预学生结果见[当前主计划](01_research_plan.md)。
本轮没有重新执行新颖性检索，不能把旧“未核实到完全相同流程”转写成新方案已具新颖性。
后续需针对教师同家族差分、在线位置门控、受控rollout/重放与已有自蒸馏/表示方法
重新比较；原有近邻、来源、阅读深度及否定性结论均保留，不因方案更换删除。

检索窗口截至：2026-11-07（UTC；北京时间整理跨至11月8日）。本轮用于检查 prior art，不改变实验配方，不启动下载模型或训练。论文链接固定到本次核对的版本；已保存检索记录、官方正文和阅读深度。

## 1. 先给判断

**两个大方向都已有明确工作；目前没有核实到一篇完整复现我们当前全部流程的论文，但这不能等同于创新性已经足够。**

| 待判断的主张 | 本轮结论 |
|---|---|
| 首次解决跨 family 的 OPD | 不成立。CompassOPD、OPRD-Bridge，以及一系列跨 tokenizer OPD 已直接研究 |
| 首次把 activation steering 用于 OPD | 不成立。ACSD、ST-OPSD、SafeSteer 已直接覆盖；DIAL 还包含学生侧 steering |
| 首次用低维方向/子空间辅助 OPD | 不成立。OPRD-Bridge、SCOPE-OPSD、PreSO 等已覆盖相关做法 |
| 首次在早期加内部辅助、随后撤去并继续 OPD | 不成立。**LastOPD、Latent-MOPD** 是本轮最需要补充的先例 |
| 普通学生生成，只在训练前向做干预 | 这个结构本身也已有近邻，例如 Trajectory Dropout；它干预训练上下文，未加固定方向 |
| 教师文本由学生读取，差向量注入学生训练前向 | 主参考 **Steering Vectors as a Training Signal** 已覆盖监督蒸馏版本 |
| 上述文本方向只注入早期学生 loss 前向，以改善真正跨 family 的 OPD，撤去后仍有收益 | 本轮未核实到全部条件同时相同的论文；这是仍可检验的较窄组合，效果与独立价值尚未建立 |

**新增加的风险比单纯“还有几篇 steering 论文”更重要：LastOPD 已覆盖早期内部辅助的时间安排；Latent-MOPD 将该安排用于其称为 cross-family 的设置；OPRD-Bridge 已有真正跨 tokenizer 的低秩内部迁移。** 只说“更相容、低维、冷启动、部署移除”，区分度不够。

## 2. 怎么判定“和我们一样”

当前方案固定为：

1. 使用固定外部教师，研究其监督为何难以被某个学生吸收，目标包含跨 family OPD。
2. 初始学生 S0 读取教师回答和自身回答，在**同一学生空间**构造 `v = mean[Phi_S0(x,y_T) - Phi_S0(x,y_S)]`；不相减两个模型的 hidden state。
3. 普通学生生成新 rollout；只在学生计算蒸馏 loss 的前向注入方向，更新学生参数。
4. 辅助仅持续早期阶段，随后撤去并继续普通 OPD；主评测与部署均无 hook。
5. 方向的选择最终依据关闭 hook 后的学生学习收益，而不仅是即时 overlap。

调查同时区分三个问题层次：

- **接口可比：**不同 tokenizer 怎样比较概率/事件。
- **监督可用：**即使已经可比，教师偏好是否给学生正确、有用的更新。
- **计算可用：**内部信息是否能被学生后续层使用，并真正改善参数学习。

我们最初强调的是后两层。第一层已有工作的存在不能证明相容性已全部解决；同样，也不能继续将“让 OPD 能跨 family”作为空白。

这里严格的 steering 指修改模型前向中的激活，例如加方向、投影或可学习残差控制。**输出 logits 重构、表示辅助 loss、参数方向**另列；它们不是相同实现，但可能构成很强的概念先例。不能靠改名来忽略它们。

## 3. 第一条线：谁已经解决跨 family / 跨 tokenizer OPD？

下表的“解决”指论文提出并实验研究该问题，不表示所有模型组合已经得到解决。

| 论文 | 具体解决的问题 | 方案 | 与我们关系及证据边界 |
|---|---|---|---|
| [CompassOPD，2609.10154v1](https://arxiv.org/abs/2609.10154v1) | 已做 tokenizer 对齐和 SFT，跨 family OPD 仍不能充分利用强教师 | 传递教师相对同家族弱参考的似然增量，并用初始学生锚定 | **问题最直接重合。** 改监督目标，不是学生固定方向干预；不能说只有我们处理相容性 |
| [OPRD / OPRD-Bridge，2606.06021v4](https://arxiv.org/abs/2606.06021v4) | 师生 hidden space 不可比，输出接口又受词表限制 | 冻结教师 PCA 低秩投影，拟合学生投影；冻结桥后在学生 rollout 上做表示蒸馏 | **问题与低维内部迁移都很近。** 有 Phi-4-mini-reasoning→Qwen3-1.7B-Base 的跨 tokenizer 结果；不是向学生前向加入固定文本差向量。该例 Avg@16 从约2.8到5.0，不能概括成迁移瓶颈已彻底解决 |
| [Breaking the Tokenizer Barrier，2606.09456v1](https://arxiv.org/abs/2606.09456v1) | 标准 OPD 无法直接比较异构 token | 共同文本边界上的 chunk 对齐与信用分配 | 已直接研究 across model families；主要处理概率接口，不用内部方向修复推理模式 |
| [SimCT，2605.07711v3](https://arxiv.org/abs/2605.07711v3) | 严格共享 token 匹配丢掉监督 | 增加双方都能表达的短多 token continuation，建立共同监督单位 | 已覆盖“在文本续写候选上比较异构模型”；我们若用此接口，应作为复用组件 |
| [Byte-Prefix Marginalization，2607.22334v1](https://arxiv.org/abs/2607.22334v1) | 异构词表映射丢概率质量或映到无关内容 | 在字节空间将教师质量映到学生前缀，并显式保留 residual 类 | 已直接做跨 family OPD；不是激活 steering。其精确性有边界条件，不能简称任意上下文的完整文本 KL |
| [ESCD，2609.34738v1](https://arxiv.org/abs/2609.34738v1) | 学生只生成了教师 token 的部分字节时，后续完成事件缺少监督 | 监督 byte-compatible completion set 的总概率 | 在已有对齐之后处理另一类监督缺失；不是学生内部相容性方向 |
| [CAST，2610.13257v1](https://arxiv.org/abs/2610.13257v1) | chunk 对齐后，怎样将优势分给学生 token；未对齐位置如何约束 | 按不确定性与局部教师信号平滑度分配 chunk credit，并锚定未对齐位置 | 有 Qwen3-8B→Mistral/Llama 的跨 tokenizer 结果；说明“接口已通、学习信号还需改”也已有工作 |
| [Rethinking Cross-Tokenizer OPD，2610.08448v1](https://arxiv.org/abs/2610.08448v1) | 覆盖更多对齐位置是否一定改善学习 | 比较 strict shared-vocabulary/top16 与 span-MSE；分析梯度方向冲突 | **对问题表述很重要。** 已将关注点从覆盖率转向监督可靠性；不加 steering，所测 span 负结果也不证明所有 span 方法无效 |
| [DVLM，2609.33816v1](https://arxiv.org/abs/2609.33816v1) | 同时解决输入分词与输出词表不一致 | 冻结教师主干，学习学生词表的输出 head，并使用 PTS/HPA | 是轻量教师接口适配；不是固定学生激活方向，也不是完全无需额外训练 |
| [Bridged Self-Distillation，2610.26397v2](https://arxiv.org/abs/2610.26397v2) | 外部/黑盒教师无法直接给学生词表上的密集监督 | 外部教师提供文本，再由学生家族的特权自教师桥接指导，结合结果验证 | 跨 tokenizer/API 教师的另一条路线；桥接在文本与学生家族教师侧，不是我们的 loss 前向方向 |
| [Can Native Safety Travel?，2610.27555v1](https://arxiv.org/abs/2610.27555v1) | 将安全预训练知识跨架构/规模复用，保留学生一般能力 | 跨 tokenizer OPD；例如小型安全教师向更大的 Llama 学生迁移 | 是跨 family OPD 的任务实例；该教师通过安全预训练得到，不能与下文 activation-steered SafeSteer 混为同一方案 |

以上第一篇最贴近“跨 family 后仍教不好”；OPRD-Bridge 最贴近“低维内部信息能否跨模型使用”；中间几篇大多处理接口和监督可靠性。对我们而言，这些层次都要承认，但不能混称同一个瓶颈。

## 4. 第二条线：谁已经把 steering 与 OPD 结合？

### 4.1 直接修改激活或连续控制的工作

| 论文 | 解决什么问题 | 干预及训练流程 | 与我们的区别 |
|---|---|---|---|
| [ACSD，2609.38342v1](https://arxiv.org/abs/2609.38342v1) | 不依赖每题参考解答，构造有效的自蒸馏指导 | 从预算内正确/其余轨迹构造固定方向，注入冻结自教师；普通学生用自己的 rollout 做 OPD | 已直接覆盖固定方向＋OPD＋学生部署移除辅助；方向放教师侧，不是学生 loss 前向，也非外部跨 family 教师适配 |
| [ST-OPSD，2610.34005v1](https://arxiv.org/abs/2610.34005v1) | 参考条件化教师出现引用和过少检查，污染学生推理 | 教师减引用方向、加重新检查方向，普通学生用 JSD 蒸馏 | 已通过激活干预改善监督；处理特权自教师行为，不是外部异构教师与学生的启动适配 |
| [SafeSteer，2606.02530v2](https://arxiv.org/abs/2606.02530v2) | 安全蒸馏的全局更新损害一般能力 | 拒绝方向构造 steered teacher，普通学生 on-policy 学局部词表项 | 明确的 activation steering＋OPD；主要是同底座安全自蒸馏。选择的是词表项，不能误称回答中少数时间位置 |
| [DIAL，2610.31438v1](https://arxiv.org/abs/2610.31438v1) | 可连续调节、随状态自适应的指令遵循 | 冻结学生主干，学习逐位置 steering adapter，带 adapter 学生生成并用 OPD 匹配 CFG 教师 | **学生侧 steering＋OPD 也已有。** 主干冻结、adapter 训练和部署均保留；目标不是撤去辅助后改善普通学生 |
| [Latent On-Policy Self-Distillation，2608.13040v1](https://arxiv.org/abs/2608.13040v1) | 手工特权上下文限制经验复用；怎样学到有效教师上下文 | 将经验压缩成可学习 latent context，条件化冻结自教师，在普通学生轨迹上联合优化 | 连续潜变量构造 OPD 教师的近邻；属于 latent-token conditioning，不是固定残差方向；不是外部跨 family 迁移 |
| [OPMD，2610.14022v1](https://arxiv.org/abs/2610.14022v1) | 教师内部信息即使数值对齐，也可能无法被学生后续计算使用 | 学 projector 把教师状态残差注入冻结学生，以输出行为优化；再由普通学生 on-policy 内化 | 广义“学生内部干预＋on-policy 蒸馏”非常接近。主内化实验用表示目标；异构 tokenizer/跨 family 扩展的已报结果是 SFT，不能据此认定整套跨 family OPD＋steering 已等同我们 |

### 4.2 输出空间 steering 与方向/表示监督

| 论文 | 做法与重合 | 为什么不是当前全部流程 |
|---|---|---|
| [OGLS-SD，2605.12400v2](https://arxiv.org/abs/2605.12400v2) | 正/负结果条件下的教师 logits 差用于修正目标，再做 OPSD；明确讨论师生推理模板/模式失配 | **“steering 修正模式不相容”这个动机也已有。** 操作在 logits，主体是特权自教师，不是外部教师指导学生固定激活方向 |
| [SCOPE-OPSD，2609.12579v1](https://arxiv.org/abs/2609.12579v1) | 把师生表示残差投影到离线冻结的 rank64 子空间，作为 OPSD 的辅助 loss；匹配随机谱和梯度强度作控制 | 已覆盖固定低秩方向辅助 OPSD；直接监督表示，不修改学生输出 loss 前向，不处理异构 family |
| [PreSO，2610.14001v1](https://arxiv.org/abs/2610.14001v1) | 从 hint 引起的表示变化估计低秩子空间，按题目内容路由，将投影纠正加到学生表示上形成训练目标 | 已覆盖少方向、跨题复用、按内容选方向与学生空间目标；是表示目标构造，不是当前训练前向注入；主体为同模型 OPSD |
| [RIDE，2609.36484v2](https://arxiv.org/abs/2609.36484v2) | 在学生 rollout 上计算 RL 教师与其 pre-RL checkpoint 的逐层残差，沿该方向外推表示目标 | 已覆盖“教师增量方向”与 on-policy 内化；方向逐前缀动态计算，并需共享表示空间，不是少量文本校准所得固定方向 |
| [PHF，2606.29340v1](https://arxiv.org/abs/2606.29340v1) | OPSD 额外匹配教师隐藏状态的 token 间转移方向与几何 | 方向几何监督已有；不等于向学生前向注入一个 steering vector |

因此，严格的 activation steering 先例与广义方向监督先例都存在。我们不能只在前一类里寻找完全同名方法，再忽略后一类对贡献的覆盖。

## 5. 本轮最需要重视的重合点

### 5.1 LastOPD：早期内部辅助→撤去→普通 OPD 已有直接先例

[LastOPD: Taming Collapse in Latent On-Policy Distillation，2609.28845v1](https://arxiv.org/abs/2609.28845v1)，重点 §4–§6。

它观察到表示蒸馏早期有收益，持续匹配则可能崩溃。方案是在最后一层加表示监督，使用必要的 MLP 桥接宽度，在前10步逐渐降低表示 loss 权重、提高 token-level OPD 权重，随后完全使用普通 token loss，推理时丢弃 projector。

**与我们重合：**普通学生 rollout、短程内部辅助、早期特殊价值、后续普通 OPD、部署无额外模块。

**仍有差别：**它需要教师 hidden state 并直接优化表示目标；我们只用教师文本提取学生方向，通过改变 loss 前向中的计算帮助学生参数更新。原设置主要是 Qwen3 尺寸变化及 R1/JustRL，不是任意异构 family 验证。

**新颖度影响：高。** 不能再将“用内部信号作冷启动”作为主要新点。需要证明前向方向干预相对于 transient representation loss 的独立价值。

### 5.2 Latent-MOPD：早期表示辅助也已用于论文称为 cross-family 的 OPD

[Latent-MOPD，2610.02381v1](https://arxiv.org/abs/2610.02381v1)，重点 §3 式(3)、§4.1、§4.3、附录 A。

它把多个领域教师的表示与 token 预测联合传递给学生，必要时使用共享线性映射；表示项早期衰减，随后只做 token-level 多教师 OPD，沿用 LastOPD 式 crossfade。

**与我们重合：**教师冻结、学生 on-policy、短暂内部辅助和后续 token OPD；比单纯常驻表示 loss 更接近我们的冷启动安排。

**证据边界：**论文明确说明其 cross-family 指不同尺寸与后训练谱系，相关模型仍共用 Qwen backbone、28个 blocks。不能把这个标签直接解释为已证明 Qwen→Llama/Phi 的异构 tokenizer 迁移。

**新颖度影响：高。** “跨谱系 OPD＋早期内部辅助”已有组合；我们的方向来源、干预位置及真正异构迁移价值需要单独建立。

### 5.3 OPRD-Bridge：低维通道改善跨模型迁移已有直接研究

它以教师 PCA 和学生投影构造固定低秩桥，再在学生轨迹上更新学生。跨 tokenizer 的 Phi→Qwen 结果意味着“无需共享词表、用低维内部通道”本身已有先例。

我们的区别是方向提取不读取教师 hidden state、不学习跨模型坐标映射；但**低维、固定、跨模型、on-policy**这几个形容词不能作为新点。其桥接后的高表示相似度也不是功能可用性的充分证据，OPMD 正是在这个问题上进一步改进。

### 5.4 OPMD：用学生实际输出检验内部辅助是否可用，也已有先例

它优化 projector 时让信息经过冻结学生的后续计算，再看输出是否接近教师。然后通过普通学生的 on-policy 表示目标内化辅助。这个动机与“教师和学生思维相容”非常接近。

我们若使用“少量固定方向、按撤去后的学习收益选择”，确实限制了辅助形式和选择标准；但仅说“在学生自己的空间里让教师信息更可用”不足以区分。尤其不能把 learned projector 改名为 steering 后再宣称新颖。

### 5.5 主参考 SV Training：向量来源与训练前向位置几乎已经给定

[Steering Vectors as a Training Signal，2610.40998v1](https://arxiv.org/abs/2610.40998v1) 的监督蒸馏分支，已经让学生读取教师文本/自身文本构造方向，再在学生训练前向中注入，最后去掉干预。公开示例包括 Qwen3 教师→Qwen2.5 学生的监督蒸馏。

它与我们最具体的差别，是**训练数据前缀、蒸馏目标以及短程辅助后的 OPD 学习机制**。将其 SFT loss 换成 OPD，再套用 LastOPD 的撤去安排，可能被认为是已有组件的自然组合；“尚未找到完全同流程”不能自动消除这个风险。

### 5.6 新近 Learning Dynamics：学生能否响应教师信号已被直接分析

[Learning Dynamics of On-Policy Distillation，2611.08016v1](https://arxiv.org/abs/2611.08016v1)，重点 §3.2–§3.4。

它将 OPD 分为教师监督转成参数更新的 conversion，以及参数变化如何影响不同前缀和后续 rollout 的 broadcast，并展示 offline preparation 可改善后续 OPD，即使即时 held-out 准确率变化很小。

这支持我们把问题收窄为**学生对教师请求的学习响应**，也说明“训练 KL/overlap 变好，任务却没进步”和“冷启动改善可学性”并非新发现。该分析不证明 steering 一定改善响应，不能直接借它给方法背书。

### 5.7 Trajectory Dropout：普通生成、只干预学生训练前向的结构也已有

[What Shared Prefixes Hide: Trajectory Dropout for On-Policy Distillation，2609.33455v2](https://arxiv.org/abs/2609.33455v2) 让学生正常生成，在学习时丢掉学生可见的部分推理上下文，教师仍看完整前缀，用来加强原本被共同历史削弱的监督。

它与我们都将干预放在学习环节，保留普通 rollout；区别是它改变学生训练上下文，我们拟直接改变学生激活。**仅仅说不改探索、只改训练前向，也不能作为新点。** 需要证明方向性干预有额外作用。

## 6. 两个条件的交集，究竟有没有现成重复？

| 工作 | 有真正跨 family / tokenizer 迁移证据？ | 有内部 steering＋on-policy 训练？ | 与当前全部流程相同？ |
|---|---|---|---|
| CompassOPD | 有 | 不采用激活 steering | 否 |
| OPRD-Bridge | 有，含 Phi→Qwen | 低秩表示目标，不是前向方向注入 | 否，但概念重合强 |
| OPMD | 跨 family 的已报扩展为 SFT | 有 learned residual injection 和 on-policy 内化，位于不同阶段 | 未发现全部条件同时覆盖；不能忽略为近重复 |
| LastOPD | 主要是同系列不同层宽/谱系 | 短程表示辅助；不是 activation steering | 否，但时间安排和功能目标高度重合 |
| Latent-MOPD | 其 cross-family 定义仍是 Qwen backbone | 短程表示辅助；不是 activation steering | 否，但跨谱系＋冷启动组合已有 |
| ACSD / ST-OPSD / SafeSteer | 已核对主体为自/同底座教师，未建立我们的异构外部配对 | 有，教师侧 | 否 |
| DIAL | 未建立我们的异构外部教师迁移设置 | 有，学生侧 adapter | 否，部署保留干预且主干冻结 |
| SV Training | 有跨模型监督蒸馏 | 学生训练前向有 steering；不是 OPD | 否，但向量来源和注入方式高度重合 |
| 当前方案 | 待实验验证 | 拟仅对学生早期 loss 前向注入固定方向 | 本轮未找到完整相同的已核实实例 |

**回答用户的两个问题：**若分别问这两类论文，答案都是“有，而且不止一篇”；若问是否已有论文同时完成当前这一整套实现，本轮的结论是“尚未核实到完全相同”，不是“可以保证没有”。

## 7. 哪些看起来相同，其实需要排除误判？

1. **SimOPD 的 cross-architecture 不等于异构 family。** §4.1 实际配对是 Qwen3-8B→Qwen3-1.7B-Base。它用双 memory bank 比较回答相似度分布，不学跨模型 projector，也不向前向加方向。它仍是强表示蒸馏近邻，不能将摘要的 cross-architecture 擅自升级成 Qwen→Llama 证据。
2. **RIDE 在多个 family 上有效，不等于每个师生对跨 family。** §5.1 明确每对学生初始化自教师的 pre-RL base，架构/tokenizer/head 都共享。其“方向”是动态 RL 表示残差，不是通用跨模型 steering transfer。
3. **Self-Policy Distillation / SPD，2605.22675v2** 确实对生成时的 KV 做能力子空间投影，但之后移除 hook，对生成语料做普通 next-token SFT。不能按名字把它认定为学生新 rollout 上的密集教师分布 OPD。
4. **DASD，2605.22263v1，及 D²-OPD，2611.07241v1** 中的 direction 主要指输出分布监督的吸引/排斥或外推方向，不是 hidden-state steering vector。
5. **REVEAL，2610.16288v2** 与“教师文本→目标模型 replay→目标自己的方向”很接近，但目标 agent 冻结、推理时保留 steering，没有当前的 OPD 参数内化。
6. **The Halt Vector，2608.28859v1** 已将 steered activation 内化到参数，但任务是停止思考控制，使用激活重建等目标，不能当作已完成外部跨 family 的普通输出 OPD。

这些工作有些应进入审稿人的比较视野，但不能为了显得覆盖全面而全部写成同一种方法。

## 8. 还能怎样表述一个有区分度的研究问题？

建议把贡献问题写为：

> 对于已有可比较监督接口、但 OPD 迁移收益仍有限的外部教师—学生组合，少量教师文本能否在学生自身空间给出一种受限的训练前向干预，使普通学生更有效地把教师监督转成参数改进？这种收益能否在撤去干预后保留，并超过相同示范与预算的 SFT/LoRA、SV-SFT 及短程表示辅助？

这比“用 steering 解决跨 family OPD”更具体，但仍是待验证问题。要建立独立价值，至少需要：

- **问题定位：**在统一 tokenizer 接口、模板和合理 SFT 起点后，确认仍有迁移瓶颈，不能把 token 对齐问题重新命名为思维不相容。
- **关键比较：**除了 Compass、Simple warm-up 与 SV-SFT，加入 **LastOPD 式 transient representation auxiliary**；读取教师 hidden 的 OPRD/OPMD 可作为更强但接口/成本不同的对照。
- **因果证据：**同信息量、长度、范数与搜索成本下，真实方向优于随机；关闭 hook 后仍改善普通学生学习；早期收益与一般扰动区分。
- **方法价值：**说明无需教师 hidden 的方向构造有什么实际好处，并报告校准与搜索成本。注意方向发现可只用教师文本，**后续输出 OPD 仍可能需要教师概率访问**，不能称整个流程黑盒。
- **跨 family 证据：**至少一个真正异构师生对；不能只用论文中的 cross-family/cross-architecture 标签替代模型谱系核对。

如果最终只有“换一个 loss、加一个方向，单个组合高一点”，新颖度风险仍较高。如果能证明这种受限前向干预专门改善某类学生的学习响应，且在真实异构迁移与强冷启动对照下成立，才有更清楚的故事。

## 9. 调查范围、证据深度与复核入口

本轮有 **16组 arXiv 元数据关键词查询，去重530条候选记录；补抓35篇固定版本的官方正文，另复用已有核心/参考论文。** 530是宽检索候选数，包含扩散、机器人等非目标条目，不代表530篇都相关或逐篇读完。35是抓取正文数，也不等于35篇都做了逐公式审读。

检索包括 cross-family、cross-tokenizer、heterogeneous、steering、activation intervention、representation、low-rank/subspace、compatibility、warm-up/cold-start，以及11月1–7日新 OPD。部分宽查询超过返回上限；另用完整返回的窄查询补充方向、投影与干预相关 LLM 工作。arXiv 元数据检索不覆盖正文每个用词，也不等于所有出版数据库的穷尽检索。通用网页检索返回无关内容，未用作“没有重复”的证据。

| 阅读优先级 | 论文 | 已核对的关键定位 |
|---|---|---|
| 最高 | LastOPD | §4失效诊断、§5 crossfade、§6.1实际模型 |
| 最高 | Latent-MOPD | §3式(2)(3)、§4.1对 cross-family 的定义、附录 A |
| 最高 | OPRD-Bridge | §3.3桥接、§4.2.4 Phi→Qwen |
| 最高 | OPMD | §3.2残差注入与内化、§4.2跨 family SFT |
| 最高 | SV Training | 监督蒸馏分支与学生空间差方向；已有正文/代码记录 |
| 高 | SafeSteer / OGLS-SD | §3.1/§3.4；§4.4，核对教师干预与输出目标 |
| 高 | SCOPE / PreSO | §3.1–§3.2；§2.3，核对固定子空间与表示目标 |
| 高 | RIDE | §4.2表示残差、§5.1每个 pair 共享谱系 |
| 高 | Learning Dynamics | §3.2–§3.4机制定位，conversion/broadcast/preparation；未复核完整推导 |
| 接口层 | SimCT/BPM/CAST/ESCD/DVLM/可靠性研究 | 本轮复核摘要及正文方法概述；未逐一独立复现实验 |

新增正文中，13篇做了方法与模型设置定点核查，14篇核对正文概述，8篇保留为摘要级候选；每篇深度在 assessment 中标注。另复用已有论文笔记。没有将下载全文等同于完整阅读。

复核材料：

- [全部查询、原始结果与官方正文快照](../audit/novelty_search_2026-11-07/)。
- [去重候选元数据](../audit/novelty_search_2026-11-07/search_pool.json)；[候选筛选表](../audit/novelty_search_2026-11-07/screening.csv)。
- [35篇正文的版本、哈希及获取状态](../audit/novelty_search_2026-11-07/source_manifest.json)。PreSO 官方 HTML 404，改用同版本官方 PDF 提取；未用404推断论文不存在。
- [重点论文的分类与阅读记录](../audit/novelty_search_2026-11-07/assessment.json)；[固定版本元数据](../audit/novelty_search_2026-11-07/selected_metadata.json)。

当前结论是可复核的范围判断，不是全球新颖性证明；没有运行论文代码或验证其报告的成绩。
