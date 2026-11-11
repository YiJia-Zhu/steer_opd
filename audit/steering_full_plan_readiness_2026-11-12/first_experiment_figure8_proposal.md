# 第一个实验：Figure 8配对下的普通OPD与全位置steering

2026-11-12。用户要求先详细说明首个实验，并询问是否与Rethinking Figure 8一致。
本文件是本次讨论的具体建议，尚非可执行配置，也不代表用户已冻结损失、方向、
新增数据规模或启动预算。本轮不下载、不生成回答、不运行GPU。
完整方案的T−R/overlap/全词表RKL主线保留；本文提出独立的Figure 8配方参考面板。

## 1. 两个组和要回答的问题

| 组 | 初始参数 | 方向 | 训练生成与学习 | 主评测 |
|---|---|---|---|---|
| F8-Vanilla | Qwen3-1.7B-Base | 无 | 原版普通OPD | 无hook学生 |
| F8-TS-All | 同一个Base | S0读T回答−S0读自身回答 | 整个OPD阶段，全有效回答预测位置注入；生成和学习重放一致 | 无hook学生 |

教师固定Qwen3-4B Non-thinking。本轮T−S作为后续直接对照，不需要R；总方案中的
T−R和R−S没有删除。本地SFT组/作者SFT权重继续暂缓。
先运行普通组，再从原始Base另起steer组；不能从普通组训练终点续训steer组。

问题：固定这个小数据预算，加入方向后能否改善独立学生accuracy及overlap？
2026-11-12用户明确：只要相较同条件传统直接OPD，overlap拉近、accuracy提高，
就算本轮符合预期。论文SFT数字由用户自行作近似参照，不要求本地SFT第三组，
也不把对齐SFT起点或达到论文SFT成绩作为本轮条件。

具体口径：两组相同初始化、教师、训练题、损失、更新预算和评测协议；主看预先规定
终点，并保留全部checkpoint曲线。关闭steer后，在同一固定前缀集计算Overlap@16，
在同一评测集计算accuracy；二者均高于直接OPD即记录为先导符合预期。
即时带hook指标另记，不与无hook学习收益混用。未设置必须超过多少百分点的新增门槛。

## 2. 与原论文的关系

来源：[正文§5.1](../../papers/sources/2604.13016v2.txt)、附录A.2表2、C.1表3；
[原fork启动器](../../ref/ref_rep/Rethinking-OPD/on_policy_distillation.sh)，
固定commit `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。
论文事实与通用脚本默认值分开：脚本当前活动模型/数据是R1/JustRL/DAPO，
必须根据论文§5.1切到Qwen3/OT3，不能直接执行原shell。

原Figure 8比较Base→OPD与Base→SFT→OPD。SFT使用约200K教师回答、全参数1 epoch、
LR1e-5、序列14336、BF16、cosine、warmup0.05；之后用与SFT题去重的约30K OT3题OPD。
OPD表2默认1 epoch，图中显示0–200步不能等同于论文只训练200步。
图中评测包括AIME24/AIME25/AMC23；本轮沿用用户指定的AIME2025一项。

新完整方案的主目标是全词表显式RKL；原Figure 8默认Student Top-K16。
为贴近用户当前关注的图8，本建议首轮保留原fork Student Top-K16路径，作为独立参考。
它不能直接充当以后全词表RKL方法的唯一对照。以后若改损失，必须重跑匹配损失的普通OPD。
这项目标选择是本次建议，不是已经批准替换完整方案。

## 3. OPD建议配置

| 参数 | 本轮建议 | 证据/说明 |
|---|---|---|
| 教师 | Qwen3-4B Non-thinking，冻结 | 同§5.1 |
| 学生 | Qwen3-1.7B-Base，全参数更新 | 同§5.1，不能用普通1.7B |
| 数据 | 原30K OpenThoughts3 OPD parquet抽100题 | 同题源，规模改变 |
| 每个trainer step的prompt数 | 64 | 表2 |
| 每prompt实时回答数 | 4 | 表2；每步256条回答 |
| mini-batch | 原fork参数64，按n=4展开为全局256条回答 | fork在worker中乘rollout.n再按设备分片 |
| 每批更新轮数 | 1 | fork默认ppo_epochs=1；检查实际更新计数 |
| 总步数 | 50 trainer steps/组 | 项目建议，不是论文总步数 |
| 蒸馏候选 | Student Top-K16，only_stu | 表2和fork |
| 权重/优势 | student_p、token_reward_direct | 保留fork真实surrogate语义 |
| 学生采样/教师评分温度 | 1.0 / 1.0 | 表2及fork |
| 生成截断 | top_p=1，无top-k采样截断 | 监督Top16与生成截断分开 |
| repetition penalty | 1.0 | fork |
| prompt / response上限 | 1024 / 7168 | 表2 |
| 学习率 | 1e-6 | 表2 |
| 优化器 | AdamW；betas=(0.9,0.999)，weight_decay=0.01 | fork默认配置，非论文表2逐项报告 |
| LR schedule | constant，warmup=0 | fork默认配置；启动时显式固定 |
| 梯度裁剪 | 1.0 | fork默认 |
| loss聚合 | token-mean | fork；累积/分布式缩放须核对 |
| 额外约束 | 无初始策略KL、无额外熵奖励/正确性奖励 | 蒸馏监督仍保留；KL coefficient=0不是取消蒸馏 |
| 精度 | FP32主参数/优化器状态，BF16前向，敏感归约FP32 | 参照fork mixed precision，实际验收后记录 |
| gradient checkpointing | 开启，hook覆盖重算 | 两组相同 |
| micro-batch/设备数 | profiling后固定，初始微批1 | 不通过缩小全局batch/长度静默改变实验 |
| seed | 42，数据访问顺序一致 | 单seed先导，非重复种子结论 |
| 保存 | 10/20/30/40/50步 | 项目新增；原shell save_freq20 |

每组64×50=3200次prompt访问，100题完整循环32遍，每题128次实时采样，
共12800条训练回答；两组共25600条。固定100题顺序循环到3200条访问流，
不使用100题单轮drop_last长期漏掉最后36题。学生参数改变后重新生成，绝不复用离线方向文本充当rollout。

原fork以student_p加权的log-prob差构造冻结reward，并经PPO surrogate更新；
不能把它未经验证地改写成手写显式KL后称同一实现。steer组的生成、student top16、
student_p权重、行为log-prob及学习重放须全部来自一致的受控学生定义。
本轮没有动态probe/门控，g始终为1；无方向组g=0/alpha=0。

## 4. steering建议配置

- 独立D_vec：建议256道OT3训练题，与100题OPD和dev/test均不重合；尚未抽取。
  独立D_dev：建议128题；本轮固定层/强度，不据AIME曲线选参。
- T/S0各生成每题1份回答，共512份。校准解码temperature0.7、top_p0.95、
  无top-k截断、max12288；这一解码借鉴C.1，题数及方向提取为本项目新增。
- 共同保留非空、完整、非退化的成对回答，报告排除与有效数量；不默认只留下教师答对题。
- 两类文本都由同一冻结Base重编码、读取第14个block输出（0-based13）；
  正文有效状态每题先均值，再对题目等权平均。排除prompt、padding和模板结构成分。
- `d=mean(Phi_S0(x,yT)-Phi_S0(x,yS))`；`v=d/||d||`；
  `h_new=h+rho*s*v`，`s`为校准中该层正文状态L2范数的中位数，rho=0.03。
  固定方向/尺度，报告近零向量与实际精度下扰动，不将0.03解释为固定alpha数值。
- 方向在OPD前提取一次；层13/rho0.03为工程起点，非论文推荐最优参数。
  学习步1–50都启用；回答token t的预测状态t−1注入，首位置为prompt末位。
  生成和训练重放两侧都注入，评测关闭；不采用旧loss-only默认。

## 5. 评测与曲线

只用用户指定本地AIME2025，共30题，avg@16、temperature0.7、top_p0.95、
无top-k采样截断、max31744，保留明确的合法stop规则。每checkpoint共480份回答。
评测0/10/20/30/40/50步；第0步可复用同一Base评测，其余各组独立评测。
avg@16是16次采样的平均正确率，不是16次有1次对就算通过的pass@16。
本地题目字符串与作者parquet不完全一致，旧审计记录了答案单位差异；保留用户指定来源，
检查grader，不声称评测文本逐字复现。

主准确率在完全无hook、无教师、无参考模型的学生上测。
主overlap建议在固定初始Base前缀集上，对各checkpoint无hook计算Top16交集/16，
同时报告共享概率质量。训练受控轨迹的overlap另记，标清策略状态；
原Figure 8训练轨迹overlap与这份固定前缀诊断的取样方式不同，不混成同一曲线定义。
固定前缀从训练侧校准/dev题预先生成，不能使用AIME选门控或方向。
回答长度、截断、重复、耗时、教师评分token和校准成本同时记录。

两组第0步无hook分数应相同：离线提取方向没有更新Base参数。
第0步复用同一Base评测即可，不围绕论文SFT起点增加本轮工作。成功判断按上述
相对直接OPD的无hook overlap及accuracy增量执行。

## 6. 尚待实现验收

需要新入口：现有项目launcher仍为R1/JustRL及teacher-top-K forward KL，
原shell含全局ray stop且当前活动模型/数据不同，不可直接执行。
先通过真实Base/教师模板、stop/token ID、候选概率与mask验收，
alpha0恢复匹配原fork基线数值/梯度，非零生成和重放一致，checkpoint重算保持hook。

CPU检查已发现教师比Base多四个已定义词元。需核对原fork实际输入及候选评分是否落在
有一致定义的支持上，记录未定义/额外head行质量；若新增mask或归一化须两组共同采用并
公开为配方改动，不能悄悄称精确复现。尚未冻结这一实现细节，也未完成GPU加载验收。

本轮准确名称为“Figure 8配对与OPD超参数下的100题steering先导”。
数据规模、训练遍数、SFT替换方式、受控rollout、评测范围和诊断取样均有明确改动。
