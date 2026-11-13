# 完整方案分支与对照实验实现清单

2026-11-13更新：本表以下为补齐前的历史检查；最新[完整队列实施](08_full_scheme_execution.md)已接入匹配随机、
共同后端/文本/激活池、实际短程选参、阶段、强基线、多seed/MATH和机制续写编排。
SFT及跨tokenizer仍单列，EasySteer在线引擎尚未移植；GPU验收及实验结果以新run记录为准。

更新：2026-11-12。结论：**核心可切换分支已接入，完整对照面板尚未齐全，所有新GPU路径仍未验收。**
当前第一步保持T−S、全位置、S0全局均值、同题100与0次OPD；不启动其他面板。

## 状态口径

- `integrated_cpu_checked`：配置/函数已接入当前执行路径且有CPU语义/调度检查；不代表真实GPU或学习收益。
- `partial`：有内核/数据/旧工具或可手工配置，但欠缺完整公平面板/自动编排/报告。
- `not_implemented*`：当前对应实验路径缺失；文档、JSON描述、保存日志不能当作实现。
- `deferred_not_implemented`：尚未实现，且用户此前要求暂缓。

## 逐项核查

| 完整稿节 | 对照 | 状态 | 证据与剩余工作 |
|---|---|---|---|
| 0 | 第一步T−S/all/S0-global，0次OPD | 已接入，CPU通过 | scripts/run_pre_opd_overlap_tuning.py:execute_diagnostic; scripts/pre_opd_overlap_stage.py:score_job；真实GPU扫描未运行；结果后人工检查。 |
| 6.2–6.4 / 10.2 | T−S/T−R/R−S（三方向开关） | 已接入，CPU通过 | src/steer_opd/method_config.py:DIRECTION_ROLES; scripts/run_figure8_campaign.py:prepare_directions；R模型/阈值及正式配置须冻结；不等于共同文本/共同过滤的三方向公平面板。 |
| 7.4 | 所有有效回答预测位置注入 | 已接入，CPU通过 | src/steer_opd/controlled_rollout.py:gate_decision; src/steer_opd/vllm_steering.py；真实模型新接口未GPU验收。 |
| 7.1–7.3 | overlap单阈值、采样前probe门控 | 已接入，CPU通过 | src/steer_opd/controlled_rollout.py:controlled_step；未自动标定阈值或进行dev学习选型；需兼容tokenizer。 |
| 7.5 | 独立有符号熵差门控 | 已接入，CPU通过 | src/steer_opd/controlled_rollout.py:gate_decision；算法为H(probe)-H(T)>tau；阈值尚未校准；不是Entropy-Aware OPD目标。 |
| 7.4 | 因果固定后段注入 | 已接入，CPU通过 | src/steer_opd/controlled_rollout.py:gate_decision; scripts/create_steering_variant.py:main；start_token需提前冻结；无自动估计长度中位数/敏感性面板。 |
| 7.5 | 固定概率随机gate | 已接入，CPU通过 | src/steer_opd/controlled_rollout.py:gate_decision；Bernoulli独立抽取，不保证注入次数或绝对位置分布匹配。 |
| 7.5 / 10.3 | 等注入率/次数的随机对照 | 未实现 | 仅存在random_draw<random_probability，没有读取主门控历史的匹配控制器。；需区分固定前缀离线精确匹配与仅基于过去的在线滚动匹配，并报告误差。 |
| 7.5 / 11.4 | 位置分布与注入次数匹配随机对照 | 未实现 | 无position-bin matching控制器/任务入口。；保存位置日志不等于已经有匹配实验。 |
| 8.3–8.5 | 同题全局均值/S0逐题缓存 | 已接入，CPU通过 | src/steer_opd/direction_bank.py:make_bank,bank_vectors; scripts/run_figure8_campaign.py:prepare_directions；跨run共同文本/提取者/过滤/尺度的匹配协议未自动编排。 |
| 8.2 / 8.6 | 当前学生每轮逐题、周期刷新 | 已接入，CPU通过 | src/steer_opd/method_config.py:Method.extraction_step; scripts/run_figure8_campaign.py:prepare_directions；缓存vs刷新仍需共用参考文本/预算并实测学习；T−S/R−S多一次bootstrap。 |
| 8.3 / 10.2 | 独立题集全局T−R＋同门控对照 | 部分 | scripts/prepare_figure8_first_run.py有独立划分；旧first_run为独立题T−S；新creator固定same_as_opd。；缺冻结相同dev/文本预算的新T−R独立全局面板；当前第一步按用户要求不安排旧/新题集比较。 |
| 8.7 / 10.5 | 本题方向vs另一题方向 | 已接入，CPU通过 | src/steer_opd/direction_bank.py:bank_vectors; scripts/create_steering_variant.py:--vector-assignment；实现是seed确定的循环错配、保证不取本题；不是任意随机置换集合。 |
| 10.5 | +v/−v/等范数random，多随机seed | 已接入，CPU通过 | src/steer_opd/direction_bank.py:bank_vectors; scripts/run_pre_opd_overlap_tuning.py:confirmation_conditions；0更新confirm编排有3个方向seed；完整多训练seed比较尚无编排。 |
| 0 / 6.6 | 单层位置×强度、alpha0/移除恢复 | 已接入，CPU通过 | scripts/run_pre_opd_overlap_tuning.py:grid_conditions; src/steer_opd/overlap_tuning.py:evaluate_candidate；15主候选+alpha0已编排，未GPU运行。 |
| 0 / 6.6 | 双层/多层与匹配名义总强度 | 部分 | src/steer_opd/steering.py:CompositeSteeringController; src/steer_opd/direction_bank.py:bank_vectors; create_variant --layers；内核和手工job支持多层rho/sqrt(m)，当前第一步仅单层网格；选两层/敏感性自动编排未接入。 |
| 7.3 / 7.5 | 训练/dev阈值、初始注入率与选型 | 未实现 | creator仅接收数值；presets阈值为null并拒绝缺失值。；没有由10%–30%触发率构造候选、dev选择tau并冻结的流水线。 |
| 10.2 / 10.5 | 改变gate时保持共同生成后端/采样器 | 部分 | scripts/run_figure8_campaign.py:generate按gate选择vLLM/HF；每面板内vanilla匹配HF。；跨all/overlap面板后端仍会变化，缺面板级共同后端开关/调度；HF内核可手工all，但不是campaign的一因素消融入口。 |
| 6.5 / 10.5 | 同文本/共同过滤/尺度的三方向比较及范数夹角 | 部分 | 原始差分和fixed_activation_scales接口存在；extract_pairs每次只处理选定两角色。；缺T/R/S共同有效交集、跨run固定文本复用及范数/夹角/抵消统一报告。 |
| 10.5 | 输出模板/长度控制后的方向 | 当前流程未实现 | 旧probe_cross_family_steering.py有截断到匹配内容长度的诊断。；当前训练只做正文mask/均值池化，不是模板归一化/长度匹配的学习对照。 |
| 10.5 | T和R共同答对集合上的三方向 | 未实现 | 当前extract_pairs只过滤空白/截断/严重重复/近零。；缺参考回答判题、共同正确交集及相同预算重提；旧工具的teacher-only正确筛选不等价。 |
| 9.6 | 仅rollout/仅learning、末期撤去辅助修复 | 未实现 | 新campaign始终同向量/mask生成并学习；旧verl_extension为不同历史loss-only实现。；本轮已改为显式拒绝未知injection阶段/日程，不再可能只改标签而继续原路径。 |
| 10.3 | 同初始化普通OPD | 已接入，CPU通过 | scripts/run_figure8_campaign.py:train_step；旧实验有真实10步；本轮新Top16/门控路径GPU仍未验收。 |
| 10.3 / 10.4 | 同文本SFT/warm-up+OPD及共同预热轨道 | 暂缓，未实现 | 本轮没有warm-up训练/预算编排；loader可指定初始化不是完整对照。；用户此前明确暂缓本地SFT，不自动新增该任务。 |
| 10.3 | CompassOPD含学生锚定项 | 未实现 | response_loss仅使用T与S奖励；R只供SV参考文本。；JSON方法描述与参考文献不是新训练目标；需独立实现/复现。 |
| 10.3 | Entropy-Aware OPD KL目标基线 | 未实现 | 已有熵差gate；原OPD loss没有按熵调整正/反KL组合。；不能用熵差激活门控冒充该基线。 |
| 11.3–11.4 | 注入前overlap分桶与师生续写/恢复能力 | 部分 | token_statistics、固定前缀与自由rollout统计已有；部分位置/长度桶已有。；缺分桶续写/正确性/恢复能力实验及统一机制报告。 |
| 10.6 / 13.2 / 14 | 多训练seed、同实际计算预算对照与多评测集 | 部分 | seed参数、逐题bootstrap、进程/训练时间/长度/截断记录存在；当前主评测AIME。；缺3训练seed自动编排、汇总std/训练结果CI、等计算预算增加普通OPD、当前流水线MATH-500。 |
| 12 | 跨tokenizer共同分布/首字节gate+span OPD | 训练未实现 | cross_family_overlap.py为字节/字面诊断；当前loss仍同token-ID评分。；完整在线门控与span蒸馏/桥接训练未接入，也未自动扩展配对。 |

## 三个容易误判的点

**随机gate不等于匹配随机对照。** 当前是独立Bernoulli抽取；没有主门控的历史注入率/位置统计输入，不能声称同次数或同位置分布。

**面板内匹配后端不等于所有消融共用后端。** 门控面板普通/steer都走HF；all面板走vLLM。比较all与overlap时还会改变后端/采样器，当前campaign缺共同后端配置，须补齐或单独控制。

HF是现有门控参考实现的选择，不是SV/位置mask必须使用的引擎。EasySteer可接已知位置列表；
精确在线overlap gate还需逐步probe/教师交互及KV分支集成，不能用旧轨迹index直接代替。
能力边界及已核对官方接口见[方法分支4.1](05_method_branches.md#41-easysteer可以按位置注入hf是当前实现选择)。

**方向可以切换不等于信息预算已经匹配。** 现有每个方向分支各读两种文本，跨run参考文本、共同有效样本和尺度还需统一冻结；数学分解CPU测试不替代公平实验流程。

## 本次修正

1. 固定前缀诊断按真实row mask留档；旧已绑定缓存的误写mask可以CPU修正，原概率/指标不重算。
2. 复用bank同时核对真实bank与metadata的方法、提取checkpoint和步数，防止复制旧bank后跑错rho/粒度/提取者。
3. 未实现的rollout-only/loss-only/末期撤去配置直接报错，不能只改标签后仍按双前向注入运行。

这些修正没有实现上述缺失面板，也没有改变当前第一步配置/数据或恢复实验。

## 后续实施优先级

进入位置消融前优先补共同后端选择、等注入率随机与位置匹配随机；进入方向/粒度比较前优先补共同文本/过滤/尺度面板。
模板/正确性控制、强基线、多seed/成本、跨tokenizer按阶段另行实施，SFT继续按用户要求暂缓。
完整原始条目/CPU证据见[coverage.json](../audit/control_coverage_review_2026-11-12/coverage.json)；测试与本次状态见[审计](../audit/control_coverage_review_2026-11-12/README.md)。
