# TODO

2026-11-14用户明确后续消融先不急，当前优先核对和准备提取/评分/训练后端效率。
[调研结论](docs/18_efficient_steer_opd_backends_20261114.md)：EasySteer已有中间层capture和DiffMean/streaming接口，提取不要求HF；CompassOPD用verl训练、vLLM采样及教师评分。
目前仅完成论文/官方源码核对，未安装环境、迁移生产代码或开始实验；以下后续消融保持暂缓。

2026-11-14再次复核：用户再次要求只检查实现，待下一步明确同意后才执行。当前执行锁和前置缺口均保留。
源码核对仍为：八方案结构一致；完整字节overlap尚未接入；T/Ref概率评分与SV提取仍HF；EasySteer GPU未验收。
本轮没有改生产代码或冻结job，也没有启动GPU。最新证据见[2026-11-14复核](audit/implementation_recheck_20261114/README.md)。

## 待用户同意后执行：新配对八方案零更新筛选

状态：**awaiting_user_approval**。2026-11-13用户要求先复核实现，下一步经用户明确同意后才执行。
当前只做代码、文档和CPU检查；不启动GPU、模型生成/评分、训练、下载完整权重或恢复旧任务。

2026-11-14用户接受：跨tokenizer静态门控按完整原生支持聚合下一字节概率，使用sum(min)，并用Compass span评分检查区分度。
此处同意的是指标定义，不解除GPU执行锁。当前新面板仍0更新；旧Student Top16学习配方与未来sampled-span学习配方分开记录，尚未自动切换目标。
已核对普通vLLM0.32的max_logprobs默认20，可配置并支持-1；当前生成入口显式20、EasySteer评分入口显式16。它们限制返回候选数，不限制模型原生词表或聚合完整字节分布。

| 设置 | 已固定 |
|---|---|
| T / Ref / S | Qwen3.5-35B-A3B / Qwen3.5-0.8B / Qwen3-4B |
| 算子 / 层 | Slerp alpha=0.1，75%单层，0-based26 |
| 方案 | T−S/T−Ref × all/static_overlap × global/per_question，8组＋共同无SV基线 |
| 数据 / 更新 | 原100训练题候选，实际共同有效交集据实报告；同源固定前缀，0次参数更新 |
| mask | 普通源rollout→教师后评分→冻结；注入后不重判 |
| 后端目标 | 生成用vLLM；8组和重放基线统一EasySteer，无自动HF降级；序列调度使用引擎默认 |
| 结束点 | 报告8组select、冻结候选confirm和默认前3条图，等待检查；不自动进入OPD |

- [x] 已实现八方案结构、四份bank、Slerp、静态mask绑定、概率缓存及select/confirm编排。
- [x] 新配置已加执行锁；锁传给独立生成/提取/评分job，执行入口在加载模型或申请GPU之前拒绝执行。
- [x] 当前路径不传max_num_seqs或手工CUDA Graph尺寸；新配置拒绝人工序列上限，HF评分batch与vLLM调度分开。
- [x] 2026-11-14重新核对生成/静态评分LLM构造与准备job，均未设置人工seq上限；5份独立阶段job保留执行锁，八方案/四bank均在。
- [ ] **修正跨tokenizer指标差异**：完整稿§12.2要求全支持下一字节概率聚合和sum(min)，现代码仅有共同span起点的Top16完整token字节交集。当前指标只作独立诊断，不能作为MD复现结果。须补全支持概率质量、EOS/控制类别、有效边界、冻结mask与CPU验收；保留旧诊断缓存，不改名冒充。
- [ ] 完整字节聚合需同时接入源rollout、T/Ref缓存、EasySteer候选评分、select阈值和图/排名；只增加一个CPU公式不足以完成执行链路。输出空格/控制类别质量与覆盖率，检查门控区分度。
- [ ] 部署隔离的受支持EasySteer fork及项目Slerp插件；固定解释器与版本。当前普通vLLM0.32不等于该fork。
- [ ] 核验35B教师完整权重；当前只备有其官方tokenizer/config元数据。
- [ ] 用户同意后先做GPU验收：无SV源概率与无hook重放、alpha0/移除恢复、首预测/t−1/EOS/padding、逐题向量绑定、Slerp、全8组共同后端。
- [ ] 补齐EasySteer真实扰动记录；当前只有mask/向量，relative_perturbation_available=false，不声称满足完整扰动留档。
- [ ] 补齐模型checkpoint内容指纹：当前新接口主要绑定模型路径及文本/tokenizer/产物hash；同路径替换权重不能仅靠路径身份检出。正式运行前固定权重revision/清单并纳入全部缓存身份。
- [ ] 若要连T/Ref概率评分也全部使用vLLM，补实现原生固定前缀批量评分及相同输出支持验收；当前这一步仍是HF。方向提取也仍是HF，需逐token中间层激活；不得把普通embedding接口称为等价替代。
- [ ] 接入EasySteer capture提取后端：单卡S0、层26、正文行与BF16逐token留档，保留原始均值/逐题权重/零方向定义；当前官方接口已有，项目尚未集成。与已截断到前27个block的HF路径做同协议验收后再选后端。
- [ ] 将T/Ref固定前缀评分迁到vLLM，合并实际文本/span分数与完整字节聚合；保持模型各自原生tokenizer/support。未来学生梯度更新优先评估verl FSDP，不把推理接口当训练接口。
- [ ] 得到用户明确执行同意且前置缺口修复后，解除执行锁；通过空闲GPU准入执行，优先1张、最多2张，不触碰其他项目。
- [ ] 实测吞吐、显存和各阶段耗时；无实测前不承诺提速倍数。

执行配置：[compass_static_eight_20261113.json](configs/compass_static_eight_20261113.json)。
复核证据：[实现与Markdown一致性复核](docs/17_implementation_recheck_20261113.md)。
早期HF准备目录runs/compass_static_eight_20261113保留，新待执行目录runs/compass_static_eight_review_20261113；不替换旧冻结job。

## 完整研究方案：后续另行决定

以下不属于本轮8组零更新筛选，不能称为已经复现，也不自动执行。

- [ ] 新配对的实际OPD目标/梯度重放、行为与学习条件、撤去SV后独立能力。
- [ ] 当前学生逐轮重提vs同S0逐题缓存、R−S、正负/随机/错题向量、匹配注入次数/位置、模板/长度/共同正确集合。
- [ ] 强基线/同文本SFT与共同预热、CompassOPD目标源码对照、多seed/MATH与机制续写、等实际计算预算。
- [ ] 后续层/强度搜索是否继续由用户决定；此前40/192配置不进入当前自动队列。
- [ ] CompassOPD官方仓库定位与逐项对照；当前span评分依据论文A.4独立实现，尚未完成官方源码复现。
