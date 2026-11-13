# Repository Guidelines

## Latest Efficiency Research — Defer Ablations, Consider Capture and verl

2026-11-14用户要求后续消融先不急，先查steer提取和高效OPD。仅调研/文档，不安装或运行环境，不改生产代码，不启动GPU/新生成评分/训练；执行锁保持。
核对SV Training官方acts.py仍HF逐题提取所有层；ACSD/ST-OPSD论文主要用固定/预算/选定位置降低成本，不能推断其有更快专用引擎。
EasySteer官方已有capture指定层/位置及DiffMean/streaming提取，前端baecd969对应项目固定fork7837b07d；提取不一定HF，当前项目尚未接该capture路径。
捕获没有服务端mean归约，正文capture可能跳过prefix-cache读取，helper单worker；仅选层不自动截断后层。当前HF已只跑前27个block，需同协议比较，不承诺倍数。
Compass明确verl训练、vLLM采样和教师评分；未来优先考虑verl FSDP训练并迁T/Ref无梯度评分到vLLM。推理引擎不替代梯度更新，默认teacher-topk损失不等于原Student Top16或跨tokenizer span目标。
详见docs/18_efficient_steer_opd_backends_20261114.md、audit/efficient_backends_20261114/；TODO已加效率准备项，后续消融不自动推进。

## Latest Recheck — 2026-11-14, No Experiment Execution

用户再次要求记录TODO、先核查实现，下一步明确同意后执行；execution_hold与既有blockers均保留，当前不启动GPU或新模型生成/评分/训练。
已重新读取拥有源码与配置：8组/4bank、Slerp0.1/层26/36层学生、全部5份准备job的hold一致；当前LLM调用与job没有人工seq上限。
接受的完整字节sum(min)仍未实现/接入，当前仍Top16完整token字节交集。不能称MD已完整复现，也不能把通过旧诊断CPU测试说成字节指标通过。
生成走vLLM，T/Ref评分和方向提取仍HF；八组默认EasySteer但受支持fork/GPU未验收。源概率一致性、扰动、权重内容hash和完整OPD仍有缺口。
新的只读审计audit/implementation_recheck_20261114/；与11月13日审计相比已核对Python源码无变化，旧配置/冻结job保持。本次仅补TODO/文档/审计和CPU回归。

## Latest Metric Agreement — Full Next-Byte Probability Overlap

2026-11-14用户接受完整原生支持的下一字节概率聚合及sum(min)作为跨tokenizer静态门控定义，并用Compass span评分检查区分度；不等于授权GPU执行。
当前新面板仍0更新，执行锁保留；旧Student Top16训练配方尚未自动替换为sampled-span训练。Top16诊断缓存不能充当完整字节概率。
vLLM0.32 max_logprobs默认20且可改/-1；当前生成显式20、EasySteer评分显式16，仅影响返回候选上限，与max_num_seqs独立；新版fork实际行为仍须验收。

## Latest Recheck — Hold Experiments and Match the Actual Markdown

2026-11-13用户要求将本实验记入TODO，下一步明确同意后才执行；当前只做代码/文档/CPU复核，无GPU/模型生成/评分/训练或完整权重下载。
TODO.md为当前待执行清单；configs/compass_static_eight_20261113.json已execution_hold=awaiting_user_approval，并保留尚未解决的execution_blockers；锁传播到全部新阶段job，在模型/GPU/执行写入之前检查。
固定T=35B-A3B、Ref=0.8B、S=Qwen3-4B、Slerp0.1/层26，8组＋基线、0更新。旧40/192网格仅历史参考。
当前生成实际用普通vLLM；T/Ref概率和SV提取仍HF，不能称为全程vLLM/EasySteer。下轮8组/基线默认选择EasySteer且无自动HF降级，但fork本地GPU未部署/验收。
新配置拒绝手工序列/CUDA Graph尺寸上限，当前LLM调用未传max_num_seqs；resolved_max_num_seqs仅记录引擎值，HF batch_size不是vLLM并发上限。
重要差异：用户完整稿12.2路径B要求全概率下一字节聚合及sum(min)，现compass_aligned_byte_top16_overlap是Top16完整token字节交集，不能称一致复现；须补指标后才进入正式实验。
EasySteer扰动量尚未记录，源概率与重放的一致性仍需GPU验收；完整OPD/跨tokenizer训练/后续控制面板尚未在新配对复现。
旧CPU准备run/config冻结保留；新待执行目录runs/compass_static_eight_review_20261113。详见docs/17_implementation_recheck_20261113.md及audit/implementation_recheck_20261113/。
本轮项目tests/全套CPU200 passed/1 skipped，相关路径65 passed；包含各独立入口的执行锁和序列上限拒绝检查，未使用GPU。

## Latest Implementation — Compass-Style Cross-Tokenizer Eight-Way Pipeline

2026-11-13用户要求参考CompassOPD代码仓库。论文v1/公开检索未定位官方仓库，已请求URL；本轮按附录A.4独立实现，不冒称源码移植。
共同UTF-8字节边界span、T/Ref原生批量概率评分、span平均logp/家族差分广播已接入；学生实际采样token保留，控制/EOS/不可对齐位置显式mask。
门控/排名指标明确为项目适配compass_aligned_byte_top16_overlap：共同span起点的Top16续写字节交集/16，再广播到学生span；不是原生token-ID overlap或完整概率输运。
新入口scripts/run_compass_static_panel.py及configs/compass_static_eight_20261113.json覆盖参考文本→4份bank→源Top16→T/Ref分数→阈值→全8组select→冻结confirm；全部0更新，仍Slerp0.1/层26。
35B官方tokenizer/config元数据固定59d61f3ce65a6d9863b86d2e96597125219dc754；T/Ref映射相同，各248077，S151669，各自原生定义支持，不复用151665。
已CPU准备原100题的新prompt及70/30训练侧select/confirm，实际有效数需参考/激活/对齐过滤后报告；不能称独立dev或沿用旧55题。
默认HF批量参考后端，EasySteer可显式选择独立解释器；新版GPU fork仍未部署/验收，无吞吐倍数，不自动降级或恢复旧训练。
CPU全套186 passed/1 skipped，覆盖真实tokenizer、实际Qwen3.5 dense/MoE缩小文本架构（Torch参考核）和8组；新35B模型GPU路径未验收。
预检runs/compass_static_eight_20261113/preflight.json为missing_assets，已查路径缺35B完整权重；仅下载元数据，无GPU、新生成/评分或OPD。
权威更新docs/16_compass_alignment_eight_way.md、audit/compass_span_adapter_20261113/；官方仓库源码逐项核对仍待用户URL。

## Latest Implementation — Static Masks, Slerp and EasySteer Adapter

2026-11-13用户明确要求修改代码，覆盖此前仅文档范围；已实现静态mask、Slerp单条/批量前向和独立评分入口，无GPU/OPD。
新入口scripts/run_static_overlap_screen.py默认CPU准备，--execute仍须空闲GPU准入；教师仅批量缓存评分，无在线probe/教师/KV双分支。
新static_overlap标签已有代码；旧overlap保留动态语义，旧生成/campaign入口拒绝新静态/Slerp配置，避免误执行。
EasySteer固定fork API已接项目steer_opd_slerp插件和逐请求selector；首预测prompt末位、后续t−1，空mask显式False。
插件仅split模式，无full-graph Slerp kernel；入口元数据只安装项目.runtime/easysteer-plugin，不改共享环境。
CPU全套165 passed/1 skipped，后续局部88 passed；真实固定上游API/selector CPU契约通过。
当前.venv-opd仍是普通vLLM0.32，新版EasySteer GPU引擎未部署/验收，无速度倍数，不能称本地迁移完成。
新配对共同概率事件计算器仍未实现，原生token-ID适配拒绝异构映射；35B教师权重在已查路径仍缺。
本轮保持Slerp0.1、75%层26、8方案＋基线，尚无新模型8组评分。最新代码状态docs/15_static_mask_slerp_implementation.md，审计audit/static_mask_slerp_implementation_20261113/。

## Latest Selection — Slerp 0.1 at 75%, Compare All Eight Schemes First

2026-11-13用户选定第一轮只固定Slerp alpha0.1、75%单层，完整比较T−S/T−Ref × all/static_overlap × global/per_question的8方案及共同无SV基线。
教师Qwen3.5-35B-A3B、Ref Qwen3.5-0.8B、学生Qwen3-4B；36层的75%对应0-based26，第27个block输出。
本轮先做冻结S0、同源前缀零更新选方案，8组全部保留；不自动执行此前40配置建议/192全网格或进入OPD。
权威规格docs/14_slerp_eight_scheme_screen.md及experiments/next_pair_slerp_scheme_screen_20261113.json；后者不是可执行campaign配置。
静态位置规则已在文档确定，但生产代码gate=overlap仍为旧动态probe，Slerp/静态适配尚未接入，不能声称已完成代码改造或8组运行。
新配对的跨tokenizer指标/教师权重仍须核验，旧151665支持/阈值和1.7B缓存不直接复用。

## Latest Next Pair — Qwen3.5 Teachers and Qwen3-4B Student

2026-11-13用户确定下一配对T=Qwen3.5-35B-A3B、Ref=Qwen3.5-0.8B、S=Qwen3-4B，并询问贪婪搜索顺序。
学生本地36层，50%/75%/100%映射0-based17/26/35；旧1.7B向量/尺度/概率/阈值不直接复用。
本地Ref与S词表ID映射不同，35B教师tokenizer尚未验收；不能直接沿用旧token-ID Overlap@16/151665支持或把学生ID喂给教师。
新配对须先定义并验收同文本前缀的共同概率事件；方向两端仍由同一个新学生读取。
docs/13_next_pair_greedy_search_proposal.md提出8方案×2算子粗筛16项→选层增4→选强度增6→8方案回查增最多14，合计最多40非零配置及共同基线。
加法/Slerp各保留一条路径，若方案换出可再局部回查最多10项；此顺序是建议，用户尚未最终选定，不伪称已运行。
本次只记录新配对/建议及本地元数据，无代码/旧配置修改、无GPU；旧192为完整候选池和历史规格，旧55题是旧配对有效数。

## Latest Decision — Static Overlap Replaces Dynamic Token Gates

2026-11-13用户明确“使用静态overlap”“相关文档也改成静态overlap，动态的开销太大”。
当前：普通学生rollout保存原生Top16→教师在同源前缀批量概率评分→O_src→冻结g_static=1[O_src<tau]。
应用后不重判idx，不要求采样前probe/在线教师回调或动态KV分支改造。SV粒度/刷新与mask静态性分开。
8方案改为T−S/T−R × all/static_overlap × global/per_question；两种算子各0.01/0.1/0.3/0.5，单层50%/75%/100%，仍192非零候选及共同基线。
当前只做同源前缀零更新比较，蓝线整段无SV源基线、橙线固定mask前向，默认前3条图/散点/差值。
已完成源回答不能被事后注入改变；后续learning-only与静态mask二次生成分开定义，用户尚未选择时不自动开启任一训练阶段。
learning-only行为/学习条件不同，现代码要求显式candidate_rkl；二次生成需自己的行为/教师概率，不能套源前缀分数。
EasySteer固定位置selector可表达已知mask；首预测在prompt末状态，t−1映射/逐题绑定/Slerp/本地迁移仍须验收。
本次仅同步文档/新规格，生产源码和旧配置不改、无GPU/新搜索。旧gate=overlap代码仍是动态实现，static_overlap仅计划标签。
旧动态图/耗时/失败结果保留原义：256条2:48:30、5条重放失败、正式更新0；不冒充新静态结果。
当前权威协议docs/12_static_overlap_protocol.md；搜索docs/10_overlap_parameter_search.md；审计audit/static_overlap_protocol_20261113/。
此节覆盖以下历史动态主线/引擎前置要求，历史记录和真实代码状态不重写。

## Latest Continuation — Static Search Cache Readiness

2026-11-13续接指定保存对话后仅核对源码/缓存并补文档，生产代码/旧配置74份hash与静态同步时一致，无GPU/新搜索。
公共T/R/S池55题、55份池文件hash匹配，现有层6/10/13/16/20；27仍需补提取。
旧16+16题独立dev与55题公共池无交集，不能直接用于本题逐题SV的8方案；新搜索须准备共同训练题源前缀并预先冻结训练侧select/confirm。
旧学生参考文本temperature0.7/top_p0.95且logprobs=None，无原生Top16；可保留作方向文本，不能冒充新源概率缓存。
根README已补当前静态入口、旧动态入口的阶段边界；证据audit/static_overlap_continuation_20261113/。
旧候选和面板仍failed_resumable，核对时无对应监督/生成/训练进程；顶层running为旧字段，正式更新0。

## Latest Review — Online Gate Cost and Engine Modification Route

2026-11-13用户询问动态overlap门控耗时及能否修改EasySteer源码。已核对旧256条rollout起止07:21:55→10:10:25，
子进程10110.226秒（2:48:30），内部生成10105.880秒；337427token、33.375token/s，教师在线评分已包含。
实际50529次注入（14.9748%）；31次显存拆批含15次OOM/16次预防压力，另24次重组批。首次等卡5:00:35另计。
这是旧双层T−R自由生成，不是新单层固定前缀网格或完整OPD步；后续重放失败、正式更新0。
源码可改：runner采样前增加probe/教师/gate/正式分支；GPU动态mask和request映射；上层当前slot临时KV提交/丢弃。
新单层候选可复用注入点及之前计算，触发后只跑上层；末层输出注入只需重算最终norm/lm_head。
末层固定token前缀可批量评估；自由生成仍依赖实际新token，不能预固定另一轨迹idx。
先统一HF批量门控/同步/分段复用并验收，匹配性能基准后再引擎迁移；加法/Slerp保持同协议。
本次只存源码/时间审计并补计划，未修改生产生成/训练代码或环境、未开GPU；未有提速倍数或迁移完成结果。
证据audit/online_gate_performance_20261113/；路线docs/11_online_gate_acceleration.md，已链接下一步192候选协议。

## Latest Selection — Eight Schemes, Additive/Slerp, Single Layers

2026-11-13用户固定：T−S/T−R × all/overlap × global/per_question完整8方案；每个方案都扫加法与Slerp。
加法rho与Slerp alpha均[0.01,0.1,0.3,0.5]；全部单层，深度50%/75%/100%，共192个非零候选及共用0基线。
28层落地13/20/27，32层15/23/31，40层19/29/39；沿用十进制四舍五入、0.5向上。
100%为最后block输出、最终norm之前，提取与注入同点，不混用归一化后的hidden_states[-1]。
新选择覆盖下方历史40%/50%/60%/75%、rho0.03及多层提案；下一步完整协议docs/10_overlap_parameter_search.md。
本次只写计划与清单，未开搜索/GPU、未改旧配置/生成训练代码。当前主项目仅加法，Slerp尚待接入和验收。
已有13/20激活按checkpoint/文本/过滤身份复用；27需补三角色提取和尺度，所有组共用实际有效交集。
先做S0零更新固定前缀overlap筛选，默认前3条对比图；结果供用户检查，不能把选定网格当作自动开训练的指令。
旧5条重放失败未解决、正式更新0；新算子及最后层路径必须验收，不能沿用旧加法结果冒充。
参数清单、CAA末层扫描源码及哈希见audit/selected_overlap_grid_20261113/。

## Latest Clarification — Relative Layer Depths Across Models

2026-11-13用户要求层候选跨模型按百分比换算。主候选40%/50%/60%/75%，组合见docs/10_overlap_parameter_search.md。
约定：十进制四舍五入p×L得到1-based block序号，再减1并限制到0..L−1；0.5向上，不用Python银行家舍入。
28层对应[10,13,16,20]，32层[12,15,18,23]，40层[15,19,23,29]。
配置支持steering_method.layer_depths，读取学生config.json实际层数，落地后仍为原Method/bank绝对索引格式。
同时给layers时必须匹配；取整碰撞拒绝，不能去重后仍称原多层组合。相同比例仅是统一搜索坐标，各模型独立选优。
CPU73 passed/2 warnings；证据audit/relative_layer_depths_20261113/。未开新搜索/GPU，旧配置与产物不改。

## Latest Task — Default Comparison Plots and Eight-Way Search Proposal

2026-11-13用户要求后续默认画逐token对比图，并准备T−S/T−R × 全位置/overlap门控 × 全局/逐题的8格搜索，
先给注入强度及层集合候选由用户挑选。本次没有启动新搜索或恢复失败训练。
绘图默认comparison，正式rollout完成后、评分/训练前自动CPU画原清单前3条，恢复缓存时同样补图。
蓝线为保留历史的当前步无注入probe、橙线为正式策略、红点为实际mask，下方为差值；不是全程无SV独立生成。
新HF批量/参考job保存record_comparison_probe；全位置额外无梯度probe与正式KV隔离，旧job/断点协议不变。
历史vLLM无probe时明确缺少配对数据，不能伪造；拟议8格统一后端。新增全位置记录路径仍须GPU验收。
CPU67 passed/2 warnings，真实旧3条用无comparison参数的默认CLI出图，原回答hash不变；无新GPU工作。
候选rho=[0.01,0.03,0.1,0.3]（共同0基线）；0-based层集合=[[10],[13],[16],[20],[10,16],[13,20],[10,13,20]]。
尚未冻结网格，详见docs/10_overlap_parameter_search.md。优先中层并复用现有55题公共激活；
论文Fig2/附录B/表8仅作为候选先验，SFT的alpha不能直接当作本项目rho。
审计audit/default_comparison_search_proposal_20261113/。原5条重放失败、正式更新0的状态保持事实不变。

## Latest Overlap Reading — 2026-11-13

对当前T−R/overlap/当前逐题方向的首批256条轨迹，CPU读取实际保存的probe/正式策略/教师Top16，
共55个不同训练题、337427个预测位置。保持同一实际受控历史，只比较当前步无注入probe与正式策略：
全位置token加权Overlap@16为58.461256%→58.474685%，+0.013429个百分点；
50529个实际注入位置为28.932766%→29.022443%，+0.089676个百分点，实际注入率14.974795%。
题目等权增量+0.011300个百分点（47题升、7题降、1题不变），这只是即时分布变化。
其中5条轨迹仍未通过既定学习重放检查，本读数暂不能作为已验收的有效性结论；正式更新仍0。
前置T−S/all/S0-global固定前缀双层候选的select/confirm题目等权增量分别−0.051164/−0.041310个百分点，
该前置扫描未显示正向增强。不同方法/前缀/聚合方式分别报告，不能把即时增量称为撤去hook后的学习收益。
证据audit/overlap_answer_20261113.json和同名Markdown；本次未启动GPU、未更新实验权重。

逐token图已按原清单前3条回答000_0/000_1/000_2生成（同题ot3-10473、seed42/43/44）。
蓝线为门控前probe Overlap@16，红色散点为实际mask，灰虚线为阈值约0.375；逐点核验阈值与mask一致。
输出runs/full_scheme_20261112/learning_selection/tr_l20_6_rho0p01/analysis/first3_token_overlap/，含PNG/SVG/CSV、3页PDF和来源hash。
绘图脚本scripts/plot_token_overlap_gates.py，只读CPU处理，不平滑、不截断、不重新推理。

对比版逐token图已另存analysis/first3_token_overlap_comparison/（同一候选目录下），保留原单线图。
蓝线为保留历史、当前步无注入probe；橙线为实际门控后overlap，下方为两者差值（百分点），散点仍标原始实际mask。
前3条000_0/000_1/000_2均值分别49.038462→49.038462%、66.411482→66.419329%、66.948198→66.948198%。
第1条0处overlap变化、第2条49处、第3条2处且平均抵消。CSV逐点与原图数据一致，3页PDF/PNG/SVG已验。
这不等于全程无SV独立生成；当前候选只有steer组，尚无与这3条配套的全程无SV生成对照。
本次只读CPU画图，没有新增模型推理或更新。

## Latest Continuation Review — 2026-11-13

接续“明确 Steering_OPD 实验下一步”时核对：首个T−R候选第1步rollout在10:10:25完成256/256条，
GPU子进程墙钟2:48:30，教师评分256/256。10:11:00训练在第73条（018_0）重放检查失败退出；
只有此前72条临时梯度累积，尚未optimizer.step/提交checkpoint，正式更新仍0，50步面板未启动。
当前panel_status与候选campaign为failed_resumable，监督器不存活；顶层campaign的running是旧记录。
本次仅新增只读GPU诊断和文档：全部256条复核，251通过、5条失败（018_0/024_1/029_0/033_3/062_2）。
切换注意力内核、显式mask、缓存末步与FP32重放均未使失败集合全部通过，根因仍未确认；
不能仅凭平均KL小归因于无害数值误差。既定mean0.005/max0.03阈值未放宽，生成/训练代码和配置未修改，
原回答/评分/方向保留，未重启已失败训练；恢复前须解决一致性问题并验收，不应直接原命令重复失败。
实际在线门控后端为hf_batched_causal_gate，先probe/教师分布，再决定正式注入，未接入EasySteer。
证据audit/replay_failure_20261113/README.md、status.json和scan/report.json；本节覆盖下方09:52/09:57运行快照。

## Current Full-Panel Implementation — 2026-11-13

2026-11-13 09:52进度核对：已取得GPU2，precheck零更新select/confirm、多层扫描、T/R/S公共池、Base-dev选型评测完成。
首个T−R候选（0-based层20/6、rho0.01）的生成/完整支持重放/临时反向验收已通过，进入10步选型。
09:51首步rollout已提交225/256条，训练更新0；正式50步面板尚未启动。日志出现长回答与显存压力拆批/OOM恢复，速度仍需优化。
当前支持完整回答复用、未完成批量回答每32token保存mask/RNG并重建KV、更新级模型/optimizer/RNG原子恢复；
中断未提交训练更新需整步重做。用户本次仅询问停止后能否恢复，未要求停止。
证据audit/full_scheme_2026-11-12/progress_20261113_0952.json；下文“未取得GPU”是更早快照。
09:57耗时审计：首次等卡5:00:35；已完成GPU子进程墙钟21:27；当前首步rollout累计2:35:38、233/256条，
日志累计15次OOM拆批及15次预防性显存压力拆批。训练更新仍0；不能报告平均完整OPD步耗时。
逐阶段开始/结束/等待分离记录audit/full_scheme_2026-11-12/timing_20261113_0957.md及同名JSON。

最新完整队列协议见docs/08_full_scheme_execution.md。本轮授权恢复运行，旧暂停记录为历史。
已经补入精确批量probe/正式KV分支、显存压力拆分/结束请求重新组批、历史匹配随机、位置bin匹配、
共同文本/过滤/激活池、正确性/模板长度/独立题集、真实10步裸学生dev选参、分阶段注入、
Compass式(9)sampled及式(10)有限候选版本、EOPD sampled/Top16-surrogate、多seed/MATH500/机制续写编排。
新GPU路径仍必须验收；EasySteer在线交互尚未集成，不把项目批量HF称作EasySteer。
初始公共池复用precheck T/S；只补一次R读取及token缓存。当前学生刷新不能复用旧激活。
逐题零差分对应零干预，不放大或回退全局；随机向量按题独立，三种方向初始公共池共用过滤和尺度。
当前自答使用固定每题seed，质量问题最多3次重采样；超预算明确记录BootstrapQualityError，不改成S0文本。
固定S0文本/当前学生读取另列对照。完整任务的未完成更新保留optimizer/RNG，已完成10步选型保留权重证据并释放optimizer。
主要裸学生/主方案保留10/25/50步模型，其他方法保留最终权重。角色/词表与原100候选集不变；所有主要方法
使用同一三角色/激活共同有效交集，单独报告实际数量，不能称过滤后仍有100训练题。
CPU验证128 passed/1 skipped；79份Python源码SHA256复核一致。新GPU验收/训练尚未开始，
两监督器存活、precheck等空闲卡。汇总audit/full_scheme_2026-11-12/README.md；不能将排队写成完成。
存储复核：整盘7.0T剩余3.4T；项目路径XFS project21003/P继承，目录额度约1.3T。
首次约122GiB，2026-11-13 02:02快照约116GiB可用；XFS配额明细读取无权限，可写空间以目录statvfs为准。
不是换主机或路径。GPU子阶段等40GiB可写额度；不调系统配额，不删旧run，不在项目外写输出。
证据audit/full_scheme_2026-11-12/storage_quota_latest.json及input_fingerprints.json。

## Latest Authorization — Complete Current-Pair Experiment Panel

2026-11-12用户明确要求“现在帮我实验完整方案，包括前面提到的未实现的分支等，然后保证逻辑的前提尽量提高效率”。
本次授权恢复代码实现、CPU/GPU验收及分阶段实验，覆盖下文历史GPU暂停/第0节结果后必须再次确认的限制；
旧run、配置、断点和数据保留，使用runs/full_scheme_20261112及独立配置，不接续旧10步曲线。
先执行T−S/all/S0-global零更新层/强度诊断，再运行现有Qwen配对的主线和对照；
SFT/跨tokenizer是否也恢复已异步询问，未答前先推进现有配对，不自动新增模型下载。
仍优先一张真正空闲卡、最多两张；逐GPU查询计算进程，禁止停止其它项目进程。
首次检查8张卡均有进程且约63–65GB驻留；独立precheck已进入空闲卡等待队列，未获得GPU。
新增精确门控批量PyTorch路径，按可用显存估算批量，probe当前slot与正式KV合并需验收；
不能把它称为EasySteer实现，EasySteer在线交互尚未集成。
随机滚动匹配只使用过去shadow gate记录，固定前缀才允许事后精确次数匹配。
不同rollout/learning注入策略用显式candidate_rkl状态重放目标，不能沿用on-policy声明。
Compass有限候选适配和EOPD Top16-surrogate适配分别命名，未声称原论文完整复现。

## Latest Clarification — EasySteer Position Masks and Online Gates

2026-11-12用户提出门控生成也优先考虑EasySteer，避免HF逐请求生成吞吐瓶颈。
官方固定版本支持SteeringSpec/ApplySpec及逐请求generation_positions；已知mask的注入不要求HF。
当前vLLM0.32自定义worker已支持全位置SV；EasySteer是另一套vLLM fork API，未迁移环境或启动GPU。
公开配置在请求admission时确定，修改default只影响新请求；没有查到可直接调用的外部教师逐步overlap门控接口。
固定前缀的overlap可转为该前缀的诊断mask/已知位置对照；不能当成干预改变轨迹后仍精确的在线mask。
严格在线方案需在实际受控历史上先做当前步无注入probe及教师评分，再按gate选择正式前向并隔离probe KV；
迁移EasySteer仍需逐步调度/KV分支集成，不能仅传旧index列表后声称等价。HF只是当前参考实现。
证据audit/vllm_steering_gate_docs_2026-11-12/及docs/05_method_branches.md第4.1节。
本次仅补文档/源码证据；未改训练代码/配置，第一步T−S/all/S0-global与实验暂停状态保持原样。

## Latest Review — Core Branches Are Not the Complete Control Panel

2026-11-12逐项核对完整方案，清单docs/07_control_implementation_matrix.md及audit/control_coverage_review_2026-11-12/coverage.json。
核心T−S/T−R/R−S、all/overlap/entropy/late、global/per_question/initial/current/周期、方向正负/随机/错题开关已有CPU路径。
固定概率random不是等注入率/次数或位置匹配随机；这两个必要对照未实现。阈值校准、共同文本/过滤/尺度面板、
模板/共同正确集合、共同后端消融、强基线/多训练seed/等计算预算等尚未齐全，不能用“分支已接入”概括为全部对照完成。
新campaign rollout-only/loss-only/末期撤去未实现，当前明确拒绝这些标签；旧loss-only工具不能代替。
本次修正混合mask诊断日志和旧缓存元数据（无需重算概率），bank复用增加方法/提取checkpoint/step核验。
本次99项CPU回归通过；所有新GPU/训练继续暂停，第一步配置/data/0更新范围保持原样。SFT和跨家族不自动启动。

## Latest Task — Preserve Token-Level Evidence Before Experiments

2026-11-12用户要求核对完整方案注入位置，并明确记下逐token对照的保存需求，避免以后因缺日志重跑。
逐token指g_t：全位置/overlap/独立熵差/因果后段/随机注入，轨迹内方向固定，不是每token重提SV。
本次补保存接口与CPU验证，不启动GPU；第一步仍T−S/all/S0-global、0次OPD，同题100，结果后停供检查。
只调层/层数/强度，不再自动安排旧/新题集对照；历史日志保留。
提取缓存需保存学生分词、offset/正文mask、prompt边界状态、各层激活及checkpoint/hash；
诊断/rollout记录逐位置Top16、全支持熵、overlap/共享质量、实际mask、方向版本。
缓存不替代新门控生成/训练轨迹或更新后学生的新计算；不能把事后指标当作采样前probe。
证据audit/token_recording_2026-11-12/；92项CPU检查通过，GPU/正式模型有效性另验。
完整稿新增9.7留档要求，docs/06_token_level_records.md记录字段。零更新提取默认BF16 token激活缓存，
含最后prompt/正文/预测位置映射；固定前缀及rollout留Top16/完整支持熵/mask/版本，评分清理不删token_statistics。
当前无新正式模型缓存或结果；只准备接口，所有campaign仍paused_by_user，第一步不改方法轴。
旧80项实现记录与“完整h[t]未保存”均为历史时点；最新以本节、完整稿9.7和token_recording审计为准。

## Latest Task — Implement Method Branches; Keep First Experiment Fixed

2026-11-12最新文档纠正：提取与未来OPD直接共用原100题，不再安排数据是否重合的原因验证或旧/新题集对照。
当前只需调层位置、注入层数与强度，仍0次OPD更新，overlap结果后先交用户检查；本次只改Markdown。

2026-11-12最新用户授权实现新计划的代码分支，第一步实验仍T−S、每个有效回答预测位置注入、S0全局均值。
本次授权覆盖下文历史“只改Markdown”限制，允许代码/独立配置/CPU测试；不启动GPU或恢复训练。
同时保留先做0次OPD更新的独立overlap诊断、结果后人工检查点，不用旧campaign代替零更新入口。
T−R/R−S、门控、逐题/刷新作为可选分支，不自动切换第一步设置。旧配置/断点/数据与Base结果保留。
实现与验收记录audit/method_branches_2026-11-12/，使用说明docs/05_method_branches.md。
代码已接入T−S/T−R/R−S、global/per_question、initial/current/周期、all/overlap/entropy/random/late，
逐请求bank/vector hash、实际mask重放和KV probe隔离；门控面板普通组匹配HF后端/采样器。
全位置主路径继续vLLM；逐题vLLM bank用eager decode；门控参考后端按请求缓存执行，吞吐未测。
独立run_pre_opd_overlap_tuning.py默认CPU准备，已冻结16/16；输出表后awaiting_user_review，绝不自动OPD。
configs/pre_opd_overlap_tuning.json显式0更新；训练入口拒绝该配置。原v2配置字节不变，第一步仍T−S/all/S0-global。
80项CPU语义/恢复/调度检查通过；所有新GPU验收和性能测量未运行，新文本/SV/overlap无实验结果。
旧全计划审计记录的是实现前缺口，最新代码状态以本节与本轮审计为准。

2026-11-12逐token记录核对：提取器仅持久化每题/角色/层的正文池化均值与逐token标量范数，
不保存完整h[t]。schema2 bank即使global实验也保留per_question均值差分。HF门控显式保存逐token mask，
全位置vLLM按规则重建；Top16/overlap日志不等于token SV缓存。详情docs/05_method_branches.md第6节。
当前新提取/评分尚未运行；本次仅核对/澄清Markdown，无代码修改、无GPU或大缓存生成。

## Historical Task — Pre-OPD Overlap Tuning; Markdown Only

2026-11-12最新指令：用户因暂未观察到效果、尤其steer没有相对普通OPD的overlap增量而停止实验。
**当前只修改Markdown，不改代码/配置/数据/断点，不运行GPU，不恢复旧实验或v2。**
下一步先冻结原始Base做训练前steering调参，完成固定前缀overlap表后停止供用户检查，
收到明确下一步指令后才考虑短程OPD，不能自动继续50步campaign。

当前协议`docs/04_pre_opd_overlap_tuning.md`：所有候选OPD更新数=0，参考生成/无梯度提取/教师评分
不算参数更新；不做backward/optimizer.step，也不执行会真实更新的旧acceptance。
旧10步报告是训练后无hook结果：Base59.29321554%、普通59.49203339%、steer59.47657943%，
steer减普通−0.01545396个百分点；它没有测量S0带hook的即时效果。当前固定同题数据，直接调参。

先T−S、S0全局固定、全回答预测位置注入，提取池与未来OPD共用原100题，dev/test独立。
先扫单层位置6/10/13/16/20（28层、0-based）及rho0.01/0.03/0.1与0基线，再视结果比较两层。
每层在本层提取方向/尺度，多层匹配名义总强度并记录实际扰动；不要混淆层位置与层数。
原32题Base固定dev前缀预先16题select/16题confirm，主表为同一S0有/无hook的配对DeltaOverlap，
逐题等权和token加权分开，附共享质量、区间、正/负/随机方向及alpha0/移除恢复检查；AIME不选型。
当前仅为文档提案，网格、同题文本和新方向均未运行。旧layer13/rho0.03不是调参结论。

即时overlap、各组自由训练轨迹overlap、训练后无hook学习overlap分开；即时提高不等于参数学习。
S0未更新时去hook应恢复原值，本阶段不要求撤去hook还能提升。正差值也不自动开训，无增益照实报告并停下。
现有campaign先做普通OPD第1步、常规diagnostic默认无hook，不能用--resume-paused充作0更新诊断。
后续需独立仅诊断入口/结果退出；本次不实现。T−R、在线逐题、动态门控、SFT和跨家族均留后续。
修改前快照与校验`audit/pre_opd_overlap_plan_2026-11-12/`；下文旧授权/准备记录不覆盖最新范围。

## Historical Preparation — Experiments Stopped; Rollout Top16 and Shared Data

2026-11-12最新用户要求停止实验、rollout直接保存完整Top16再交教师评分，SV提取与OPD共用题集。
**当前只改代码、准备数据与CPU验收，不恢复任何GPU实验。此暂停覆盖下文旧续跑授权。**
旧campaign已标记paused_by_user，两组均10/50步，Base与普通10步AIME完整，steer10步266/480保留。
已停止核对身份的本项目supervisor/生成进程组，未终止外部进程；证据audit/user_stop_top16_20261112_185502/。
新接口与同题方向用独立配置/数据/run版本，不能把旧steer断点换新方向后当成同一曲线。
SV仍为S0提取的全局固定T−S；不自动扩展T−R、在线逐题或门控。
训练轨迹overlap与关闭steer的固定dev前缀主overlap分别报告。GPU概率/梯度验收须在用户恢复后完成。
新配置configs/figure8_shared_top16_v2.json；新数据/run均figure8_shared_top16_v2_20261112。
新100题direction=opd精确复用原题ID/内容/编码，原dev128/AIME30不变；原Base评测协议一致可复用。
新rollout逐位置raw Top16存为int32/FP32二进制并hash绑定回答，先概率后回答原子提交。
新路径跳过HF学生无梯度评分，教师同一次前向评分/自身Top16/overlap；学习用保存概率冻结奖励。
跨后端Top16条件及Top16加其余事件双向KL校验不等于完整支持KL，后者GPU启动验收仍必需。
入口对paused_by_user或execution_hold拒绝默认续跑，明确恢复授权后才可用--resume-paused。
实现与CPU证据audit/rollout_top16_shared_data_2026-11-12/；48项CPU检查通过，旧配置/断点不修改，不混接方向。
已按相同协议CPU复制Base评测/固定前缀/教师评分，1031文件SHA256一致；新run的initial_evaluation_reuse.json留来源。
新SV文本/方向尚未生成，新两组训练均0步；没有新GPU工作，不用旧验收代替新Top16接口GPU验收。

2026-11-12完整方案源码复核：第16节2–2–2尚未实现；当前仍T−S/全位置/S0全局固定。
T/R角色接入、当前学生逐题重提、每请求向量、采样前overlap probe/教师交互、KV隔离与记录mask重放均待实现。
现有rollout Top16是实际受控策略分布，事后overlap诊断不能充当门控前probe。
本轮仅检查/记录，无训练逻辑修改，无GPU启动；审计audit/full_plan_implementation_review_2026-11-12.md。

## Historical Task — Default vLLM Scheduling and Resume

2026-11-12前向次数澄清：额外HF无梯度学生评分用于获取Student Top16候选及旧概率/重放校验，
普通/steer均有，非SV算法必需。原Rethinking fork同样在教师评分前调用compute_log_prob。
两段式可行但尚未实现rollout保存Top16或学习/教师融合；当前流程不变，证据`audit/student_forward_count_2026-11-12.md`。

2026-11-12第10步初步结果：同32个dev问题/初始Base固定前缀、35994个预测位置、全部无hook，
Base Overlap@16=59.29321554%，普通OPD10步=59.49203339%，steer OPD10步=59.47657943%。
steer相对普通为−0.01545396个百分点，当前没有观察到overlap提升，不能声称steering有效；
单seed早期点微小差异，未做题目级显著性检验。accuracy尚未齐，原实验继续。
用户查询后临时用空闲GPU7计算原定两组指标并已释放；正式GPU2评测未中断，后续复用结果。
证据`audit/step10_overlap_2026-11-12/README.md`；顶层报告现可显示overlap而accuracy待评。

2026-11-12用户明确授权移除所有自有生成入口的num_seq上限、改完继续实验；覆盖下面旧文档同步任务的
“只改Markdown”范围。当前主生成、campaign、验收、性能诊断及两个旧辅助入口均已移除人为序列上限，
CUDA Graph尺寸改由vLLM自动选择，全量请求一次提交。原科学配置、训练batch64/n4、评测n16保持原样。
vLLM实测自动解析1024、自动图最大512；24项CPU回归与普通/steer各256条真实分布验收通过。
两组第10步断点兼容性检查通过，当前配置删除旧字段但沿用历史冻结config hash校验，不改权重/Adam/RNG。
仅停止本项目已验证身份的supervisor/生成进程组，复用261份字节未变的评测答案，剩余219份续跑。
新supervisor 1869755，日志`runs/figure8_first_run_20261112/supervisor_010_default_scheduler_20261112.log`，
正式仅用GPU2；验证短时最多两张卡，未终止外部进程。实时状态仍查实际PID与产物。
证据：`audit/remove_sequence_caps_2026-11-12/README.md`及运行目录runtime_updates/20261112_default_scheduler/。
历史运行JSON和第三方默认实现保留；旧参数名称只用于元数据兼容/日志读取，不再施加运行上限。

## Previous Documentation Task — Extraction Timing and Markdown Synchronization

2026-11-12最新任务：按用户再次修订的`docs/Steering_OPD_完整研究方案.md`同步其它Markdown，
新增“什么时候提取steer”。**本次只改文档，不改现有代码、JSON/CSV、数据划分或运行产物，
不启动、终止、迁移已有任务。** 既有首轮执行授权仍有效；不能用旧暂停字段推翻它。
新版面板不是本次文档同步所启动的实验。

完整方案主线为**2–2–2：T−R差分＋overlap单门控＋当前学生在线逐题提取**。
每轮fresh rollout前，用同一当前学生checkpoint读取当前OPD题的T/R完整参考回答，
关闭steering与梯度，提取本题方向。方向在同批同题的多个rollout及其学习重放中固定；
学生更新后再遇同题时重新读取。不是每生成一个token就重提。
T/R文本可提前生成并逐题缓存；缓存参考文本不等于缓存学生激活/向量。

提取粒度与刷新频率分开：同题同文本同S0的“全局均值vs逐题固定缓存”检验粒度，
“逐题固定缓存vs当前学生每轮重提”检验刷新。独立数据全局为迁移对照；周期刷新低优先级。
主方案提取与OPD同题，dev/test独立；旧“方向题必须与OPD分离”只适用于独立全局对照。
保存question_id、参考文本/hash、提取checkpoint/步数、每题原始差分/范数、实际向量与
vector_id，以及本轮rollout/门控/概率绑定；恢复时不用新提取向量替换旧轨迹的训练条件。

现有首轮仍是普通OPD/T−S全位置、初始学生全局固定方向，100题OPD/256题独立方向/128题dev，
各50步。配置`configs/figure8_first_run.json`、已有数据/文本/向量/断点均保留，不能冒充新版2–2–2。
当前运行入口是`scripts/run_figure8_campaign.py`；原fork预览`launch_pilot.py`不支持新版动态提取。
T−R在线逐题与overlap动态门控尚待实现验收；本地SFT、C1及其它扩展不自动加入首轮。

修改前10份文档及用户输入原文保存于`audit/extraction_timing_sync_2026-11-12/`。
本轮不改用户核心稿，不把其文献阅读声明全部继承为已核对证据；SV Training C.2/表8本次局部复核，
历史新颖性正文保留，新版不宣称完成完整新颖性审计。原Student Top-K16反向KL基础配方继续沿用，
不新增KL研究轴，不将原fork奖励/PPO surrogate与直接子集KL求导混称为完全相同梯度。

## Existing First-Run Authorization and Historical Recovery Record

2026-11-12 max_num_seqs复核：用户要求优先默认或较高上限、先查证汇报。
安装vLLM0.32.0与官方tag源码一致；本机71.694 GiB、非A100、离线LLM的默认max_num_seqs为1024，
默认max_num_batched_tokens为16384。满32K容量17.19条不能作为max_num_seqs硬上限，之前压8/16过保守。
动态KV分配允许更高上限，频繁抢占重算时再按完整吞吐调低。候选为默认1024，或训练256/评测512；
尚未做高上限GPU吞吐对照，不能声称已最快。改动还需解除绑定的小CUDA Graph尺寸，避免隐藏限制。
本次仅查证，正在运行配置未改；证据`audit/vllm_seq_limits_2026-11-12/README.md`。

2026-11-12 15:55性能排序：两组已到10步，普通组第10步完整评测进行中。GPU7短时profiling已结束，
正式仍只用GPU2；累计最多两张卡、没有触碰外部进程或改变科学协议。
Base截断输出贡献94.79%的生成token，多数尾部明显重复；16K/16路decode profile的attention kernel占85.12%，
采样相关kernel仅0.67%。长上下文attention/生成量优先于prefix cache、采样器或仅更换steer组件。
两次正式campaign停止至恢复共5小时42分（含修复时间）。证据见`audit/performance_ranking_2026-11-12/README.md`。
词表文字纠正：Base tokenizer.json的151665不是运行时全部词元数；官方tokenizer_config另含四个控制token，
AutoTokenizer两端实际均151669。实验继续冻结151665输出子集，不再称它为运行时全部共同已定义支持，
也不称那四个token仅教师定义；未改变原配置/分母/断点hash。

2026-11-12 vLLM/EasySteer复核：当前入口一次enqueue全部pending（每步256条/每次评测480条），
8/16/64是同时执行上限，逐条落盘不等于串行生成；job中batch_size=32没有拆批作用。
尚未启用prefix caching、chunked prefill和编译，不能声称已充分优化。
EasySteer官方0.29/V2的direct可映射末prompt状态+全部generation状态；仅有CPU源码确认，
当前仍是0.32/V1自定义hook，未切换后端。本机conda easysteer为旧0.1版，不等于最新版。
其冻结base训练adapter的训练入口不能直接替换当前OPD学生更新。
证据见`audit/vllm_easysteer_review_2026-11-12/README.md`。另已确认steer第8步完整概率检查及更新完成。

2026-11-12 14:34效率/恢复复核：普通组8步、steer组7步已提交；13:30短回答采样log-prob
检查失败后停止。原未提交steer第8步整批归档，现已单GPU从断点重跑，保留题目/seed/向量，
为极短回答增加来自实际rollout的完整分布验收，未直接放宽采样误差阈值。
从第9步起新rollout并发64、新AIME并发16，独立`runtime.json`记录实际运行参数；原科学配置不改。
两组各96条真实模型分布检查及17项CPU测试通过。长上下文小基准吞吐仅提升17.47%，
不声称完整AIME翻倍；原Base评测94.79%的生成token来自191条撞上长度上限的回答。
证据见`runs/figure8_first_run_20261112/recovery_20261112_replay/README.md`；实时进度仍查PID/断点。
完整评测频率缩减已询问用户，收到答复前仍保留原日程，不因提速自动改变科学协议。

以下保留原有首轮授权及11:54时点恢复记录；不是当前实时进度，查状态仍核对真实PID与产物。


2026-11-12 11:54恢复记录：用户查询进度时，两组均为1/50步，方向提取完成126个有效配对。
初始Base AIME2025共480条评测已完成，avg@16=5/480=1.0417%。07:13后因matplotlib缺失
导致CPU绘图异常退出，旧状态文件误留running。已在项目venv补绘图依赖，将绘图/汇报失败
与训练解耦，统一记录CPU/GPU异常退出，状态查询核对真实PID及启动时刻；4项回归测试通过。
已从原checkpoint在单张空闲GPU2恢复，复用全部已完成文本/评分/评测，不改实验配置或重做第0步评测。
恢复证据见`runs/figure8_first_run_20261112/recovery_20261112_plot_failure/`；未得到steer收益结论。

2026-11-12最新执行授权：用户已要求开始写代码并直接运行首轮普通OPD/T−S全位置steer两组。
该授权覆盖此前本轮暂停；C1、SFT及其它扩展不自动恢复。优先单张空闲卡，最多两张，
每次GPU任务先检查显存/利用率/compute PID，绝不终止本项目之外的进程。
实现需保存可恢复的数据划分、校准文本、方向、rollout/评分进度、模型/优化器/RNG状态；
加速以正确性验收为前提，可用vLLM或EasySteer，不能降级为仅loss注入。
首轮具体运行配置和实际验收/进度以新campaign记录为准；旧JSON的暂停字段不覆盖此次授权。

首轮实现记录：`configs/figure8_first_run.json`与`runs/figure8_first_run_20261112/IMPLEMENTATION.md`。
使用新单GPU HF/vLLM入口，不直接执行历史fork shell。首次原始head采样的256条中出现2条
无tokenizer定义ID，未做参数更新即归档。正式两组统一151665个共同有效词元的生成/评分/训练
支持；该归一化修正已记录，数据split不变。生成/评分按条持久化，参数更新原子保存optimizer/RNG。
实际进度通过`scripts/figure8_status.py`查看，不把验收更新称正式实验收益。

首轮成功标准：相较同条件传统OPD，关闭steer后的相同固定前缀Overlap@16提高，
且相同无hook评测协议下accuracy提高。用户自行以论文SFT数字作近似参照，不要求本地SFT组或
精确复现SFT成绩。初始评测和实现验收不是方法有效性证据，正式训练收益尚未建立。

## Research Mission and Authoritative Documents

当前先完成0更新overlap诊断与用户检查点，以下保留后续完整研究目标。

核心问题：**当前学生从当前训练题的强弱教师文本中提取逐题方向，再按师生分歧干预，
能否改善OPD并使收益在撤去参考条件后保留？**

研究三个问题：减谁、何处注入、什么时候/按什么粒度提取。以2–2–2为中心逐项改变，
保留T−S/R−S、全位置/独立熵差/随机/后段、独立全局/同数据全局/逐题缓存等对照。
不将全九格或全部三维组合设为起步条件，有交互证据再补交叉；不预设在线重提优于便宜缓存。

- 用户逻辑来源：`docs/Steering_OPD_完整研究方案.md`。
- 方法、数据关系与比较：`docs/01_research_plan.md`。
- 固定论文、原配方及阅读边界：`docs/02_paper_settings.md`。
- 当前执行范围、逐题存储与恢复验收：`docs/03_execution_plan.md`。
- 历史新颖性证据及新版提示：`docs/09_novelty_audit_cross_family_steering_opd.md`。
- 导航：`docs/README.md`。

有实质进展、失败或用户纠正时更新本文件和对应主文档。以日志、checkpoint、process/campaign
及真实进程为证；计划、启动、即时overlap和诊断完成不能写成方法有效。

### Model and Data Selection

- 当前首轮T=Qwen3-4B Non-thinking→真正Qwen3-1.7B-Base，对应Rethinking §5.1配对，
  R=Qwen3-0.6B为后续候选。普通Instruct不替代Base；R的模式、revision、后训练路线及能力差待核验。
- 主方案为每道OPD训练题保存T/R参考文本，dev独立选型、测试不提取答案方向。
  独立全局提取集不与OPD题重合；同数据全局与逐题缓存共享OPD题、文本和过滤规则。
- 原30K OpenThoughts3 OPD题不同于200K SFT示范及OpenThoughts-114k；保存原ID、去重与长度规则。
  已有首轮100/256/128划分与126个有效方向配对的历史记录保持原样；有效数以实际产物为准。
- AIME2025共30题仅评测，不能用于方向、阈值、配对或checkpoint选择；MATH-500等后续评测不自动加入。
- C1候选Qwen3-30B-A3B-Instruct-2507 / Qwen3-0.6B→Granite4.1-3B，Compass已有配对；
  C2同教师→Qwen3-4B、C3 Qwen3.5-4B / 0.8B→Qwen3-4B为后续控制，资产/接口待落实。
  JustRL/R1共享谱系对照不作为独立跨家族证据，不随文档更新恢复这些任务。
- 总参数、激活参数、文本主干、可训练参数分别报告；A3B不等于3B总参数，模型文件存在不等于已验收。
- 当前首轮统一冻结151665个输出词元的受限支持，生成/评分/训练/评测分母一致。
  两端AutoTokenizer运行时均151669；配置额外定义的四个控制token及剩余head行均排除于该输出子集。
  不把这个受限子集称为运行时全部共同词元，原始head范围与已定义token范围仍须区分。

### Method Boundaries

1. 两端回答由同一个学生checkpoint读取：主方案为本轮当前学生，固定对照才为S0。
   关闭hook/no_grad读取，正文池化，不相减异模型hidden states；单独T−R通常两次学生prefill。
2. T−R无需学生自身回答；在线T−S/R−S先生成无hook bootstrap，再另采受控OPD轨迹。
   不用尚未生成的受控轨迹构造引导自身的方向；固定S0文本另列变体，bootstrap/读取成本全计。
3. 三差分共享题目、文本、过滤、层、权重及提取者，原始逐题d(T−S)=d(T−R)+d(R−S)。
   单位归一化后不要求等式；全局对照先平均原始差分再归一化，不平均单位方向冒充同协议。
4. 每题本轮向量停止梯度，沿rollout与学习重放固定，不穿过提取过程反传，不在一个更新里偷偷换方向。
   保存提取checkpoint/文本hash与实际向量；同题多rollout复用，同题不同轮保留不同版本。
5. 当前门控为源无SV rollout完成后批量教师评分生成的静态mask，g_static=1[O_src<tau]；
   注入后不动态重判。源版本/hash/输出支持和预测idx绑定，更新学生或源回答变化时刷新。
6. 当前零更新搜索在同源前缀固定mask前向；首回答预测在prompt末状态，后续t−1对齐，屏蔽padding。
   后续同回答learning-only或静态mask二次生成分别定义，未选择时不自动切换；新轨迹须自己的概率与评分。
7. 基础目标沿用Student Top-K16/student_p/token_reward_direct/PPO及token-mean。
   loss支持取实际受控学生Top16，静态门控取无SV源Top16；阶段不同必须显式目标，不能直接沿用同策略声明；冻结候选/教师/向量/门控，理论与源码口径分开。
8. 当前先按0更新的固定前缀即时overlap调层/强度并交用户检查；
   后续学习方案仍须真实短程更新后的无干预dev收益验证，各方向等调参预算。
   熵差仅独立对照，不与overlap做AND/OR；后段不读本条最终长度，静态随机对照可在已知源mask范围离线匹配。
9. 主结果只测无教师/无R/无向量/无门控学生。逐题方向含训练参考信息，同文本及T/R两源基线匹配信息；
   记录参考生成、当前学生重读、bootstrap、源教师批量评分/固定mask前向/必要时二次生成、学习与评测全部成本。
10. 同数据全局vsS0逐题缓存隔离粒度，S0逐题缓存vs当前学生重提隔离刷新。
    逐题本题/其他题向量控制有效；全局均值两侧集合/权重不变时打乱题目配对不改变方向。
11. 不同tokenizer先验收共同动作/span接口；平均log-likelihood不是整词概率，字节门控不是原生Top16。
    纯文本教师只能供方向文本，不能直接提供静态overlap所需的token概率评分；跨家族主张限于实际配对证据。

### Literature Interpretation

SV Training已有初始学生全局/逐题固定及每200步全局刷新，C.2为Qwen2.5-1.5B/GSM8K SFT；
不能把其刷新结果直接外推为本项目OPD结论。其§5.1 GRPO用方向投影修正advantage，
不等同于前10步hidden hook。逐题、刷新和训练时steering本身不作为首次提出。
Compass同动作似然分解不证明不同文本激活差自动消偏。Rethinking §8确有跨家族混杂讨论，
共享质量与overlap集合比例分开；梯度抵消与本项目机制均非已证实。
Entropy-Aware/桥接等候选不继承核心稿全部阅读声明；SCOUT本地固定2609.38360v1来源仍保留。

### TODO — Update with Evidence

- [x] 改为先0更新overlap调参与用户检查点，仅同步Markdown。
- [ ] 后续独立诊断入口及层位置/层数/强度扫描，结果退出，不进入OPD。
- [ ] 诊断执行获准后提供完整select/confirm与逐题数据，等用户明确下一步。

- [x] 同步2–2–2、当前学生逐题提取、训练题参考复用与dev/test独立边界。
- [x] 分清全局/逐题粒度与固定/刷新频率，保留两组桥接比较及本题/他题控制。
- [x] 在文档中定义每题参考、向量、提取checkpoint及rollout/mask的版本化恢复契约。
- [x] 保留已有首轮配置/授权与原RKL配方，不将新方案冒充当前运行，不更改用户核心稿。
- [ ] 新版实施：逐题缓存与重提接口、batch/题目绑定、提取无梯度及断点一致性。
- [ ] 新版实施：源rollout静态mask绑定、固定位置/Slerp/首预测映射与阶段目标验收。
- [ ] 冻结新增面板、参考文本、dev选型与总成本预算，再形成独立执行记录。
- [ ] 根据无干预学习收益逐项消融，在线无增量时收敛缓存/全局版本。
- [ ] 后续强基线、重复seed与外部家族验证，按实际证据限定主张。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：方法分支、零更新入口与逐token留档已实现，92项CPU检查通过；所有实验/GPU继续暂停。
第一步仍T−S、全位置、S0全局均值，同题100池；已CPU冻结16题select/16题confirm，没有新参考文本/SV/overlap结果。
旧v2配置及数据字节保持，新分支GPU验收/吞吐待用户恢复后执行。当前诊断只到结果表退出，不自动进入OPD。
SFT、跨家族及其它研究面板不自动启动；当前实现与边界见docs/05_method_branches.md及audit/method_branches_2026-11-12/。
<!-- CROSS_FAMILY_RUN_STATUS_END -->

## Scope and Startup

The project root is `/home/zhenning/data/working_tmp/steer_opd`. Keep project edits, checkouts, processed datasets, checkpoints, logs, and generated caches inside this directory. Run commands from this root; do not modify neighboring projects.

Start with `README.md` and `docs/README.md`. Read `docs/01_research_plan.md`, `docs/02_paper_settings.md` and `docs/03_execution_plan.md` before implementation. Read `docs/09_novelty_audit_cross_family_steering_opd.md` before novelty claims.

## Environments and Shared Assets

Select an environment for each stage and recheck dependencies before the new experiment:
Add models/datasets or install packets/create env is allowed. But don't delete/modify dataset/models in /home/zhenning/data/huggingface_datasets or /home/zhenning/data/huggingface_models.

```bash
cd /home/zhenning/data/working_tmp/steer_opd
conda activate verl          # OPD / verl work
```

For SFT / LLaMA-Factory work, switch to:

```bash
conda activate llamafactory
```

For Steer may need switch to:

```bash
conda activate easysteer
```


For noninteractive commands, use `conda run -n verl python ...` or `conda run -n llamafactory python ...`. Environment existence does not establish dependency compatibility or training readiness.

Models are stored under `/home/zhenning/data/huggingface_models/<model-name>`, with flat project links under `assets/models/<model-name>`. Datasets are under `/home/zhenning/data/huggingface_datasets`. Treat existing shared assets as read-only inputs. Inspect their actual layout and reuse local assets; do not redownload or relocate them by default. The model download helper now defaults to the shared model directory; `--root` controls project links, dataset downloads, and receipts.

## Project Structure

- `docs/`: Chinese research notes, related work, novelty audit, and experiment plans.
- `experiments/`: planned groups, stage matrix, download manifest, and run-record template.
- `scripts/`: asset-preview and package-validation utilities.
- `papers/`, `ref/`: core/reference PDFs, version manifests, and source snapshots.
- `audit/`: retrieval evidence, source checksums, and validation reports.

Training code is under `src/steer_opd/`, with semantic checks in `tests/test_training_semantics.py` and project launchers under `scripts/`. Read `docs/03_execution_plan.md` for the requirements of the new experiment; model, data, loss and hook behavior must be checked before launch. Check individual `process.json` and `campaign.json` records before claiming a run is complete. Planned configurations are not experimental results.

## Development and Validation

```bash
conda run -n verl python scripts/validate_package.py
conda run -n verl python scripts/download_assets.py --list
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' conda run -n verl python -m pytest -q tests/test_training_semantics.py
```

The validator checks material integrity, links, manifests, and planned groups; it writes `audit/package_validation.json`. Asset commands preview by default. Neither command validates model quality. No formatter, linter, or coverage threshold is configured.

GPU launches must go through `scripts/run_on_idle_gpus.py` or an equivalent fresh `nvidia-smi` check of memory, utilization and compute processes. Never stop processes outside this project. Do not execute the reference launcher's global `ray stop --force`; project runs create independent local Ray sessions and record their PIDs. Shared assets remain read-only, and new data/cache/checkpoint outputs stay inside the project.

## Style and Research Integrity

Use four-space Python indentation and `snake_case` filenames/functions. Follow numbered Markdown names such as `docs/09_novelty_audit_cross_family_steering_opd.md`. Record paper versions, evidence depth, model revisions, and validation commands.

The current design uses matching controlled rollout and learning replay, per-question directions fixed within each rollout/update batch, and overlap-only gating. Extract with the current student before each batch; distinguish reference-text caches from fixed S0 vector caches. Persist question IDs, extraction checkpoint/version, vector artifacts and rollout/mask bindings. Use the pinned Rethinking Student Top-K16 reverse-KL-family surrogate as the shared base recipe. Verify cache isolation, prediction masks and gradient recomputation before training. Distinguish shared-lineage results from independent-family evidence.

## Commits and Reviews

No Git history is currently available. Use concise, imperative commit subjects, for example `docs: clarify steering baselines`. Reviews should describe the problem, changed files, evidence or checks, and limitations. Keep weights, credentials, and large run artifacts out of commits.
