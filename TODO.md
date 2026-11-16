# TODO

当前授权：新增教师空间S−T镜像消融的代码、文档与CPU验证，不启动GPU实验。原100题和1000题的T−S四方案均已完成。OPD、退火和其他参数消融不执行。
实验统一为[ablation计划](docs/ablation.md)：原100题overlap已完成，1000题原四方案已完成，新增教师S−T尚未GPU验收，param与baseline尚未开展。当前无SV基线仅用于overlap配对；后续baseline对比需要独立设计和验证。

当前为T−S固定方向、all/static_overlap × global/per_question四方案，固定强度及共同无SV基线；Slerp0.1、层26、学生冻结、0更新。师生两批100条回答及学生Top16缓存已完成；用户已授权SV采用全样本、2048总上下文及全部保留回答token均值，GPU0、1重新验收已通过，100题激活与两份bank已提交，教师前缀评分、四方案select/冻结confirm及双指标图均已完成；已移除近似ACC，现有师生输出和Top16缓存保留，模板锁保留。


- [x] 按新规范迁移overlap（Top16最大一一前缀兼容匹配）与P_overlap（Top16原始概率最大兼容质量）；两者共用vLLM Top16 ID/logprob，不重新归一化，指标v4及内容hash绑定；删除完整词表与字节投影采集。
- [x] CPU验证文档1/0.84算例、非贪婪/容量争抢、同词表ID退化、别名/EOS/控制/空token、独立枚举最小割及缓存篡改；生成/评分仍用EasySteer/vLLM已有logits，无额外候选续写。
- [x] Top16 v4格式师生GPU验收通过（2026-11-15）；后续100题capture及四方案已通过；1000题使用新验收凭据。
- [x] 两种指标分别记录源/候选/delta、题目等权与token加权汇总，自动输出前3条的两套PNG/SVG/CSV/PDF。
- [x] 固定T−S方向，教师文本与普通学生源rollout构成唯一方向文本对；仅S0 capture正文并构建2份bank。
- [x] 接入EasySteer原生生成、S0指定层capture、T批量评分及固定mask前向，无人工seq上限；保留HF显式参考。
- [x] 使用上游steering配置指纹，补模型内容hash与离线缓存绑定，保留池化均值/范数、位置与mask，不保存逐token隐状态。
- [x] 每请求seed=42，固定Python/NumPy/PyTorch/vLLM与子进程hash seed；保留代码/环境/权重/引擎身份。
- [x] 恢复改为整批阶段/完整候选提交：一次公开generate/capture连续运行，返回后保存和原子提交；未提交整批重算，已完成阶段/候选复用。移除逐请求检查点与自定义调度循环；CPU故障测试覆盖SIGTERM/SIGKILL及提交边界。
- [x] 实时tqdm/console.log、结构化事件/心跳、阶段起止/耗时/ETA与只读进度查看；保留后续训练需要的中间产物。
- [x] 清理旧运行结果、重复Markdown与历史审计快照，保留核心研究稿、必要输入/源码及删除清单。
- [x] 隔离运行环境已安装，CPU导入/Slerp注册/范数/API及固定fork文件hash检查通过；实际GPU与模型架构支持另验。
- [x] 再核对当前4方案队列的后端：生成、提取、T评分和候选评分均走EasySteer/vLLM；CPU Top16匹配和显式HF参考单独说明。
- [x] 集成手动启动sh：开头设置物理GPU池/方案ID/超参数，--check预览、--run验收后执行或恢复、--status查看；指定池外GPU不使用，改参数须新PARAM_NAME。
- [x] 启动修复：spawn避免fork分配pinned-memory阻塞、匹配CUDA工具链、初始化超时与中断清理；HF_PARITY默认0，仅显式设1做HF对照。
- [x] 真实GPU通过学生capture位置/token、全位置/静态mask批量与逐请求、零强度/移除恢复、源采样重放及原生支持检查；学生统一启用batch-invariant，自动调度保留。证据见[启动排查记录](audit/easysteer_startup_20261114/README.md)。
- [x] 精简为当前实验调用链：删除旧pilot/Figure8/full-scheme、OPD训练/动态门控、模型数据下载脚本与旧配置；裁掉共享文件中的无用训练逻辑，保留原始数据/权重、文献和当前验收测试。
- [x] 用户手动准备35B教师权重；已补项目链接，14分片结构/索引与固定config/tokenizer元数据检查通过。未运行下载；完整权重内容哈希在启动时记录。
- [x] 原目录迁移与100条教师文本复用核验：保留原提交身份，restart_top16.json记录旧配置/清理清单，teacher_text_reuse.json绑定SHA和生成合同。学生source后来已提交100条；T评分及四方案已完成并冻结。
- [x] 修复capture只设置host budget、漏设device_budget_bytes的问题；按实际调度token计算设备预算，GPU前检查原select/confirm可用配对，CPU测试143项通过。
- [x] 删除近似ACC活动代码、测试和派生结果；保留师生100条原始输出，补齐学生Top16原提交复用核验，144项CPU测试及包检查通过。
- [x] 按用户最新要求参考sv_training提取SV：全部样本参与，prompt+回答总上下文2048，对全部保留回答token均值，不按正确性/finish_reason/重复过滤；只跳过无有效回答token的样本，原70/30划分不变。
- [ ] EasySteer逐token实际扰动范数留档；当前明确relative_perturbation_available=false。
- [x] 新SV协议通过145项CPU测试及包检查、师生GPU自检；100题/200份回答的激活与两份bank提交成功。
- [x] 完成教师前缀评分、四方案select/冻结confirm及双指标图；2026-11-15 17:01停止，等待用户检查。
- [x] 优化源指标复用及双指标阈值校准：排序count累计计数代替逐阈值全量扫描，保留overlap/P_overlap兼容；255484位置验证两项阈值及100题mask完全一致。
- [ ] 实测时间/显存/吞吐；提前不承诺提速倍数。

[当前配置](configs/ablation.json) · [实验协议](docs/ablation.md) · [实现说明](docs/implementation.md)

- [x] 1000题CPU数据准备及来源留档完成：原100题保留、新增900道去重数学题，冻结700/300；161项CPU测试及包检查通过。
- [x] 完成1000题overlap的GPU验收及四方案700题select/300题confirm计算；结果已交用户分析复核；新输入保留原100题并补充900题，原100题输入和结果不改。
- [ ] param：后续确定参数、候选范围及选型预算，再实现参数实验。
- [ ] 后续param/OPD训练消融：固定a对比100批2a→0，idx 80–99无SV；原四组与新增教师镜像组均为固定强度，不执行题级退火。
- [ ] 后续OPD入口接入真实训练批次退火及恢复留档，验证Slerp逐批alpha传递、idx≥80的无SV尾段及无额外前向；当前overlap不执行题级退火，基础a=0.75时的1.5起点不得直接用于Slerp。
- [ ] baseline：后续确定比较方法、初始化、OPD目标、评测和公平预算，再实施对比。

完整OPD目标/梯度路径、撤去SV后独立能力、刷新/匹配控制、多seed/强基线、层/强度扩展与CompassOPD官方仓库源码复核均留后续，当前不自动执行。原Student Top16学习配方不因新增P_overlap自动改变。

- [x] 按用户要求取消EasySteer/HF逐token隐状态落盘，保留池化均值/范数与SV；清理本轮原始状态并维护提交/结果索引，保留教师生成、2048读取和注入评分参数。

- [x] 新增显式教师S−T方案5–8：教师层29读取学生／自身回答，教师空间作差并注入教师沿固定学生文本评分；原四组与数据／结果保持。
- [x] 接入教师整批capture／独立bank、共同题集、跨tokenizer mask投影、共同无SV基线与教师重复漂移控制；记录原生教师注入率及映射到学生坐标的覆盖率。
- [ ] 新教师路径的双卡GPU验收及八组实际对照：等待用户明确启动指令，当前不运行。

- [x] 教师镜像消融CPU验证：172项测试、包检查及八组／子集--check预览通过；覆盖S−T减法顺序、教师tokenizer／层、共同题集、教师原生mask、共同基线、无干预／零强度、缓存恢复及原生tqdm计数。真实教师双卡GPU验收仍待授权。

- [x] 按输出结构.md接入step0／1／2／3／6，分离RUN_NAME与PARAM_NAME；step4／5仅预留。
- [x] 按组件科学依赖与提交SHA复用上游；新增alpha／层位／门控／教师侧扩展的复用与失效CPU测试。
- [x] 1000题旧产物CPU核验注册为引用，保留原owner、字节及完成标记；不复制大张量，不启动GPU。

- [x] 按用户新要求将已完成100／1000题的16083个文件同盘mv至step结构，验证全部SHA及inode，并更新复用引用和默认输入；旧runs／data／logs移除。
- [x] 删除其它五个旧版／prepared／未完成实验及无用启动记录；保留两批原始数据、回答、SV、概率、指标、图、原始配置和代码。
- [x] 为100／1000题生成独立step6汇总及两批合并CSV／分析，历史完成参数名防覆盖，状态查看支持迁移后的原结果。
