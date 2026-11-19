# Repository Guidelines

## Current authorization and experiment

2026-11-19训练输出位置授权：用户将后续train步骤改回/home/zhenning/data/working_tmp/steer_opd/outputs；该路径解析到/local_nvme/data/zhenning/working_tmp/steer_opd/outputs。configs/ablation.json的train_output_root与启动入口默认值均为outputs，训练输出为outputs/step_4_train/ablation/<模型组合>/<数据集>/shared/<运行名>/，其他阶段路径保持。旧~/steer_opd_outputs仅保留显式指定及已有checkpoint读取能力；不迁移或恢复旧outputs，不改冻结运行配置。训练启动和checkpoint保存继续检查实际目标文件系统的空间与可见配额。更改训练保存位置时使用新的TRAIN_RUN_NAME，保留原冻结请求。此次只改输出默认值、相关文档并做CPU验证，不启动GPU或重启训练。

2026-11-18最新存储授权：新配置采用retention_policy=minimal_v1。SV bank完整提交并校验后，只清理本运行新建且明确标记为minimal的pooled.pt，保留bank／尺度／有效题集／身份及pool退休凭据；跨参数恢复先核验bank，不要求已退休pool重新出现。OPD仅保留当前批请求／回复／Top16／对齐工作区，有限梯度更新及标量提交成功后清理；不长期保存训练及训练内评测原文、逐轨迹监控或Top16历史。用户最新确认模型checkpoint与评测对齐，每16步及末步保存，checkpoint_retention=all_models_latest_state_v1：各保存点的模型分片及配置全部保留，优化器／scheduler／RNG／dataloader状态只留最新完整点；新完整checkpoint和模型SHA清单及latest_complete提交后，核验旧模型并写退休凭据，再只删除旧恢复状态。overlap当前状态实时覆盖，已消费的控制回复及时清理；每步标量、准确率曲线及配置继续保留。覆盖下文历史“池化和训练逐步概率长期保留”及“只留最新模型／每20步保存”约定。此次只改代码、文档并做CPU验证，现有outputs及正在备份的文件保持原字节，不启动GPU，不解除hold。

2026-11-18最新清理授权：用户要求删除全部exploration相关实验，并确认包括OPD诊断和教师基准。已删除exploration源码、outputs各step的exploration分类、对应compar诊断／补充评测及.runtime/diagnostics临时导出；不自动恢复或重启。保留ablation／param、OPD训练本体、共享模型／数据和核心研究稿；固定外部文献已迁至ref/opd_repetition_literature_20261118。清理凭据audit/exploration_cleanup_20261118.json。

2026-11-17最新授权：用户明确启动扩展参数搜索，仅使用物理GPU2／3，已有结果核验复用。固定S−T／教师空间／all／global／方案5，提取与注入同层，教师0-based19／29／39 × Slerp／additive × alpha=.05／.1／.3／.5，共24组，层29的Slerp .05／.1两组引用原完成结果，预计新增22组；旧.2仅保留历史结果。原1000题及700／300划分、上下文2048、seed42、预测位置和FP32池化不变。additive严格为h+alpha×原始global均值差分，系数一次，无额外范数缩放；Slerp仍沿单位方向。层39取final norm前block输出。先CPU检查与逐点GPU验收，再算双指标／图／汇总，0次更新，不做OPD／退火／baseline／独立评测。入口scripts/run_param_search.py，唯一模板parameter_search定义网格，模板hold保持，独立验收配置执行；新搜索身份st_all_global_L19_L29_L39_ops_n1000_seed42_v2，参数名以st_all_global_v2结尾。复用源生成／概率／对齐／同层capture与bank；启用16进程CPU指标与无损gzip。运行中不改冻结Python源码或配置，旧搜索身份与结果不覆盖。此授权取代下文历史GPU／param限制。CPU回归219项及包检查通过，两组旧结果完整性及上游复用已CPU核验。

实验统一命名ablation，唯一执行计划docs/ablation.md，唯一配置configs/ablation.json，手动入口scripts/run_ablation.sh，主编排scripts/run_ablation.py。当前phase=overlap；原100/1000题T−S四组已完成，新增教师S−T待GPU验收；param参数实验与baseline方法对比均为后续计划，不自动执行。新输出遵循输出结构.md：outputs/step_<编号>_<阶段>/<RUN_NAME>/<模型组合>/<训练数据集>/<shared或测试集>/<PARAM_NAME>/，RUN_NAME表示ablation／param／compar；100／1000题输入现均在step0各自PARAM_NAME下，划分与内容不变；原runs／data／logs已移除。完整研究稿保留方法背景，实验进度以ablation计划为准。

2026-11-14用户授权实现两种overlap、双图/留档、EasySteer提取/评分与缓存集成，并清理过期MD和旧实验结果。允许代码/CPU验证及项目隔离环境准备；用户最新明确恢复启动GPU0、1上的4组固定强度SV overlap对照（all/static_overlap × global/per_question及共同无SV基线），到overlap/P_overlap图停止；S0冻结、0更新，不执行OPD或后续消融。教师权重由用户手动下载，代理只做完整性核验，不运行下载逻辑。execution_hold=awaiting_user_approval，不能自动解除。用户纠正退火属于后续真实训练消融，当前overlap保留4组固定强度；其他后续消融暂缓。

固定T=Qwen3.5-35B-A3B、S=Qwen3-4B。Slerp alpha=.1，75%单层26（36 blocks，0-based）；固定T−S方向，all/static_overlap × global/per_question完整4方案，固定强度及共同无SV基线，S0冻结、同源前缀、0更新。仅允许用户本轮明确新增的教师S−T镜像轴，不自动扩展其他方向或层/强度网格。

当前唯一配置模板configs/ablation.json；权威说明docs/ablation.md、docs/代码框架逻辑.md及用户完整研究稿。2026-11-14按用户要求移除旧实验入口/训练与动态门控/下载代码及旧配置，仅保留当前调用链和显式HF数值参考。不为历史兼容重新引入旧代码；后续OPD按独立任务实现。

2026-11-14新增用户手动启动入口scripts/run_ablation.sh：开头定义物理GPU池、方案ID和超参数，默认--check仅CPU预览；只有用户显式--run才在新PARAM_NAME下启动小样本GPU验收，师生两模型通过后生成带验收凭据的独立配置并执行所选方案。原模板hold保持不变，用户本轮明确要求代理启动4组overlap面板范围。默认4方案/Slerp0.1/层26；用户显式修改参数或选择子集才改变运行配置，不自动网格搜索。已有run配置不可改；物理GPU池可在恢复时更换，仍只用指定池内的空闲卡，最多两张。

2026-11-14用户停止初始化阻塞后授权GPU排查。已复现fork在V2 pinned CPU torch.zeros阻塞，统一spawn；采用与torch CUDA版本一致的compiler/header组合，初始化限时600秒。此授权用于诊断和小样本验收，不自动恢复正式面板。启动默认HF_PARITY=0，仅EasySteer师生两模型自检；HF对照须用户显式设1。当前已移除概率统计hook，Top16由公开logprobs接口返回。旧static_panel_v1仅启动验收，源码/协议变化后用新PARAM_NAME，不覆盖旧身份或结果。

本轮此前学生原生GPU小样本检查已通过，教师曾完成双卡初始化/评分但有重复概率漂移；Qwen3.5 GDN不支持batch-invariant，T单次评分SHA冻结供所有学生候选复用，漂移单独记录。TP关闭FlashInfer autotune与跳过图像/视频预热的最新修改尚待双卡复验；GPU2无进程/显存0而利用率持续100%，准入拒绝，不绕过或重置。证据audit/easysteer_startup_20261114/README.md，正式面板/更新仍0。

## Semantics

- 用户后续授权按docs/overlap.md和完整稿§12.2迁移代码及CPU验证，不启动GPU或实验。overlap用Top16最大一一前缀兼容匹配数/16，P_overlap用同一兼容关系的Top16原始概率最大匹配质量，不重新归一化；同词表仅按相同ID。异构普通token完整字节互为前缀，EOS只配终止，其他控制精确匹配，空普通token非万能前缀。明确称跨family适配。
- 指标metric_version=4，matching_rule_version=native_token_prefix_compatibility_v1；仅保留vLLM返回的Top16 ID(int32)/原始logprob(FP32)及观察token logprob，删除完整词表与字节概率采集/RPC，旧缓存/阈值/mask/排名和GPU凭据不可混接。两指标分别保存/统计/画图；一个gate_metric冻结mask，rank_metric显式选定，默认均overlap。
- 共同UTF-8安全span起点比较，向学生span广播；终止EOS目标/不可对齐位置记NaN、明确覆盖率。EOS进入Top16时作为终止候选匹配；Top16以外尾部不参与。
- 普通源rollout后批量教师评分，冻结strict-< mask，候选不重判idx。无在线教师/probe/KV双分支。
- 默认学生方向为T−S：同一S0读取教师回答及自身无SV源rollout，2048-token上下文内保留的回答原始激活逐题池化后作差；global平均原始差分后单位化，逐题零方向零干预。全局/逐题2份bank共用有效交集和学生侧尺度。当前不加载Ref模型；共同span仅由S/T两方确定。
- 首预测是prompt末状态，预测t对应prompt_len−1+t。提取与注入同一block输出，norm之前；capture完整hidden+residual。
- 完整OPD、learning-only/二次生成、刷新和裸学生收益另行定义；固定前缀增量不等于学习收益。

## Runtime and efficiency

默认EasySteer固定前端baecd969/fork7837b07d、wheel0.29；与旧.venv-opd普通vLLM0.32隔离。使用scripts/bootstrap_easysteer.py准备，不改共享环境。
全量pending提交由引擎调度，不设置max_num_seqs或人工CUDA Graph尺寸。HF只作显式参考，不自动降级。
两项指标共用vLLM原生logprobs/prompt_logprobs=16，额外观察token不在Top16时只保留路径似然，不加入指标。候选最多32节点，精确计算容量匹配；无完整词表缓存、字节投影或自定义概率RPC。每位置概率记录约132字节，激活另计。正文capture在内存中获取选定层/位置的BF16状态，返回后求FP32均值，仅保存池化结果/范数和SV，不将逐token隐状态落盘；池化定义不变。
采用上游steering指纹；完整prompt评分与正文capture遵守跳过prefix读取规则。固定mask评分当前关闭prefix caching；普通无SV生成可按支持开启。
模型内容、Top16协议、文本/tokenizer、向量、概率文件和mask身份均须校验。权重首次流式hash，stat未变时复用，不逐token重hash；新checkpoint/源前缀不能套旧概率。
CPU通过不等于GPU验收。实际GPU运行优先一张空闲卡，最多两张；逐GPU计算进程准入，绝不停止其他项目。

## Workspace and checks

输出/缓存/临时安装目录放项目内，共享assets模型/数据只读。旧结果已按用户授权清理，不自动恢复或把旧失败/训练结果混成新配对结果。
保留用户核心研究稿、固定论文/源码与来源清单；不再通过堆叠历史前言记录当前状态。
CPU测试用source scripts/runtime_env.sh后CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q tests。包检查scripts/validate_package.py，不加载GPU模型，并检查本地模块/入口引用完整。

## Reproducibility, resume and logs

用户2026-11-17最新明确：OPD每道题的4条rollout固定分别使用seed42／43／44／45，其他题和后续批次复用这四个值，取消整批递增。在分发worker之前按每题rollout序号分配并核验，重试／resume不依赖worker调度或进程计数器；基础seed、评测每请求及既有overlap／SV每请求仍42。旧formal_v3同题四条回答相同，已标记invalid_sampling；整批递增的formal_v4已按用户纠正停止、0次更新。新formal_v5_seed42to45从初始学生重跑，禁止继承旧训练状态，只用两张空闲卡。子进程启动前固定PYTHONHASHSEED；配置、支持、数值设置、代码/包/权重/引擎身份留档。GPU诊断确认学生批形状数值波动影响零干预/重放检查，因此Qwen3-4B的生成/capture/评分统一启用vLLM batch-invariant，T保留默认，自动调度/图保留；吞吐影响未做正式规模结论，不承诺跨硬件逐bit一致。
恢复粒度固定为engine_call：整批教师参考生成、整批学生源生成、整批S0教师/自身capture、2份bank、每角色整批概率评分、每个完整候选（含baseline）。每次公开generate/capture调用连续运行，返回后统一处理与保存，StageCheckpoint用SHA清单和原子complete.json提交；没有标记的当前整批重算，已提交阶段/候选完整复用。不在引擎内逐请求落盘，不自定义enqueue/step调度，不保存KV/RNG。源/教师概率、池化均值/范数、bank/尺度/mask/指标和代码/权重内容清单保留供后续训练，逐token隐状态不保存，新的前缀/checkpoint仍须重算对应概率与激活。
用原命令加--execute --resume，绝不绕过hold。完整缓存先CPU校验，无需GPU；仍存活的旧监督器接管等待，不重复启动。每阶段单写入锁，Ctrl+C/SIGTERM处理仅影响本项目自有子进程组。
外层步骤/阶段不显示tqdm，只打印带时间的当前阶段、开始/结束、耗时及缓存/失败；EasySteer/vLLM原生加载、生成、评分/capture的tqdm实时保留，仅更新内存计数。15秒心跳与完整JSON事件只写events.jsonl/progress.json，不在终端刷屏；console.log/startup.log保留阶段文本及原生输出。结构化日志记录北京时间起止、排队/活跃/墙钟时间、恢复数、前向完成数、已提交数与ETA；completed不等于committed。首条未完成或异质阶段总ETA未知时不伪造。scripts/show_ablation_progress.py --watch只读查看，不启动实验。

当前默认GPU_IDS=0,1、STOP_AFTER=panel、RUN_NAME=ablation、PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1、CONFIRM_TOP_N=4，只加载S/T两模型。以下旧目录重启记录仅属于已完成的100题运行。用户明确要求移除当前Ref调用并先删除旧运行目录，在同一路径重启，不新增运行目录或归档副本；已停止自有作业并删除旧overlap4_v1及旧launcher日志。不得加载Ref权重/Tokenizer、生成Ref验收/评分或使用旧三方对齐缓存；S/T共同UTF-8 span使用schema2与角色身份留档。保留4组固定强度、双指标、原100题与70/30划分，完成图表后停止，不进入OPD。模板hold保留，独立运行配置承载本次已授权范围。

当前4组固定强度，退火不在overlap阶段交叉，不给原100题分配退火idx。后续100个真实逻辑训练批次idx 0/40/80对应2a/a/0，80–99无SV；Slerp通过scale退火而非预乘payload，2倍起点要求a≤0.5。保留通用计算接口，OPD实际训练调度另行实现。

存储按实际目录的用户/项目配额统计，不按整盘空闲绕过限制；未获授权不迁出项目。2026-11-15用户明确改用Top16原始概率计算P_overlap（无重归一化），已删除完整词表缓存与字节投影采集。用户随后明确授权重新启动；最新指定GPU0、1，保留overlap4_v1原路径。已核验100条教师文本与原提交SHA、checkpoint、请求及生成参数，保留其原job/complete/provenance身份，通过teacher_text_reuse.json在CPU复用。restart_top16.json记录旧配置与清理清单；清理失效启动配置、旧验收及未提交source，使用新Top16 GPU验收凭据后从学生source继续。模板hold保留，不新建运行目录，完成四组图后停止。

Capture预算区分host/device：host按全部保留BF16行及标签，device按引擎实际max_num_batched_tokens、层宽/层数和工作区计算，另留余量。用户最新明确授权参考sv_training提取并重启：全100题，不按正确性、生成截断或重复过滤；prompt+strip后的回答右截断至sv_max_context_len=2048，对prompt之后全部保留回答token（含回答内标记）求均值，只跳过没有回答token的样本。DIRECTION_VERSION=sv_training_teacher_self_answer_mean_v2。max_num_batched_tokens可为16384，属于多请求总调度预算，不把单题SV上下文扩大至16384。不调整教师/学生生成采样，保留两批100条原提交和Top16缓存，重新校验source_rollout_reuse.json；旧未提交capture/旧GPU验收在原目录记录迁移后清理，使用GPU0、1重新验收并恢复4组overlap，图后停止，0次更新。模板hold保留。

用户最新要求节省硬盘：教师较完整推理和当前生成/读取/注入流程保持；EasySteer与HF提取均不再保存逐token隐状态，只留小型池化均值/范数、SV bank和文本/Top16概率/mask/指标。已完成capture的原始状态按显式清理凭据移出提交清单后删除，保留pooled.pt原字节与bank/结果身份；原100题四方案select与冻结confirm已完成。用户随后明确授权扩展到1000题重跑四方案，该1000题运行现已完成；本轮仅新增教师消融的代码与CPU验证。

2026-11-15当前1000题运行：RUN_NAME=overlap4_1000_v1，保留原100题并从现有本地OpenThoughts3数学子集按seed42补充900道去重、两方prompt不超过1024 token的题；新数据及700/300划分冻结于data/ablation/overlap_1000/，sampling_manifest.json记录来源SHA。四方案在select/confirm均评分（confirm_top_n=4），global池包含全部1000题，confirm为训练侧复核。新运行重新整批生成/评分，不拼接旧100题的生成缓存；原100题输入/划分/结果不改。Slerp0.1、层26、Top16原始概率、SV总上下文2048及pooled_only保持，零更新，图后停止。模板hold保留，独立运行配置承载此次明确GPU授权；其他项目进程绝不停止，指定GPU繁忙只排队。

教师S−T实现约束：只在教师隐空间计算H_T(学生回答)−H_T(教师回答)，不能用学生向量取负或跨模型相减。默认教师层29、Slerp0.1、读取总上下文2048，global/per_question与all/static_overlap组成方案5–8；两方bank使用共同有效题集并独立尺度，不存逐token隐状态。静态mask在原无SV共同span冻结后投影到教师原生预测位置；指标仍在学生坐标，真实教师mask与映射展示mask分别保存。联合八组使用已提交学生组无SVbaseline和冻结教师评分作为完全相同基线，另存teacher_repeat_drift，不自动扣除漂移。GPU验收绑定包含teacher_layers和新方向版本；未通过不得声称教师capture／注入已验收。默认开关0保留四组，开关1默认PARAM_NAME=n1000_slerp0.1_S26_T29_seed42_v1与Confirm八组。本轮禁止自动--run。

2026-11-16用户授权按输出结构.md重构代码输入输出并复用前置step，范围为代码、CPU验证和原产物SHA注册，禁止自动GPU运行。stage4训练和stage5独立评测只预留路径。step1分teacher／student，step2分reader_student／reader_teacher及activations／banks，step3分native_scores／alignment／source_metrics／calibration及split／role，step6保存CSV／分析与来源索引。metadata保存配置／job／上游引用，logs保存外层日志；底层engine_call提交格式保留。

跨参数只按组件科学合同与SHA复用，冻结引用不可改；新checkpoint或新前缀必须重算对应概率／激活。alpha不参与bank键，层位／SV长度不参与rollout和native键。原100／1000题输入与运行结果保持原字节，1000题上游已通过register_ablation_artifacts.py核验注册为外部引用，不复制大张量，不把历史候选或旧验收冒充当前结果。本轮物理迁移已同步修改owner、proof及descriptor SHA，旧目录移除；未完成这种身份更新时不能删除被引用owner。模板hold不解除。

100／1000题参数名分别n100_slerp0.1_layer26_seed42_v1、n1000_slerp0.1_layer26_seed42_v1。100题Select四组／Confirm两组，1000题两部分各四组；汇总在step6。所有16083个原文件的SHA／inode已核验。其它五个旧版／prepared／未完成实验目录及对应无用启动记录已删除。迁移后的原始job、配置、complete及代码快照不改字节，通过artifact_paths解析历史位置；不得恢复旧目录或混用旧GPU凭据。历史完成参数名不允许--run覆盖，新设置使用新PARAM_NAME。下文旧路径仅为历史描述。
