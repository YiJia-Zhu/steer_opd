# 实现与运行说明

统一实验名称为`ablation`，100题及1000题的原四方案均已完成，新增教师S−T消融待GPU验收；本轮仅代码、文档与CPU验证。入口为[run_ablation.sh](../scripts/run_ablation.sh)，计划见[ablation.md](ablation.md)。新运行按[输出结构](../输出结构.md)保存于outputs的step0／1／2／3／6，默认RUN_NAME=ablation、PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1；配置、job与结果清单记录experiment/phase。param与baseline后续另行实施。

入口默认STOP_AFTER=panel。既有GPU0、1、RUN_NAME=overlap4_1000_v1的1000题四组实验已完成；新增教师S−T本轮不启动GPU，仍不进入OPD。CPU数据准备脚本[prepare_ablation_data.py](../scripts/prepare_ablation_data.py)保留原100题，从现有本地OpenThoughts3数学子集按seed42抽样并去重补充900题；两方prompt均须不超过1024 token。输入现位于step0的n1000_slerp0.1_layer26_seed42_v1目录，冻结700/300划分，confirm_top_n=4，四方案在两部分均评估。原100题数据/70–30划分及overlap4_v1结果不改，新1000题不混接旧整批缓存。

source_overlap模式仅源rollout、T前缀评分和两种源图，不生成教师参考回答，不capture、不构造bank或注入SV，不运行OPD。主编排在source_overlap分支提前返回，启动自检同样跳过SV相关检查。source_overlap.py逐题读取已冻结原生概率计算两指标，保存前3条token图和全部题均值图、CSV及汇总；没有候选差值。

原all/static_overlap × global/per_question四组与共同无SV基线已完成；当前只授权新增教师S−T代码与CPU验证，未启动新实验。退火仅保留后续真实100训练批次的规则和强度计算接口；不会为题目生成强度idx表，不运行题级退火。未来idx 0/40/80对应2a/a/0，末20批无SV，计算scale后整批提交EasySteer，Slerp不可用预乘向量替代alpha。

当前代码按[指标说明](overlap.md)使用Top16原始概率双指标及版本化缓存。100题实验已完成；1000题使用相同科学参数和新的整批教师/学生生成、capture、bank、评分与GPU验收身份，不启动OPD。global方向按全部1000题提取池构造，包含confirm文本，confirm仅为训练侧冻结复核。

方向协议为`sv_training_teacher_self_answer_mean_v2`：同一冻结S0读取T/S2048-token上下文内保留的回答，分别池化后作差；S回答复用普通无SV源rollout。两份bank共用过滤与尺度，全局版先平均原始差分再归一化，逐题零方向精确无干预。当前共同span仅对齐S/T，不计算家族差分。

## 教师空间S−T镜像消融

显式TEACHER_ABLATION=1增加方案5–8（all/static_overlap × global/per_question）；默认关闭，原方案1–4保持。新方向为同一T0读取学生源回答减去读取自身回答的池化均值，教师独立75%层29、隐宽2048；学生原方向为层26、隐宽2560。S−T不是原学生向量取负，也不引入跨空间映射或Ref。

[capture_student_activations.py](../scripts/capture_student_activations.py)新增--reader teacher，读取教师tokenizer，按学生／教师顺序整批capture，提交activations_teacher/pooled.pt；[prepare_compass_banks.py](../scripts/prepare_compass_banks.py)同样新增--reader teacher，构造banks_teacher/S-T_global.pt与S-T_per_question.pt。两个读取者共享有效题目交集，各自池化和尺度独立，均不保存逐token隐状态。

[teacher_screen.py](../src/steer_opd/teacher_screen.py)将学生冻结mask按已有共同span映射为教师原生预测mask。教师使用自身prompt和学生回答的教师分词进行prefill，注入自身S−T向量后评分完整学生文本；学生概率固定。指标始终广播到学生坐标。原生教师mask保存在native_steering_mask；绘图steering_mask为教师span映射到学生坐标后的展示标记，不能视为学生被注入。

run_ablation.py按角色启动独立GPU子阶段（学生1卡、教师最多2卡），在select/student、select/teacher及对应confirm目录各自提交，根summaries.json合并8组。联合比较时，教师组加载并核验学生组提交的无SVbaseline概率，教师无SV概率继续冻结，确保两组基线完全一致。教师无SV重放另存teacher_repeat_drift.json，报告漂移，不自动校正候选delta。注入率分别报告教师原生比例和映射后的学生覆盖比例。

新增配置与teacher_layers／teacher_direction_version纳入GPU验收绑定。显式启用镜像组后，教师自检增加指定层capture、标签／有限值、全位置／静态mask及无干预重放检查；教师GDN的重复数值漂移单独报告，学生原严格检查保持。CPU检查不等于教师双卡capture／注入已验收。本轮不运行GPU，新参数的模型产物只在后续显式启动时创建，既有100/1000题结果保持。

## 双指标与数据

[prefix_matching.py](../src/steer_opd/prefix_matching.py)定义两项指标共用的兼容关系：异构tokenizer的非空普通token允许完整字节串互为前缀，EOS只配终止事件，其他控制token按具体拼写精确匹配；映射相同时仅按相同原生ID。`overlap`用增广路径求Top16最大一一匹配数/16，别名ID各占一个名额，不能贪婪固定同名项。边界并列沿用固定引擎返回的原生Top-K排名，不做事后扩集或把观察token额外项算入，策略随指标记录留档。

`P_overlap`仅使用两方Top16的原始概率exp(logprob)，不重新归一化。最多32个候选节点组织成小前缀森林，以精确最小割求容量匹配；同词表退化为Top16 ID交集上的sum(min)。每个token的概率只能贡献一次容量，不能把所有兼容对的min直接相加。另记录两方Top16保留质量，便于解释截断范围。

直接使用vLLM公开的logprobs/prompt_logprobs=16，不安装额外logits采集hook、不调用概率RPC。已删除完整词表缓存与258类字节概率投影；每位置ID、logprob及观察token logprob约132字节，约38万位置为48 MiB，文本、元数据及池化结果另计；逐token隐状态只在内存中处理，不落盘。

当前S原生支持151669，T原生支持248077；均为连续已定义ID，不能沿用旧151665子集。ByteLevel以原始字节处理空格和半个UTF-8字符；其他decoder会拒绝。源EOS目标没有文本span而不评分，EOS进入Top16时仍参与有效边界的P_overlap。

两指标都在合法共同span起点评分，再向span内学生tokens广播。保留alignment_id、alignment_mask、原生token范围与共同字节边界；明确广播位置和未评分原因。Compass span logp按原生token平均，不称所有字符串分词路径的精确概率，也不把两个overlap指标当成OPD损失。

源rollout、T和候选评分共用Top16 IDs/logps及sampled_logps。额外返回的观察token若不在Top16，只提供路径logp，不加入指标。概率文件hash、模型内容、tokenizer字节/EOS/控制语义、支持范围与指标版本共同绑定缓存。

## EasySteer路径

隔离环境已安装并通过CPU导入、Slerp注册/范数及API检查，验证记录见[运行时检查](../audit/dual_overlap_implementation_20261114/runtime_cpu_check.json)；真实GPU状态另见上文排查记录。固定前端baecd9698f32aadef68b41536f3db477b513a0d7、fork 7837b07d21c0ddffd043a30001f2c5654290aec7，基座wheel为vLLM0.29.0。普通vLLM0.32环境不能冒充此fork。API源码与hash见[固定源码清单](../ref/easysteer_runtime/manifest.json)。

[bootstrap_easysteer.py](../scripts/bootstrap_easysteer.py)按官方wheel+Python overlay方式安装项目隔离的.venv-easysteer和Slerp插件，不覆盖.venv-opd或共享环境，不加载模型：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/bootstrap_easysteer.py --install
```

| 路径 | 实现 |
|---|---|
| 原生参考/源生成 | [generate_static_sources.py](../scripts/generate_static_sources.py)：全量请求一次公开LLM.generate，返回后保存整批。源采样TP1，教师文本最多TP2 |
| S0激活提取 | [capture_student_activations.py](../scripts/capture_student_activations.py)复用[easysteer_extraction.py](../src/steer_opd/easysteer_extraction.py)：一次公开capture，S0层26读取T/S2048-token上下文内保留的回答；BF16状态只在内存中处理；整批返回后仅提交池化均值/范数和元数据 |
| 方向bank | [prepare_compass_banks.py](../scripts/prepare_compass_banks.py)在CPU校验已提交激活，构造T−S全局/逐题2份bank并整体提交 |
| T概率 | [score_compass_sources.py](../scripts/score_compass_sources.py)：自身tokenizer、完整观察文本prompt_logprobs，取得真实观察token分数，T允许TP2 |
| Top16概率 | [easysteer_static.py](../src/steer_opd/easysteer_static.py)：公开prompt_logprobs/logprobs返回原始Top16，不收集完整词表，不使用额外统计RPC |
| 四组固定mask | [easysteer_static.py](../src/steer_opd/easysteer_static.py)：逐请求SteeringSpec、Slerp插件与已知prompt_positions，无在线教师/probe |

正式实验的模型前向均经EasySteer/vLLM。Tokenizer/config读取、字节与span对齐、池化、bank构造、Top16匹配、文件和绘图由CPU完成。HF只保留显式HF_PARITY=1及hf_batched_static数值参考，默认不加载HF模型。后端路由核对见[审计清单](../audit/coarse_resume_20261114/backend_audit.json)。

所有概率均为raw logprobs，BF16模型的lm_head输出FP32。Qwen3与Qwen3.5嵌套输出头均裁剪到已定义支持，排除补齐head行。Top16按引擎原生rank提取，额外观察token不能替代排名第16项，返回数据须通过形状、排序、唯一ID、范围与原始概率质量检查。

source预测t使用状态prompt_len−1+t。固定前缀评分将完整观察回答放入prompt，额外生成一个token只用于完成前向并丢弃，预留+1上下文容量。按真实绝对位置取统计，不能把prompt概率错配到回答token状态。

capture返回完整post-block hidden+residual，正文池化排除prompt末预测状态。host budget按全部选中BF16行及每行12字节标签计算；device_budget_bytes按引擎实际max_num_batched_tokens、层宽与层数覆盖图输出、选择/类型转换工作区，另留25%及16 MiB余量。两者分开设置，不再遗漏设备预算而沿用256 MiB默认值；默认16 MiB staging/fetch保留。按sv_training预先右截断输入至2048总token；对截断后全部回答位置capture并求均值，不再按结构标记排除回答token。16384等max_num_batched_tokens是多请求调度总预算，与每题2048上下文限制分开，调度仍自动。capture_budget.json记录实际预算，上限不会预先分配满。

## 缓存与指纹

复用上游EasySteer指纹：payload内容SHA、scale、layer、selector、algorithm、normalize参与身份，显示名称不参与。固定前缀评分关闭prefix caching；普通无SV生成按引擎/模型支持允许缓存，完整prompt_logprobs和正文capture会按上游规则跳过读取，避免缺失逐位置概率/激活。不同受干预KV不能仅因文本相同就混用。

[checkpoint_identity.py](../src/steer_opd/checkpoint_identity.py)补充模型内容版本：首次按权重shard和config/tokenizer逐文件流式SHA256，缓存于.runtime，之后按inode/大小/mtime/ctime检查并复用；最近两秒写入的文件仍重hash，避免快速同大小覆盖撞时间戳。身份纳入参考/源job、激活bank和原生评分绑定；同路径替换权重会失效，不逐token重hash。模型资产仅只读。

源overlap/P_overlap各计算一次，紧凑数组保存在原run的alignment/<key>.metrics.pt，配套SHA提交标记绑定源/教师概率、原题、对齐与指标实现；select/confirm构造mask时复用。两项指标均支持阈值校准：先统计相同分数的count，再排序不同分数并累计计数，保留strict-<、nextafter候选和较低阈值优先的原规则；复杂度O(N+U log U)，overlap最多17档。gate_metric仍只选择其中一项控制注入。CPU阶段单独记录source_metrics_and_gate_threshold，避免沿用上一GPU阶段的显示名称。

静态mask保存两种源指标、单一gate_metric、源token/hash、阈值、合法位置与mask_id。候选评分只读这些mask；单独保存source-rollout与无SV重放的两种差值。概率文件SHA、checkpoint与bank哈希校验续跑；已提交评分直接复用，补图不加载模型。

前缀规范使用metric_version=4、matching_rule_version=native_token_prefix_compatibility_v1，P_overlap定义为top16_raw_probability_maximum_prefix_compatible_mass。新格式不与旧指标、阈值、mask、排名或GPU凭据混接。旧overlap4_v1的teacher_text_reuse.json、source_rollout_reuse.json及restart记录仅适用于原100题身份；新1000题运行不复制这些凭据。任何复用均须核验checkpoint、全部请求/生成参数、代码及文件SHA；新前缀或checkpoint取得自己的概率与激活。

## 分阶段产物与跨参数复用

[stage_io.py](../src/steer_opd/stage_io.py)生成step0至6的标准路径，RUN_NAME为实验类型，PARAM_NAME为参数标识。[stage_campaign.py](../src/steer_opd/stage_campaign.py)按实际依赖组织组件；[stage_contracts.py](../src/steer_opd/stage_contracts.py)分别定义生成、capture、bank、对齐、原生评分、源指标、阈值及角色候选的合同，目录名不参与科学身份。

[artifact_reuse.py](../src/steer_opd/artifact_reuse.py)只接入已完整提交的产物。组件metadata/artifact.json保存依赖、来源完成标记及文件SHA。跨参数通过metadata/reuse.json冻结descriptor SHA、artifact_id和owner，直接读取旧文件，禁止重定向冻结引用。stage3/metadata/upstreams汇总实际来源；step6生成comparison.csv、analysis.md和metadata/source_index.json，export索引不复制大文件。缺失完成标记则整阶段重算；篡改或输入合同改变时报错。

alpha在实际candidate_vectors调用中显式传入，bank本身不依赖注入alpha。层位／SV长度只影响对应capture和bank；原生T评分覆盖完整学生源池，与SV过滤无关。阈值依赖Select有效题集，换层导致交集改变时也会重算。加入教师组时，学生capture可复用；学生bank取决于共同有效题集是否改变。完整失效表见[输出结构](../输出结构.md#7-参数变更与复用规则)。

CPU历史注册由[register_ablation_artifacts.py](../scripts/register_ablation_artifacts.py)和[legacy_stage_import.py](../src/steer_opd/legacy_stage_import.py)完成，核验原始code／runtime、模型、文本、提取、bank数值和概率／指标身份；仅已审阅的学生capture兼容变更可适配。旧提交不写入，原始文件现已同盘rename至各step，注册快照和证明保存在metadata内。原候选结果保留为历史；不把旧GPU验收移作当前凭据。底层工作器的complete／stage_input／reproducibility格式保留。

[output_migration.py](../src/steer_opd/output_migration.py)将两批完整结果物理移动到各step，先核验完整提交与目的冲突，再逐文件同盘rename、验证inode／SHA，修复注册owner／proof与descriptor SHA，校验全部产物后才删除明确列出的旧实验。原文件不改字节；需要改变的仅是注册与位置包装。迁移可按清单继续，已完成后重复执行只报告已完成。

[artifact_paths.py](../src/steer_opd/artifact_paths.py)按audit/stage_output_migration.json解析冻结记录中的历史路径。配置／默认输入直接使用step0；旧目录没有软链接。该解析只用于读取保留的原始路径，不重新引入旧实验入口。1000题可复用上游的科学合同和artifact_id保持；100题旧协议及两批候选均明确标记历史完成结果，不能充当当前GPU验收。step3/metadata/historical_run.json阻止在完成参数名上覆盖执行；状态入口可读取metadata/layout_config.json。

## 输出与边界

每候选comparison/overlap与comparison/P_overlap分别生成前3条的PNG/SVG/CSV及first3.pdf，清单保留源顺序/不可用原始条目。蓝线为完整同源无SV重放，橙线为固定mask前向，红点为实际注入，下方差值单位pp；缺失位置保持NaN，不能伪造为0。

summaries.json同时有metrics.overlap和metrics.P_overlap，包括题目等权配对差值/区间和token加权值。阈值在select校准，排名仅看rank_metric；confirm不反向选参。源/候选P概率及两种delta均落盘。

EasySteer逐token实际扰动范数尚未接入（relative_perturbation_available=false）；HF参考能记录。完整OPD梯度路径、GPU一致性/缓存隔离和性能仍待验收，不能称完整研究面板已复现。当前execution_hold与GPU准入在独立入口均生效，没有自动HF降级。

CPU验证命令：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q tests
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/validate_package.py
```

测试包含Top16原始质量不重新归一化、尾部不参与、容量争抢与枚举最小割、EOS/控制/补齐行、跨分词对齐、t−1映射、缓存篡改、正文池化、双图/汇总/续跑、4方案和执行锁；均为CPU验证。

## 可复现性、恢复与进度（2026-11-14）

[experiment_runtime.py](../src/steer_opd/experiment_runtime.py)统一Python、NumPy、PyTorch和vLLM种子为42；**每个采样请求也固定42，已经移除42+i**。子进程启动前固定PYTHONHASHSEED=42、PYTHONUNBUFFERED=1，并使用各自解释器的CUDA工具链。数据划分、问题顺序、采样参数、原生支持、精度、模型/代码/包版本和实际引擎配置留档；未提交调用恢复时仍提交完整原请求集合。

保留vLLM自动调度、CUDA Graph及现有attention实现，不强制慢速确定性内核。GPU批形状、调度、硬件/驱动和归约可能产生数值差异，因此固定seed并不保证跨调度逐bit相同。已提交输出不重新采样；新的输出保留实际GPU与引擎配置供核对。

| 恢复单元 | 持久化与恢复方式 |
|---|---|
| 参考/源生成 | 整批教师文本、整批源生成各一个恢复点；公开generate返回后保存。源token/Top16 ID/logprob与回答hash绑定 |
| 激活提取 | S0读取完整T/S池，一次公开capture；返回后在内存中求均值，仅保存池化均值/范数及配置身份，整体提交；不保存逐token隐状态。随后CPU构造2份bank，单独整体提交 |
| T评分 | 全部源前缀一次generate，返回后保存真实观察token logp、Top16 ID/logprob与评分身份，整批提交 |
| 候选评分 | baseline及每个候选各一个恢复点，每次generate覆盖全部原前缀；返回后计算双指标/delta并整体提交。画图失败仅补CPU图 |
| 全流程 | --execute --resume重新进入原配置；缓存完整的阶段可在CPU校验后跳过GPU准入；发现旧监督器仍存活则接管等待，避免重复启动 |

StageCheckpoint先保存完整阶段的文件，最后原子写入complete.json，清单绑定输入/运行时/候选名称及文件SHA256。恢复时校验已提交文件；未有完成标记的当前阶段文件不复用，整批重跑并覆盖。已提交阶段禁止覆盖，损坏或身份改变会报错。逐请求RowCheckpoint及.objects硬链接机制已移除；生产阶段仍有单写入者文件锁。

配置resume_granularity=engine_call。一次公开引擎调用连续运行，不插入逐请求落盘、检查点或自定义enqueue/step循环；指标和文件处理均在调用返回后执行。vLLM/EasySteer内部仍按自身机制调度、传输capture数据。中断可能损失当前整批或当前候选的计算，已完成阶段与候选复用。当前不保存vLLM在途KV/RNG；真实GPU恢复数值与耗时仍须验收。

provenance保存代码快照、软件/数值设置和权重内容清单；原始参考文本、池化均值/范数、2份bank、共同尺度、实际源tokens与行为概率、教师span分数、静态mask、两指标与图均保留。最终artifact_index.json统一列出来源、校验和及后续训练的适用范围，复用既有大文件哈希，避免重复读取激活。当前没有optimizer/RNG训练断点，因为参数更新数为0；未来新增轨迹/学生checkpoint必须取得自己的行为概率、教师评分与激活，不能把固定源前缀候选分数冒充新on-policy数据。

日志同时实时显示并写入step3/logs/console.log；GPU子阶段另有各自step/logs/attempts/阶段时间戳/stdout.log及process.json，记录排队、启动/结束、GPU/驱动、命令、PID、退出码与耗时。阶段输出目录有events.jsonl、最新progress.json及每次attempt快照：北京时间含UTC+08:00、首次/本次开始、已耗时、累计活跃时间、含中断等待的墙钟时间、恢复数、完成数和ETA；每15秒心跳，模型加载、评分、绘图等有分阶段时间。

外层启动验收、campaign、CPU指标/绘图和bank阶段关闭tqdm，只打印带时间的阶段开始/结束、耗时、缓存复用和错误。完整JSON事件及15秒心跳仍写events.jsonl/progress.json，终端不再打印心跳JSON。EasySteer/vLLM原生模型加载、CUDA Graph、生成、评分/capture的tqdm仍实时转发，不等子进程退出后才显示；原生进度回调只更新内存计数，不逐请求提交文件或事件。completed为本次前向完成数，committed为可恢复的已提交数；--status/--watch分别显示两者。ETA按本次完成请求估计；第一条完成前、GPU排队、异质阶段的全流程ETA均显示未知。Ctrl+C/SIGTERM记录可恢复中断；SIGKILL无法写结束事件，恢复时保留最后心跳并记录接续。CPU故障测试覆盖这两种真实进程中断。

在项目根目录只读查看进度（不会启动模型）：

```bash
python scripts/show_ablation_progress.py --watch
```

推荐使用[统一启动脚本](../scripts/run_ablation.sh)，参数及方案编号见[实验协议](ablation.md#一条命令启动)：

```bash
bash scripts/run_ablation.sh --check
bash scripts/run_ablation.sh --run
```

--run由用户明确执行，先做师生小样本GPU自检，再执行面板。默认HF_PARITY=0，设1才增加HF对照。观察token logp平均绝对差≤0.05、最大≤0.2；返回Top16 ID并集上的截断概率总变差≤0.03，可选HF激活相对L2≤0.03；零强度/移除恢复用1e-5容差。截断总变差不代表完整词表一致；阈值不因失败自动放宽。

使用spawn创建引擎，避免CPU线程池被fork后阻塞pinned-memory分配。编译器与CUDA头文件匹配torch CUDA版本；TP>1关闭FlashInfer autotune，纯文本Qwen3.5跳过图像/视频预热。初始化限时600秒，生成/评分不使用该限时；中断只清理本项目已验证PID/进程组。已移除概率采集hook，不改上游runner。

既有GPU诊断发现学生BF16批形状波动影响零干预/重放检查，学生生成、capture和评分统一开启batch-invariant；教师保留默认，自动调度和CUDA Graph保留。策略写入源码与engine_config；新Top16格式仍须重新GPU验收。

Qwen3.5 GDN_ATTN不支持当前batch-invariant。T仅评分一次、SHA冻结供所有候选复用；教师重复漂移单独记录。当前验收检查Top16原生支持和观察token；学生零干预、mask、批量/单请求及源重放仍严格检查，显式HF_PARITY=1时师生跨后端对照也严格执行。

验收结果绑定代码、师生两模型权重、层、强度和教师TP；已完成验收阶段按完整调用复用，配置变化或文件损坏拒绝混接。通过后生成新run的独立配置，凭据传入每个子阶段；原模板的hold保留。再次--run即按粗粒度恢复。GPU池按物理编号限制，最多两张，不因指定卡繁忙跳到其他卡。shell文本及实际GPU选择每次单独留档。

SV提取采用sv_training规则：1000题全部参与，教师/学生回答的正确性、finish_reason和重复程度均不作为过滤条件；以prompt+strip后的回答右截断至sv_max_context_len=2048，在prompt之后所有保留回答token上平均所选block激活。仅无回答token时跳过。教师生成上限12288、学生源生成上限7168；2048仅约束SV读取，不截短原始生成或教师前缀评分。

当前存储策略为pooled_only：EasySteer与HF路径均在内存中池化，移除逐token状态序列化回调，只提交pooled.pt及小型元数据，再构造SV bank。旧capture清理先完整校验，hidden_state_cleanup.json保留原complete清单，原子更新保留文件清单后删除teacher/student原始状态；中断后可续清理。原pooled.pt不改，bank和已完成候选结果不改。artifact_index仅索引实际保留文件。


## 后续训练loss组件：传统OPD／CompassOPD

[training/losses.py](../src/steer_opd/training/losses.py)提供可求梯度的采样OPD loss；[training/config.py](../src/steer_opd/training/config.py)定义默认设置与所需角色。已接入下述固定verl版本的step4训练入口和评分桥接，尚未GPU验收；现有overlap／参数实验入口及配置身份保持，本轮不启动训练。

| loss_type | 所需模型／策略 | 监督信号 |
|---|---|---|
| traditional_opd（默认） | 当前学生Sθ、冻结教师T | 同一span内平均logp的T−Sθ差分，停止梯度后广播到学生token |
| compass_opd | Sθ、T、教师同家族弱Ref，加训练前学生冻结副本S_ref | 同一span的T−Ref差分，减去逐学生token的anchor_coef×(log Sθ−log S_ref) |

Compass路线默认Ref路径assets/models/Qwen3.5-0.8B、anchor_coef=0.5，锚点系数与steer alpha分别定义。S_ref在首次更新前从学生初始化冻结，不能被当前学生checkpoint替代。两路线共用相同的文本span对齐程序，S_ref沿原始学生token逐位置评分；只有Sθ的loss前向保留梯度。当前S−T干预教师时，T的评分可来自T_SV，Ref与S_ref均保持无SV。Ref不参与SV提取／注入，未重新引入T−R激活融合。

Tokenizer复用[compass_alignment.py](../src/steer_opd/compass_alignment.py)的align_response：学生实际采样ID不变，各评分者独立重分词同一精确响应字节；span内按各自原生token的观察logp求平均，再广播到学生位置。不能一致重建响应字节的样本拒绝，不能形成共同边界的位置由mask排除。loss不读取Top16候选或overlap数值；教师、Ref和锚点只需观察token原生logp。实现依据本地[CompassOPD论文](../ref/2609.10154v1_compass_opd.pdf)§2、§3式(4)、§4及附录A.4，尚未核验官方训练源码。

`distillation_loss`返回loss、loss_sum、valid_tokens、alignment_mask、停止梯度的advantages及教师信号／学生位移。传统路线拒绝额外Ref评分；Compass路线缺少Ref或冻结学生锚点会报错。`token_mean_loss`按有效token数合并不同长度回答，避免按回答等权改变训练权重；完全没有有效对齐token的样本须由后续训练器明确跳过或报错。

只读预览（不加载模型，不写实验输出）：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m steer_opd.training
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m steer_opd.training --loss compass_opd --anchor-coef 0.5
```

未来正式训练配置可在唯一模板的派生配置中用`training_loss`选择路线；没有该字段时默认traditional_opd。当前正在执行的overlap模板不改，避免使冻结参数搜索身份变化。配置示意如下，仅供后续训练使用：

```json
{
  "training_loss": {
    "loss_type": "compass_opd",
    "anchor_coef": 0.5,
    "reference_model": "assets/models/Qwen3.5-0.8B"
  }
}
```

CPU预览已核验Ref与T的家族／权重及SV身份；真实运行前仍需通过多卡GPU验收，检查实际显存、权重同步、评分与带梯度更新。优化器、训练恢复与新轨迹评分已接入下述verl入口。新的rollout或checkpoint重算对应原生评分；学生学习logp使用新的带梯度前向。loss组件按自身源码与复用的对齐源码单独记录SHA，当前overlap源码身份与运行配置保持。


## 基于verl的step4训练入口（代码已接入，GPU待验收）

用户指定以verl为训练框架。本实现固定官方源码revision `61adb0c7eb90730306e7794c2bf248331211e657`，使用该版本仍提供的同步RayPPOTrainer入口和FSDP worker；通过公开`set_loss_fn`接口安装采样OPD loss，不改上游文件，不启用其按原生教师ID计算的Top-K KL。源码归档在项目.runtime/verl_training，归档SHA与实际Python／YAML源码在预览时核验；.venv-opd运行学生端，.venv-easysteer运行教师端，两套vLLM不混装。

入口为[verl_entry.py](../src/steer_opd/training/verl_entry.py)，可用[run_verl.sh](../src/steer_opd/training/run_verl.sh)调用；[verl_plan.py](../src/steer_opd/training/verl_plan.py)集中定义训练准备默认值。这些值是代码准备默认值，尚未作为正式训练实验执行；教师层位／强度应在overlap选参完成后显式确定，不将默认层29／Slerp0.1视为最终选参结论。

| 项目 | 当前准备配置 |
|---|---|
| GPU划分 | 学生物理0／1：FSDP训练与vLLM rollout分时；教师物理2／3：EasySteer TP2；两组可显式修改且必须不相交 |
| 数据 | 原1000题保持；700题Select供学生更新，300题Confirm保留，不自动跑验证或benchmark；SV仍使用原1000题global bank，Confirm不称独立未见验证集 |
| 训练 | 全局batch16、每卡microbatch1、100个更新批次、学习率1e-6；固定强度，不执行退火 |
| on-policy约束 | 每批新rollout只更新一次；ppo_epochs=1，mini_batch_size=全局batch，n=1，temperature=1／top_p=1／top_k=-1 |
| 学生数值 | 每个采样请求seed42；FSDP／学生vLLM启用batch-invariant，保留自动引擎调度；不承诺跨硬件逐bit一致 |
| 教师注入 | S−T／global／all，同层bank核验；Slerp或原始差分加法，系数只乘一次；学生rollout和训练前向不注入 |
| 存储 | 只保存各模型观察token logp及请求身份，不落盘训练Top16或逐token隐状态；每10步保存verl训练checkpoint，最多保留1份 |
| 显存准备值 | 教师TP2 fraction0.60；Compass额外Ref0.06、冻结S₀0.18在教师组首卡分进程共用；学生rollout0.4，FSDP参数／优化器offload；均尚待GPU实测 |

执行链：当前学生生成→verl重算当前学生无梯度logp→独立EasySteer进程组对同一文本批量评分→共同span对齐并冻结优势→学生带梯度前向→FSDP按全局有效token数同步更新→verl同步rollout权重。Ref和冻结初始学生仅在Compass路线启动，各自使用独立进程和原生tokenizer；教师侧首卡共享显存的配置必须另做验收，CPU通过不代表显存已足够。

[verl_adapter.py](../src/steer_opd/training/verl_adapter.py)校验评分顺序与学生实际token ID，排除未对齐／EOS／padding，只把有效位置交给worker；完全无有效位置的样本权重为0，整批无有效位置则拒绝更新。各microbatch loss按dp_size／全局有效token数缩放，经FSDP跨卡平均得到正确的全局token均值；不添加PPO clip、任务正确性奖励、额外KL或优势标准化。verl rollout阶段的零奖励仅为接口占位，在actor更新前由真实OPD优势完整替换。

[scorer_service.py](../src/steer_opd/training/scorer_service.py)持有冻结评分引擎，整批调用返回后按SHA提交观察logp；[scorer_client.py](../src/steer_opd/training/scorer_client.py)通过项目内请求队列取回结果。缓存绑定完整prefix、tokenizer对齐、模型内容、SV及训练代码身份；新的学生轨迹重算冻结模型评分，相同冻结评分条件可复用；当前学生训练logp始终来自带梯度的新前向。学生优化器／调度器／数据迭代恢复以verl已提交checkpoint为准，尚未保存的更新会重做，不能用评分缓存冒充参数checkpoint。

只读检查与CPU准备示例：

```bash
bash src/steer_opd/training/run_verl.sh --check
bash src/steer_opd/training/run_verl.sh --check --loss compass_opd
# 本地固定源码尚不存在时，仅下载框架源码，不下载模型：
bash src/steer_opd/training/run_verl.sh --bootstrap-source --check
# CPU导出700／300 parquet并冻结配置，不占GPU：
bash src/steer_opd/training/run_verl.sh --prepare --run-name verl_st_trial_v1
```

准备产物位于outputs/step_4_train/ablation/<模型组合>/<训练数据集>/shared/<参数名>/，metadata保留计划、完整verl配置与来源SHA，data保存训练／Confirm parquet，scorer保存每批实际请求／原生评分／角色日志，checkpoints由verl管理。当前不修改正在执行的configs/ablation.json，也不更改其execution_hold；训练配置从唯一模板读取共同模型／数据定义后独立冻结。

`--run`另要求与完整训练plan binding匹配的GPU验收凭据，检查通过之前不会启动Ray／模型。当前尚未生成该训练凭据，未做多卡前后向、实际权重同步或断点恢复的GPU实测；已有overlap验收不能替代训练验收。本轮只完成代码、真实verl接口的CPU检查和文档，不启动训练。
