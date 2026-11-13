# Repository Guidelines

## Latest Task — Default vLLM Scheduling and Resume

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
5. 主门控只用g=1[Overlap@16<tau_O]，在采样前停止梯度；probe仅当前步无注入、保留历史受控KV。
   正式分支从同一旧KV出发，probe上层缓存不污染正式结果，不用整段完全无hook分布替代probe。
6. 生成与学习重放同题向量及历史mask，首回答预测状态t−1在prompt边界，排除无关prompt/padding，
   hook保留至backward重算结束；batch/packing不能串题或串向量。g=0也有OPD监督。
7. 基础目标沿用Student Top-K16/student_p/token_reward_direct/PPO及token-mean。
   loss支持取实际受控学生Top16，门控支持取probe Top16；冻结候选/教师/向量/门控，理论与源码口径分开。
8. 方向/层/强度/阈值按真实短程学习后的无干预dev收益选择，各方向等调参预算。
   熵差仅独立对照，不与overlap做AND/OR；后段不读本条最终长度，在线随机匹配只用过去。
9. 主结果只测无教师/无R/无向量/无门控学生。逐题方向含训练参考信息，同文本及T/R两源基线匹配信息；
   记录参考生成、当前学生重读、bootstrap、在线教师/probe/重算、学习与评测全部成本。
10. 同数据全局vsS0逐题缓存隔离粒度，S0逐题缓存vs当前学生重提隔离刷新。
    逐题本题/其他题向量控制有效；全局均值两侧集合/权重不变时打乱题目配对不改变方向。
11. 不同tokenizer先验收共同动作/span接口；平均log-likelihood不是整词概率，字节门控不是原生Top16。
    纯文本教师只能供方向文本，不能原样运行在线概率门控；跨家族主张限于实际配对证据。

### Literature Interpretation

SV Training已有初始学生全局/逐题固定及每200步全局刷新，C.2为Qwen2.5-1.5B/GSM8K SFT；
不能把其刷新结果直接外推为本项目OPD结论。其§5.1 GRPO用方向投影修正advantage，
不等同于前10步hidden hook。逐题、刷新和训练时steering本身不作为首次提出。
Compass同动作似然分解不证明不同文本激活差自动消偏。Rethinking §8确有跨家族混杂讨论，
共享质量与overlap集合比例分开；梯度抵消与本项目机制均非已证实。
Entropy-Aware/桥接等候选不继承核心稿全部阅读声明；SCOUT本地固定2609.38360v1来源仍保留。

### TODO — Update with Evidence

- [x] 同步2–2–2、当前学生逐题提取、训练题参考复用与dev/test独立边界。
- [x] 分清全局/逐题粒度与固定/刷新频率，保留两组桥接比较及本题/他题控制。
- [x] 在文档中定义每题参考、向量、提取checkpoint及rollout/mask的版本化恢复契约。
- [x] 保留已有首轮配置/授权与原RKL配方，不将新方案冒充当前运行，不更改用户核心稿。
- [ ] 新版实施：逐题缓存与重提接口、batch/题目绑定、提取无梯度及断点一致性。
- [ ] 新版实施：probe/正式KV隔离、动态门控与同题向量因果重放验收。
- [ ] 冻结新增面板、参考文本、dev选型与总成本预算，再形成独立执行记录。
- [ ] 根据无干预学习收益逐项消融，在线无增量时收敛缓存/全局版本。
- [ ] 后续强基线、重复seed与外部家族验证，按实际证据限定主张。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：已有首轮普通OPD/T−S全位置全局固定实验的授权与配置继续有效；实时进度看run记录。
新版2–2–2为待实施设计，本轮仅同步Markdown。SFT、C1及其它扩展不自动启动。
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
