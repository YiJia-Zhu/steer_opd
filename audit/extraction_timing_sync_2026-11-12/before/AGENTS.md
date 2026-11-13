# Repository Guidelines

## Execution Hold — Latest User Direction

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

2026-11-12最新成功标准澄清：首轮只检验steer相对同条件传统直接OPD的增量。
用户自行查看论文SFT数字作近似参照，不要求本地SFT对照，也不将SFT起点或精确复现
论文SFT成绩作为首轮条件。符合预期指：关闭steer后，同一固定前缀上的师生overlap
提高，并且相同评测协议下accuracy提高；带hook即时指标另记。本轮不新增SFT训练/下载，
不据此自动启动GPU；下方完整方案的后续强基线不阻塞这一先导。

2026-11-12最新任务：用户已授权统一反向KL目标、默认配置和代码入口，并与Rethinking仓库逐项对照。
**允许CPU代码/配置检查；实验、模型/数据下载继续暂停，不据任何旧计划自动恢复。**
主计划摘要为`docs/01_research_plan.md`，工程验收为`docs/03_execution_plan.md`；
完整稿的基础配方本轮按用户要求更新；前次原文快照见`audit/plan_integration_2026-11-12/`，
本轮修改前快照见`audit/rethinking_alignment_2026-11-12/before/`。

最新主候选：同一冻结初始学生读取T/R文本的**方案2 T−R方向＋overlap单阈值门控**。
门控仅`g_t=1[O_t<tau_O]`，不与熵差做AND/OR组合，不使用连续强度或可学习控制器。
熵差保留为独立门控对照及诊断。方案1 T−S和R−S对照全部保留。

生成与更新使用同一受控策略定义：生成当前token前先probe/门控，再从实际受控分布采样；
学习前向重放历史干预。主评测无教师、无R、无steer、无在线门控。
旧“普通rollout＋仅loss前向hook”、全位置默认主方法及前10步日程均已降为单独对照。
基础目标现统一为Rethinking原fork Student Top-K16反向KL系列surrogate：only_stu、student_p、
token_reward_direct、三维PPO及token-mean。全词表显式KL另作消融，不默认使用。
默认配置为`configs/rethinking_opd.json`，比较证据为`audit/rethinking_alignment_2026-11-12/`。

用户此前已选：OpenThoughts3原30K题源中取100题OPD，AIME2025共30题仅评测。
100题仍指OPD题池。最新设计要求方向题、OPD、dev与测试分开；完整稿建议方向题256–512，
新增方向/dev题数及预算尚未冻结，不自动执行文本生成或扩大训练。
此前50步、batch64/n4、LR1e-6是预算起点，首轮组数、在线门控开销和GPU预算仍待确认。
本地SFT组及作者SFT下载继续暂缓；后续SFT强基线不构成恢复授权。

当前机制配对候选：Qwen3-4B Non-thinking教师、真正Qwen3-1.7B-Base学生，R候选为
Qwen3-0.6B。三者assets软链接已发现，revision、模式、能力差和新链路仍须验收。
普通Qwen3-1.7B不能替代Base；模型文件存在不代表受控生成/梯度重放已实现。
C1跨家族及C2/C3扩展暂停。当前没有正式新增训练收益结论。

当前默认JSON和规划矩阵已标注原fork RKL及受控方法要求；`launch_pilot.py`已改为原fork配置
预览，显式移除vendor/verl及旧loss-only hook环境，不再默认R1/JustRL/前向KL。
受控rollout/重放尚未实现，E3/E4实际执行明确阻止；配置可组合不代表可训练。
现有新torch/vLLM与原fork未做GPU兼容验收，数据处理与特殊token协议仍待完成。

2026-11-12启动准备补查（本轮配置统一前的历史记录）：已读取用户完整稿并检查现有代码/本地tokenizer。
4B与1.7B-Base普通词元及规范化BPE规则一致，共同有定义词元151665个；教师多四个
新增词元，EOS/模板有差异，不能仅凭151936行输出head认定原生完整支持完全相同。
共同支持集协议仍需冻结。现有5项CPU语义测试通过，仅覆盖旧组件，未验收新受控链路。
证据和下一步建议见`audit/steering_full_plan_readiness_2026-11-12/README.md`；
没有加载权重、下载、GPU任务或新增数据划分，原暂停约定保持。

## Research Mission and Authoritative Documents

此前首轮Figure8讨论记录于`audit/steering_full_plan_readiness_2026-11-12/first_experiment_figure8_proposal.md`，
其“全文词表主目标保持”已被本轮用户要求覆盖：现在普通组与全部方向/门控组共用原Student Top-K16
反向KL系列源码行为，原论文数学子集KL与冻结奖励/PPO不宣称精确梯度等价。100题/50步、
AIME单项、额外steer为明示差异；默认batch64/n4、LR1e-6、prompt1024/response7168、
AdamW/betas/weight decay、constant LR/warmup0、clip/熵系数、token-mean、shuffle=False、
save20及离线avg16协议已逐项核对。SFT组继续暂缓，不恢复GPU。

核心问题：**教师有额外能力但学生难以通过OPD获得收益时，学生空间的教师同家族差分
和按overlap触发的激活干预，能否改善训练后的独立学生能力？**

研究两条轴：方向减谁（T−S/T−R/R−S），以及哪些回答位置注入（全位置/overlap/因果后段）。
方向训练前一次提取并冻结，不把动态提取、多层路由扩为主轴。少量方向是否够用、是否减少
独立SFT需求或更省总算力，均是待验证问题。同家族低相容先验证，跨家族为后续外部证据。

- 用户完整逻辑及本轮基础配方补注：`docs/Steering_OPD_完整研究方案.md`。
- 方法、矩阵与当前范围：`docs/01_research_plan.md`。
- 固定论文版本、阅读范围与资产：`docs/02_paper_settings.md`。
- 实现、配置迁移和恢复验收：`docs/03_execution_plan.md`。
- 2026-11-07历史新颖性审计及新版本提示：`docs/09_novelty_audit_cross_family_steering_opd.md`。
- 导航：`docs/README.md`。

有实质进展、失败或用户纠正时更新本文件和对应主文档。计划、启动、即时诊断不等于方法有效。

### Model and Data Selection

- 本轮共享词表机制配对F1对应Rethinking §5.1：Qwen3-4B Non-thinking→Qwen3-1.7B-Base。
  R=Qwen3-0.6B是本项目方向扩展候选，核对thinking、后训练来源及教师相对能力，不作先验保证。
- C1独立家族候选：Qwen3-30B-A3B-Instruct-2507 / Qwen3-0.6B→Granite4.1-3B。
  Compass表2有配对，OpenThoughts-114k共同SFT、DeepScaleR-Preview OPD；资产/词表接口未落实。
- C2同教师→Qwen3-4B、C3 Qwen3.5-4B / 0.8B→Qwen3-4B仅为后续同家族/4B级控制，
  不替代独立家族证据，不自动新增下载。
- JustRL→R1-1.5B、R1-7B→R1-1.5B可作共享谱系机制对照，不算跨家族证据。
  Phi/Smol不进入当前正式主面板。
- 总参数、激活参数、文本主干、可训练参数分别报告。A3B不等于3B总参数。
- 原版Base、RL后checkpoint、thinking版本不得以普通Instruct或普通同规模模型代替。
- 原30K OpenThoughts3 OPD题集不等于200K SFT示范，也不同于OpenThoughts-114k。
  本轮保留原题源、抽样ID、去重/长度规则及四份数据hash，AIME只评测不选型。
- MATH-500、额外benchmark、更多学生需后续预算约定；本轮不自动加入。

### Method Boundaries

1. 三类回答由T/R/S0生成，但激活都由同一S0读取。每个学生/初始化独立提取方向。
   单独T−R通常两次学生prefill，完整三方向需三类读取；不声称整套系统仅一次prefill。
2. 三方向共用题目、过滤、层、池化及权重，原始`d_T-S=d_T-R+d_R-S`；单位归一化后不要求该等式。
   激活差不自动消除家族偏移或分离纯能力/风格，需长度、格式、正确性等控制。
3. `g_t`在采样当前token前由同一学生前缀上的T和probe分布计算，停止梯度。
   probe当前步关闭注入，历史受控KV仍保留；禁止替换为整段完全无hook重放。
4. 主方法rollout及学习重放均使用相同受控定义。hook在回答预测状态t−1，首位置在prompt边界；
   排除无关prompt、padding和无效位，保留至backward重算结束。probe/正式缓存隔离且重放历史干预。
5. 主loss统一为原fork Student Top-K16/student_p奖励/PPO surrogate，g=0也接受监督。
   loss候选使用实际受控学生Top-K，门控使用当前probe Top-K，二者不混用；冻结候选、
   教师、方向和门控。token-mean、clip等按原配方。全词表KL仅作另列消融；特殊词元检查仍必需。
6. 主门控只用overlap，阈值训练侧校准/独立dev选择后固定。熵差单独对照，不组合；
   后段阈值不读本条未来长度。随机/位置匹配在线只用过去，事后严格匹配仅为离线诊断。
7. 方向/层/强度/阈值最终按真实短程学习后的无干预dev收益选择，不按即时overlap最大选择。
   所有方向共同调参预算；主结果无教师、无R、无steer、无门控，带干预推理单独报告。
8. 正式做三方向×三位置策略九格＋普通OPD，分阶段验收而非整矩阵同时开跑；补正/负/随机方向、
   相同比例/位置随机门控、独立熵差及混杂控制。简单乱配全局均值的题目不改变方向，不算破坏对照。
9. 原始起点和共同SFT起点分轨比较，SFT/LoRA/Compass使用同信息及合理调参预算；
   本地SFT当前暂缓，不据论文曲线证明替代SFT。方向生成、搜索、在线教师和probe/重算全部计费。
10. 不同tokenizer的span评分不等于共同完整候选分布。扩展A需验证分布对齐器；
    扩展B用完整下一字节概率相交系数单阈值，不加熵差条件，span目标另表；
    不把平均log-likelihood、候选归一化或字节指标称作原生完整KL/Top16 overlap。

### Literature Interpretation

Rethinking/Simple主要兼容词表实证不替代本项目跨家族证据。Rethinking §8确有跨家族
混杂讨论；共享Top-K概率质量和overlap集合比例分开，梯度抵消未获直接机制验证。
SV Training §4.1/4.3为teacher-forced训练hook，§5.1 GRPO为正确−错误方向评分修正advantage；
不把后者说成前10步hidden hook。Compass似然分解不证明激活分解有相同消偏保证。
完整稿新增Entropy-Aware OPD为待核对候选；不继承其未经本项目复核的公式/数值为既有证据。
SCOUT已在本地固定2609.38360v1查到，题名为On the Off-Policy Teacher in On-Policy Distillation，
不因输入稿的“未定位”说明删掉已有来源。历史新颖性审计不当成新受控方案已完成审计。

### TODO — Update with Evidence

- [x] 依据2026-11-12最新完整稿同步项目Markdown，保留原文及修改前证据。
- [x] 明确T−R主方向、overlap单门控、独立熵差及三乘三矩阵。
- [x] 基础配方统一为原Student Top-K16反向KL系列源码行为；旧forward-KL入口已归档。
- [x] 原fork配置默认与CPU语义检查；全量材料校验仍因原有12号文档并发快照缺失未通过，不伪造历史证据。
- [ ] 冻结新增D_vec/dev规模、首轮面板、KL归约、资源与总预算；当前不运行/下载。
- [x] 配置默认及入口已对齐固定原fork，受控方法仍标记未实现；不恢复训练。
- [ ] 恢复获准后验收合法支持、T/R模式/能力、原fork GPU依赖及受控链路。
- [ ] 验收非零hook、当前步probe/正式KV隔离、采样行为概率和teacher-forcing重放一致性。
- [ ] 最小闭环后补完整九格、随机/熵差对照及无干预dev确认。
- [ ] 后续强基线、重复seed、冻结测试和外部家族验证；只按实际证据扩大主张。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：2026-11-12已对齐原fork默认及CPU配置预览；受控T−R＋overlap主方案尚未实现/获准启动。
100题OPD及AIME2025仅评测约定保留，新增方向/dev预算和面板待冻结。所有实验/下载暂停。
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

The current design uses matching controlled rollout and learning replay, fixed student-space directions, and overlap-only gating. Use the pinned Rethinking Student Top-K16 reverse-KL-family surrogate as the shared base recipe. Verify cache isolation, prediction masks and gradient recomputation before training. Distinguish shared-lineage results from independent-family evidence.

## Commits and Reviews

No Git history is currently available. Use concise, imperative commit subjects, for example `docs: clarify steering baselines`. Reviews should describe the problem, changed files, evidence or checks, and limitations. Keep weights, credentials, and large run artifacts out of commits.
