# Repository Guidelines

## Execution Hold — Latest User Direction

2026-11-12最新成功标准澄清：首轮只检验steer相对同条件传统直接OPD的增量。
用户自行查看论文SFT数字作近似参照，不要求本地SFT对照，也不将SFT起点或精确复现
论文SFT成绩作为首轮条件。符合预期指：关闭steer后，同一固定前缀上的师生overlap
提高，并且相同评测协议下accuracy提高；带hook即时指标另记。本轮不新增SFT训练/下载，
不据此自动启动GPU；下方完整方案的后续强基线不阻塞这一先导。

2026-11-12最新任务：按用户修订的`docs/Steering_OPD_完整研究方案.md`同步其它Markdown。
**只更新文档，实验、模型/数据下载继续暂停；不据任何旧计划自动恢复。**
主计划摘要为`docs/01_research_plan.md`，工程验收为`docs/03_execution_plan.md`；
本轮完整稿保留原文与hash，修改前快照见`audit/plan_integration_2026-11-12/`。

最新主候选：同一冻结初始学生读取T/R文本的**方案2 T−R方向＋overlap单阈值门控**。
门控仅`g_t=1[O_t<tau_O]`，不与熵差做AND/OR组合，不使用连续强度或可学习控制器。
熵差保留为独立门控对照及诊断。方案1 T−S和R−S对照全部保留。

生成与更新使用同一受控策略定义：生成当前token前先probe/门控，再从实际受控分布采样；
学习前向重放历史干预。主评测无教师、无R、无steer、无在线门控。
旧“普通rollout＋仅loss前向hook”、全位置默认主方法及前10步日程均已降为单独对照。
兼容词表主目标采用全词表反向KL；原Rethinking Student Top-K16另列参考，不能混称原配方复现。

用户此前已选：OpenThoughts3原30K题源中取100题OPD，AIME2025共30题仅评测。
100题仍指OPD题池。最新设计要求方向题、OPD、dev与测试分开；完整稿建议方向题256–512，
新增方向/dev题数及预算尚未冻结，不自动执行文本生成或扩大训练。
此前50步、batch64/n4、LR1e-6是预算起点，首轮组数、在线门控开销和GPU预算仍待确认。
本地SFT组及作者SFT下载继续暂缓；后续SFT强基线不构成恢复授权。

当前机制配对候选：Qwen3-4B Non-thinking教师、真正Qwen3-1.7B-Base学生，R候选为
Qwen3-0.6B。三者assets软链接已发现，revision、模式、能力差和新链路仍须验收。
普通Qwen3-1.7B不能替代Base；模型文件存在不代表受控生成/梯度重放已实现。
C1跨家族及C2/C3扩展暂停。当前没有正式新增训练收益结论。

`experiments/`旧JSON/CSV及历史启动器仍描述此前loss-only/候选目标草案；本次仅同步Markdown，
这些文件尚未迁移，不能作为新版主方法启动配置。恢复前必须重写配置、面板并验收。

2026-11-12启动准备补查：已读取用户完整稿并检查现有代码/本地tokenizer。
4B与1.7B-Base普通词元及规范化BPE规则一致，共同有定义词元151665个；教师多四个
新增词元，EOS/模板有差异，不能仅凭151936行输出head认定原生完整支持完全相同。
共同支持集协议仍需冻结。现有5项CPU语义测试通过，仅覆盖旧组件，未验收新受控链路。
证据和下一步建议见`audit/steering_full_plan_readiness_2026-11-12/README.md`；
没有加载权重、下载、GPU任务或新增数据划分，原暂停约定保持。

## Research Mission and Authoritative Documents

2026-11-12首轮讨论补充：用户要求先说明第一个实验与Rethinking图8的对应关系。
具体建议记录于`audit/steering_full_plan_readiness_2026-11-12/first_experiment_figure8_proposal.md`：
普通Base OPD与T−S全位置steer两组，可先保留原Student Top-K16作Figure8配方参考；
steer组生成和学习均注入，评测关闭。此为待冻结建议，未批准改动完整稿主目标/方向，
不自动恢复GPU。以后全词表RKL主方法仍需匹配的普通RKL基线，不能直接沿用Top-K组替代。

核心问题：**教师有额外能力但学生难以通过OPD获得收益时，学生空间的教师同家族差分
和按overlap触发的激活干预，能否改善训练后的独立学生能力？**

研究两条轴：方向减谁（T−S/T−R/R−S），以及哪些回答位置注入（全位置/overlap/因果后段）。
方向训练前一次提取并冻结，不把动态提取、多层路由扩为主轴。少量方向是否够用、是否减少
独立SFT需求或更省总算力，均是待验证问题。同家族低相容先验证，跨家族为后续外部证据。

- 用户最新完整逻辑：`docs/Steering_OPD_完整研究方案.md`，保留原文。
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
5. 主loss为共同合法支持上的全词表RKL，g=0也接受监督。教师、门控、方向冻结；逐状态surrogate
   不含状态访问分布完整导数，参数变化后的旧门控与陈旧策略须明确记录。Top-K近似需另定义验收。
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
- [x] 区分受控生成/重放主方案与旧loss-only、原论文Top-K目标和新全词表KL。
- [x] 同步Markdown链接/结构及主方法语义检查；旧全量validator仍因原有`runs/README.md`缺失失败，不能宣称全量通过。
- [ ] 冻结新增D_vec/dev规模、首轮面板、KL归约、资源与总预算；当前不运行/下载。
- [ ] 恢复获准后迁移旧配置/矩阵，验收合法支持、T/R模式/能力与新概率接口。
- [ ] 验收非零hook、当前步probe/正式KV隔离、采样行为概率和teacher-forcing重放一致性。
- [ ] 最小闭环后补完整九格、随机/熵差对照及无干预dev确认。
- [ ] 后续强基线、重复seed、冻结测试和外部家族验证；只按实际证据扩大主张。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：2026-11-12仅同步Markdown；受控T−R＋overlap主方案尚未实现/获准启动。
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

The proposed design uses ordinary student rollouts and early steering only during the loss forward pass, followed by removal. Verify hook scope, prediction masks, and gradient recomputation before training. Distinguish shared-lineage experiments from genuinely heterogeneous families.

## Commits and Reviews

No Git history is currently available. Use concise, imperative commit subjects, for example `docs: clarify steering baselines`. Reviews should describe the problem, changed files, evidence or checks, and limitations. Keep weights, credentials, and large run artifacts out of commits.
