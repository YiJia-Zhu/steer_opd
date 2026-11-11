# Repository Guidelines

## Execution Hold — User Request, 2026-11-10

用户最新要求：**“先kill掉吧，先想好方案”**。暂停后续所有实验启动与续跑，
先进行方案讨论和只读结果复核；用户明确恢复运行后再启动。此前“规划后直接运行”
的授权已被本次停止要求覆盖，不得据旧授权自动重启。

停止核查时，最近一轮诊断已于2026-11-10T13:07:03+00:00自然完成；原调度及worker
均已退出，无需发送终止信号。再次扫描确认本项目实验进程为0，已保留全部日志与结果。
证据：`runs/cross_family_steering_probe_20261110_02/user_stop_20261110.json`。
完成诊断不表示跨family OPD有效；尚未进行跨family参数更新和学习收益验证。

## Paper-Aligned Selection — User Correction, 2026-11-10

用户最新要求：**模型与数据优先对应Rethinking OPD、Simple-OPD和CompassOPD的
实际实验设置，便于比较；不再随意增加配对或下载。** 当前仍暂停运行与下载。
选型以 `docs/14_paper_aligned_models_and_data.md` 和
`audit/paper_alignment_2026-11-10/selection_audit.json` 为准，覆盖旧文档的冲突建议。

- 跨family主配对建议采用Compass已有的Qwen3-30B-A3B-Instruct-2507→Granite4.1-3B，
  教师家族参考Qwen3-0.6B；原数据为OpenThoughts-114k教师重答SFT与DeepScaleR-Preview OPD。
  同一教师→Qwen3-4B作为论文已有的同family对照，按其DAPO设置单独记录。
- 教师/参考/Qwen3-4B本地已有；Granite的精确发布映射和权重、原冷启动语料尚有缺口。
  本地DeepScaler处理版40,315题可以先复核，不凭目录名认定完全等于Compass官方题集。
- Rethinking/Simple的JustRL→R1-1.5B共同配对和Rethinking的R1-7B→R1-1.5B对照已有资产。
  Rethinking冷启动的1.7B-Base、Simple主分析的RL后8B教师不能用本地Instruct/普通8B替换。
- Phi-3.5/SmolLM3移出正式主实验及baseline；已有权重/日志/结果保留为历史探索。
- 不能因为某模型在论文中出现，就把任意教师/学生拼接称为原论文配对；不能更换数据、
  初始化、loss或评测后直接比较文献表格分数。所有改动明确标为adapted。
- 论文范围核对：Rethinking/Simple已报的OPD连线均在兼容词表的模型谱系内，
  Rethinking §8确实讨论Qwen→LLaMA等cross-family的预训练数据/tokenizer/架构混杂，
  但未据此完成本文自己的异构tokenizer实验；不要把“未实验验证”说成“完全未提及”。
  不是跨tokenizer OPD证据。Simple §3.2的GPT-5.5提供离线SFT文本；**SFT后仍有OPD**，
  §3规定每个warm-up checkpoint接75步，逐token评分教师仍为DAPO RL后的Qwen3-8B-Base。
  “跨family发生在SFT数据来源”不等于“整组实验只做SFT”，不得省略后续OPD阶段。
  Compass §2/A.4才提供本项目异构配对所需的共同文本span评分；多token span取平均
  token log-likelihood并mask未对齐位置，不称为完整文本概率或精确跨词表KL。
  Steering不解决输出词元一对多对齐。证据：`audit/paper_alignment_2026-11-10/tokenizer_scope_review.json`。
- 缺口只记录，先讨论最小实验面板和预算；未获用户明确恢复，不补下载、不自动运行。

## Research Mission and Current Priority — 2026-11-10

This section records the user's research objective. It takes priority over the
older same-family-first schedules in docs/06 and docs/10. Read it at the start of
every work session and update the TODO/status below after meaningful progress,
failures, or a change in the user's direction. Verify status from actual artifacts
and processes; link evidence and distinguish planned, running, and completed work.

### 我们要解决的问题与最终目标

**核心研究问题：能否通过 steering 学生，改善真正跨 family 的教师—学生 OPD 迁移？**
强教师的高分不保证其监督能被某个学生有效吸收。我们研究：对于同一个固定强教师，
不同学生所需要的教学适配，能否由学生自身空间中的少量激活方向表达，并依据
**撤去干预后的实际学习收益**选择这些方向，改善后续 OPD。

最终需要建立的是可重复的迁移收益、适用范围和成本优势，以及相对 SFT/LoRA
冷启动和 CompassOPD 的独立价值。方法是否有效、是否更省、是否有足够新颖性
均须验证，不预设结论。不能用教师/学生初始分数、环境就绪或训练启动代替此目标。

### 方法边界：教师文本 → 学生 prefill → 少量学生方向

1. 从独立训练校准题中取少量题，让固定教师生成推理回答，并取得初始学生自己的回答。
2. 对学生 j，教师回答和该学生回答都由**同一个冻结、无 hook 的初始学生 S0_j**读取。
   先对每条回答的内容 token 激活取均值，再计算配对差并跨题平均：
   `v_j[l] = mean_i(Phi_S0_j(x_i, y_T_i)[l] - Phi_S0_j(x_i, y_S_j_i)[l])`。
   教师文本的读取是学生的一次 prefill；参考项可另做学生回答的 prefill，或在验证
   等价后复用缓存。绝不直接相减教师 hidden state 和学生 hidden state。
3. 每个学生独立提取和选择自身方向；同一教师文本可以复用，跨 hidden space 不复制向量。
4. 诊断时可在固定普通学生前缀上开/关 hook 比较。主训练仍使用**普通学生 rollout**，
   仅早期蒸馏 loss 前向注入方向并更新学生参数，之后移除方向、继续普通 OPD。
5. 主结果评测普通、无 hook 的学生。只有推理时保留 steering 的收益另列为消融。
6. 教师文本即可用于方向发现；后续概率蒸馏通常仍需教师评分，不能把整个流程称为黑盒。

### 必须按此顺序推进

**第一步：跨 family 的原始 overlap/相容性诊断。** 参考 Rethinking OPD 的学生前缀
分析，先固定普通学生生成的前缀池，再比较教师与学生。真正异构 tokenizer 不能直接
比较 token ID、原生 top-k 或原生熵；必须先定义共同文本/字节事件、对齐和覆盖率。
共享词表的论文 Overlap@16 与跨 tokenizer 的扩展指标分开命名、分开报告。
同时记录共享概率质量/分布差异、位置/题型/长度，避免仅以集合 overlap 判断相容。

**第二步：steering 能否改善这些诊断。** 在同一固定前缀、同一输出比较接口上，
比较无干预、真实方向、同层同范数随机方向、负方向；报告开 hook 的即时变化。
方向来自校准集，层/强度只用独立 dev。overlap 变化是诊断信号，不能单独证明
“思维相容”或参数学习改善；要控制格式、回答长度和候选覆盖变化。

**第三步：再做 steer-assisted OPD，按实际学习收益选方向。** 各候选从相同学生
初始化做短程更新，关闭 hook 后比较 dev 收益、后续普通 OPD 曲线及成本，再决定
哪些方向进入正式实验。保持 rollout 来源和 loss 估计正确，不能把普通学生采样
直接当作被 steering 学生采样的 reverse-KL/策略梯度估计。


- **SFT/LoRA 适配是直接竞争方案。** 对同一批教师示范，比较 full SFT→OPD、
  LoRA warm-up→OPD、SV-SFT→OPD 和我们的方案；给各方法合理调参预算。
  教师生成、学生 prefill、方向搜索、warm-up、教师评分和训练均计成本。
  小样本实验不能冒称 Rethinking 的200K教师示范完整复现。
- **CompassOPD 是跨 family 强基线。** 它使用教师相对同家族弱参考的似然增量，
  并锚定初始学生；我们研究学生内部训练前向的适配。二者可能互补，须比较
  Vanilla、Steer、Compass、Compass+Steer，而不是先假定 Compass 无法解决问题。
  Compass 面板中各组从同一教师轨迹 SFT checkpoint 开始，共享输出接口与成本口径；
  原始学生冷启动面板另列。移植后的不同目标/接口必须标为 adapted。
- 若同信息/成本的 SFT 或 Compass 已足够，或者真实方向不优于随机方向，应记录
  负结果并收窄主张。新颖性边界见 docs/09；不能把已有组件的组合自动称为创新。

### 当前 TODO 与证据（持续维护）

- [x] 重申并记录上述研究目标、方法边界和“overlap → steering → OPD”优先级（2026-11-10）。
- [x] 已有同谱系 R1/JustRL 工程验证、128个校准配对及真实/随机方向；它们只作工程对照。
  `runs/pilot_20261110_01` 的E0和E3均已完成100步；尚无其训练后评测结论。
  队列调度已按用户优先级调整停止，不再自动追加同family E4与长评测，当前产物均保留。
- [x] 核对三篇论文的实际模型/数据与本地资产；第14篇记录对应面板和缺口，未进行下载。
- [ ] **P0：** 讨论并固定论文对应的最小主线：Compass的Qwen3-30B/0.6B→Granite，
  另以同教师→Qwen3-4B对照；保留Rethinking/Simple原配对作为机制与warm-up参照。
  核对checkpoint、初始化、题目集合、长度及成本。至少一个真正异构family；多异构family
  主张需要扩展Compass原有同教师多学生设置。当前只规划，不下载/运行。
- [x] **历史P1诊断：** 定义并验证跨 tokenizer 的共同输出比较接口，报告对齐/候选覆盖及概率口径；
  先用共享词表控制验证可还原 Rethinking 的原生 Overlap@16，再测真正跨 family 的基线。
  已完成16题×3前缀×3学生的144状态基线，教师重编码失败0；四模型评分与原生logits差0。
  已修复Phi独立解码省略首空格的问题，复用原rollout重算；5项指标/真实tokenizer回归测试通过。
  权威结果为 `runs/cross_family_overlap_20261110_02/summary.json`，复核为
  `audit/cross_family_priority_2026-11-10/baseline_review.json`；解读见 `docs/13_cross_family_diagnosis_20261110.md`。
  共享词表控制还原原生Overlap@16；真正跨family仅报告明确标注的扩展诊断。
- [ ] **P1论文配对诊断：** 在最终确定的原论文模型/数据面板上重新固定普通学生前缀，
  按原论文/异构扩展分别诊断。历史Phi/Smol数值不能替代Granite等正式配对结果。
- [ ] **P1训练接口：** 独立落实并验证跨tokenizer的可微OPD目标和覆盖率；先证明无hook可学习。
  现有字节诊断不是OPD loss，现有共享ID训练器不可直接用于异构tokenizer。
- [ ] **P2：** 保存固定学生前缀和无干预诊断；提取每个学生的方向，测真实/随机/负方向
  对同一前缀 overlap、概率质量及分布差异的影响，区分即时干预与参数学习；正式论文配对待执行。
  历史Phi/Smol探索实现了长度匹配池化、中层单方向及rho=0.01/0.03/0.1诊断；
  `runs/cross_family_steering_probe_20261110_02` 的5个阶段均exit0，实际采用24个教师正确完整配对；
  已完成历史诊断执行，结果保留供只读复核，不再自动推进这两学生。后续实验已暂停。
  首次教师vLLM的FlashInfer MoE编译失败（CUDA headers不匹配），日志保留在同名`_01`目录；
  重试使用Triton MoE后端，未修改共享环境。队列不会自动把overlap最佳者选为训练方向。
- [ ] **P3：** 用短程 OPD 后的无 hook dev 收益选择单方向/少方向，比较 E0/真实/随机；
  再决定正式步数与种子。不能以带 hook overlap 最大者直接充当最终方法。
- [ ] **P4：** 加入同示范 full SFT、LoRA、SV-SFT、Compass及组合对照；验证是否仍有独立收益。
- [ ] **P5：** 正式跨 family、多学生、重复种子、早/中/晚干预和成本/保留性实验；
  MATH-500只作冻结测试，选择只用训练/dev。
- [ ] 每次阶段结果、失败、队列改变时更新本节，附实际结果/日志路径；优先交付对上述
  研究问题的证据，避免再次把同 family 初始评测或环境恢复当作主要研究进展。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：按用户要求暂停后续实验。最近诊断队列已于2026-11-10T13:07:03+00:00完成，
随后核查本项目无残留实验进程。证据：`runs/cross_family_steering_probe_20261110_02/campaign.json`、
`runs/cross_family_steering_probe_20261110_02/user_stop_20261110.json`。
下一步先讨论方案、只读复核；未经用户明确恢复运行，不启动新实验。
<!-- CROSS_FAMILY_RUN_STATUS_END -->

## Scope and Startup

The project root is `/home/zhenning/data/working_tmp/steer_opd`. Keep project edits, checkouts, processed datasets, checkpoints, logs, and generated caches inside this directory. Run commands from this root; do not modify neighboring projects.

Start with `README.md`. Read `docs/09_novelty_audit_cross_family_steering_opd.md` before novelty claims, and `docs/06_experiment_chapter.md` through `docs/08_codebase_plan.md` before implementation.

## Environments and Shared Assets

Two existing Conda environments are available, verified on 2026-11-08. Select one for each stage:
Add models/datasets or install packets/create env is allowed. But don't delete/modify dataset/models in /home/zhenning/data/hugggingface_datasets or /home/zhenning/data/huggingface_models.

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

Models are already downloaded under `/home/zhenning/data/huggingface_models`. Datasets are under `/home/zhenning/data/hugggingface_datasets`; preserve this exact spelling. Treat these shared directories as read-only inputs. Inspect their actual layout and reuse local assets; do not redownload or relocate them by default. The download helper's `assets/` default is not the current asset location.

## Project Structure

- `docs/`: Chinese research notes, related work, novelty audit, and experiment plans.
- `experiments/`: planned groups, stage matrix, download manifest, and run-record template.
- `scripts/`: asset-preview and package-validation utilities.
- `papers/`, `ref/`: core/reference PDFs, version manifests, and source snapshots.
- `audit/`: retrieval evidence, source checksums, and validation reports.

An initial training extension is implemented under `src/steer_opd/`, with semantic checks in `tests/test_training_semantics.py` and project launchers under `scripts/`. Read `docs/10_math500_launch_plan.md` and `runs/README.md` for the current pilot scope. Check individual `process.json` and `campaign.json` records before claiming a run is complete. Planned configurations are not experimental results.

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
