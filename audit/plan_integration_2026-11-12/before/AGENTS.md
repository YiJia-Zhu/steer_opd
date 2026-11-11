# Repository Guidelines

## Execution Hold — Latest User Direction

2026-11-11最新任务调整：暂缓本地作者SFT对照及其权重下载，SFT结果由用户阅读
论文了解。当前仅比较Base→OPD与Base+steer→OPD的accuracy/overlap。
教师固定Qwen3-4B Non-thinking，学生为真正的Qwen3-1.7B-Base，不能用普通
Qwen3-1.7B代替。已授权这一任务的准备工作，
C1不启动。用户最新已同意OPD改用原OpenThoughts3题源，先抽100题；AIME2025
30题仅评测。用户要求复述细节、检查后再启动，因此仍不下载或运行GPU。
当前待确认建议：100题固定池循环至50个trainer step，batch64/n4，等效32遍；
原fork drop_last=True，不能将100题一轮误写成100步或长期只训练前64题。
两条主曲线，无本地SFT训练或初始化；steer校准复用这100个训练题，
第14层（0-based13）及rho0.03为项目候选。最新用户澄清：总方案保留方向方案1和2，
两者主设置均在整个OPD训练期，对所有参与loss的回答token预测状态注入方向；
rollout、评测、部署不注入。此前“前10步干预、后40步撤去”降为后续日程消融。
目标是检验能否承担模式适配作用、减少或省去独立SFT冷启动，尚不能声称已替代SFT。
当前两组先导仅含普通OPD/方案1；总方案中的方案2未被删除，是否加入首轮另行冻结。
最新讨论新增一个独立设计轴：可按无hook师生overlap/熵差选择是否注入，但当前
仍以全有效回答位置固定注入为基线，不认定每个位置必然需要steer。门控候选先
考虑overlap，熵差作诊断；指标从相同普通学生前缀、更新前分布计算并冻结/停止梯度，
不据steer后的指标反向选择本次注入。低overlap/大熵差均非必要性或收益证明。
动态门控需同注入比例随机位置对照、共享质量/学习收益及额外评分成本；暂不启动或
自动加入首轮。方向方案1/2与位置门控是不同设计轴。
这些为待确认的小数据先导设置，不能声称完整复现论文单轮30K训练。配置与证据见`experiments/rethinking_figure8_20261111.json`
及`docs/03_execution_plan.md`最新任务说明。

最新资产复核：用户已补齐软链接，`assets/models/Qwen3-1.7B-Base`与
`assets/models/Qwen3-4B`权重文件、配置及tokenizer存在，safetensors头/文件长度
检查通过；不是模型加载或来源哈希验收。原30K OPD parquet仍在，AIME2025软链接
有效且共30题；本轮模型和数据均无需再下载。作者`lllyx/Qwen3-1.7B-SFT`未找到，
但最新两组方案不需要它。论文SFT曲线只作背景，不作本轮同条件对照。
记录见`audit/rethinking_figure8_2026-11-11/current_assets.json`。未执行下载或GPU任务。

当前仅进行方案与文档准备，尚未恢复实验或模型/数据下载。
实验有效性、数据划分与实现验收均须按新任务重新确认，不自动启动其他排程。
正式跨family训练收益尚未建立。

论文机制复核：SV Training §4.1/4.3在teacher-forced监督/蒸馏训练中注入方向，
测试不注入；§5.1 GRPO SV-Rescue使用正确−错误方向对rollout打分、修正错误
rollout的advantage，恢复all-wrong组的信号，不等同于前10步hidden hook。
Rethinking §5.1是OPD前的独立SFT冷启动，§5.2是OPD期间的模板/题源对齐。
共同动机可类比训练信号可用性，具体机制与早期日程不能混为论文已有结论。

## Research Mission and Authoritative Documents

核心问题：**能否用少量教师示范提取学生自己的激活方向，改善跨family教师—学生
之间的OPD适配，减少独立SFT冷启动需求，并使无steer评测的学生保留学习收益？**

以实验结果决定范围，不预设steering有效。“同一教师、多个学生”只是可选对照，
不是主问题或首轮硬要求；每次训练冻结教师，不等于所有实验必须共用一个教师。
跨family在排查分词/实现/长度问题及约定预算后仍没有可靠收益，就回到Rethinking
共享词表设置，研究更窄的OPD学习困难与冷启动成本。回退须重做对应实验，保留
跨family负结果，不能只换措辞或继续宣称跨family有效；共享词表也无收益就接受负结果。

- 方法和完整实验逻辑：`docs/01_research_plan.md`，覆盖方案1/2、概率接口、基线与规模控制。
- 论文配对、数据与证据：`docs/02_paper_settings.md`。
- 工程现状、验收和恢复：`docs/03_execution_plan.md`。
- 近邻与新颖性边界：`docs/09_novelty_audit_cross_family_steering_opd.md`。
- 导航：`docs/README.md`。

每次有实质进展、失败或用户纠正时更新本文件和对应主文档。状态以实际日志、
checkpoint和process/campaign记录为证；计划、启动、诊断完成不能写成方法有效。

### Model and Data Selection

优先对应Rethinking OPD、Simple-OPD、CompassOPD实际师生连线和数据，不随意拼接或下载。

- C1主要异构配对：Qwen3-30B-A3B-Instruct-2507 / Qwen3-0.6B参考→Granite4.1-3B；
  Compass表2已有配对，原题源为OpenThoughts-114k教师SFT与DeepScaleR-Preview OPD。
- F1回退：Qwen3-4B non-thinking→Qwen3-1.7B-Base，Rethinking §5.1的SFT修复设置；
  对应OpenThoughts3题源，Base缺口先核实，不以Instruct替代。
- C2为可选同family对照：Qwen3-30B→Qwen3-4B，Compass已有配对，DAPO-Math；非首轮硬要求。
- C3为有主结果后再加的规模控制：Qwen3.5-4B / Qwen3.5-0.8B参考→Qwen3-4B，4B教师连线见Compass图1(b)。
  新加Compass/steering组是本项目实验；同一广义Qwen谱系的代际变化不代替C1独立family证据。
- 总参数、激活参数、文本主干和可训练参数分别报告。A3B不等于3B总参数。
- 严格独立family的3B→3B可讨论Ministral-3-3B→Granite4.1-3B，但该连线是新配对，
  教师优势及更弱同家族参考未落实，非首批，不自动新增下载。
- JustRL→R1-1.5B是Rethinking/Simple共同配对，R1-7B→R1-1.5B为Rethinking机制对照。
  若转向回退路线，可在它们上研究共享词表下的效果，不能把结果称为跨family证据。
- 已有Qwen30B/0.6B/4B、Qwen3.5-0.8B、R1/JustRL资产优先复用；Granite、Qwen3.5-4B
  及共同SFT语料缺口先核实。普通Instruct不替代Base，普通8B不替代Simple的RL后8B。
- MATH-500只作冻结测试。DAPO须按新任务重新核验去重与划分；本地DeepScaler
  40,315题与官方题集的等价性待核验。OpenThoughts-114k与OpenThoughts3不同；
  现有30K OT3 OPD题不冒充200K SFT示范。

### Method Boundaries

1. 方案1：教师T和学生S0在校准题上生成文本；同一个冻结S0读取两类回答，
   `v_total = mean(Phi_S0(x,y_T)-Phi_S0(x,y_S))`。不直接相减不同模型hidden states。
   通常每题两次学生prefill，不能声称整套训练只需一次prefill。
2. 方案2：同一S0再读取教师同家族弱参考R的回答，统一三文本池化/样本权重后构造
   `v_cap=T-R`、`v_offset=R-S`及`v_total_triplet=v_cap+v_offset`。
   单独方案2只需T/R两份文本，通常两次学生prefill；S自身回答用于方案1与偏移对照，
   完整比较才读取三类文本。R对训练轨迹的概率评分只在使用Compass目标时需要。
   这些方向是否有用由实验判断，不先认定为纯能力或纯风格。
3. 每个学生独立提取自身方向；初始checkpoint改变后重新提取。先单方向，再2/4方向；
   方向和层/强度最终按真实短程更新后的无hook dev收益选择，不按即时overlap最大选择。
4. 方案1/2均在整个OPD阶段的loss前向，对所有有效回答token预测状态加固定方向，
   hook保持到backward重算结束。普通学生rollout、评测与部署无hook；仅前H步
   启用作为后续日程消融。严格核对首回答预测位置t−1和mask，padding/无效位置排除。
5. 异构tokenizer先做共同文本span/有效候选对齐。多token平均log-likelihood是评分，
   不是整词概率；保留空格、验证UTF-8/round-trip、mask未对齐位置并报告覆盖。
6. 普通学生采样不等于steered policy采样，不能将未修正的sampled reverse-KL/策略梯度
   公式直接加hook。当前方法面板设计为显式有限候选RKL及Compass式(11)候选目标，
   原版无hookspan采样基线另列；这些新训练接口尚待实现验证。
7. 候选归一化目标、原论文目标及完整词表/文本KL分开命名。概率评分及teacher/reference
   调用成本仍存在，方向发现只用文本不意味着整个系统是纯黑盒。
8. 面板A比较原始学生的full SFT、LoRA、SV-SFT、方案1/2和随机；面板B共同SFT后比较
   Vanilla、Compass与方向组合，锚点和方向均来自共同初始化。相同示范、合理调参、总成本公平。

### Literature Interpretation

Rethinking/Simple已报OPD配对在兼容词表谱系内。Rethinking §8确实讨论Qwen→LLaMA
的数据/tokenizer/架构混杂，未完成本文自己的异构实验；不要说它完全未提到cross-family。
Simple GPT-5.5文本SFT后仍接75步OPD，逐token教师为原Qwen RL教师；这是跨family
SFT→Qwen内部OPD。Overlap是相同学生前缀的top-k集合交集/k；共享质量、任务收益
与范围必须同时报告。Rethinking梯度抵消与本项目低维相容假设均非已证实机制。

### TODO — Update with Evidence

- [x] 按用户纠正重写主问题：跨family适配先试，效果不好时回退Rethinking设置；语言尽量简单。
- [x] 收敛docs为总方案、论文证据、执行验收和新颖性审计，删除重复探索稿。
- [x] 写清方案1、Compass启发的方案2、三文本分解、4B级规模控制与严格family限制。
- [x] 写清目标/采样分布、原版与adapted、SFT/LoRA公平比较、成本和失败处理。
- [ ] P0：讨论并冻结最小面板、资产/数据版本、预算与主要终点；当前不下载/运行。
- [ ] P1：用户恢复后完成正式配对无干预overlap、共同事件覆盖和无辅助训练可学习性。
- [ ] P2：正式配对的总差/增量/偏移与真实/随机/负方向诊断，验证新架构hook。
- [ ] P3：真实短程更新后无hook选择，独立dev-confirm与撤去后的曲线。
- [ ] P4：面板A/B强基线、原版span参考及同信息/成本比较。
- [ ] 路线判断：用dev检查跨family真实学习收益；无收益时排查后回退F1/R1，不自动铺开多学生。
- [ ] P5：仅在对应路线有依据后做重复seed/冻结测试；C2/C3为后续控制，主张限于实际范围。

<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->
执行状态：OpenThoughts3中100题做OPD、AIME2025仅评测已明确；50步及steer细节待用户检查确认，没有新增GPU任务。
跨family及其他下载/训练任务暂停，不自动启动。
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
