# Steering OPD方法分支与零更新第一步

最新跨tokenizer实现见[Compass式接口与8方案](16_compass_alignment_eight_way.md)，已CPU验收，GPU运行和官方仓库源码对照仍待完成。

后续代码已更新：[静态mask/Slerp与EasySteer适配](15_static_mask_slerp_implementation.md)已接入新入口并通过CPU验收。
旧overlap保持动态语义；新版EasySteer GPU fork和新配对概率接口仍未验收。以下旧实现快照按各自记录时点理解。

## 2026-11-13最新：静态overlap取代动态逐步门控

当前使用普通无SV源rollout→同前缀教师批量概率评分→一次性冻结idx/mask，注入后不重新判定。
当前零更新搜索在同一源回答前缀重放比较，保留8方案、两种算子、4强度及3单层，共192非零候选和共同基线。
规则与应用阶段边界见[静态协议](12_static_overlap_protocol.md)，网格见[下一步实验](10_overlap_parameter_search.md)。
历史动态代码/结果保留，不再要求采样前probe、在线教师回调或动态引擎改造作为当前主线前置条件。
本次只同步文档：静态适配、Slerp及EasySteer迁移仍待实现/验收；未改生产代码或旧配置，未启动GPU。
后续同回答学习前向注入与静态位置二次生成分别列明，未自动选择训练阶段或新增rollout。

下面代码选项和旧阶段状态属于实现快照；当前门控定义以静态协议为准。

2026-11-13最新能力与完整面板见[实施协议](08_full_scheme_execution.md)。已新增批量精确门控、自适应拆分、
匹配随机、阶段/目标、共同激活池和选参/多seed等；本文以下HF逐请求/未编排状态为当时快照。
新GPU路径尚待验收，EasySteer精确在线门控未移植；用户已恢复运行授权。

> 覆盖范围更正：本文描述已接入的核心开关，不代表完整研究稿全部对照可直接运行。等率/位置匹配随机、共同后端/共同文本公平面板及其他缺口见[逐项实现清单](07_control_implementation_matrix.md)。所有新GPU路径仍未验收。

更新：2026-11-12。用户已授权实现代码分支；当前未运行GPU或恢复实验。
**第一步仍为T−S、全回答预测位置注入、初始学生全局均值。** 当前只准备训练前即时overlap诊断，
不更新参数，完整结果后停在用户检查点。旧v2配置、数据、Base评测与两组历史10步不改写。

## 1. 已实现的独立选项

| 轴 | 参数 | 选项 |
|---|---|---|
| 差分 | direction | T-S、T-R、R-S；两份文本均由同一学生checkpoint读取 |
| 粒度 | granularity | global：原始差分先平均；per_question：保留本题方向 |
| 提取者 | extractor | initial：S0固定；current：本轮起始学生 |
| 刷新 | refresh_interval | current默认每轮；大于1为显式周期刷新变体 |
| 历史已实现注入 | gate | all、动态overlap、独立signed-entropy、random、late；新静态mask适配另见第4节 |
| 方向控制 | direction_control | positive、negative、等范数random |
| 题目控制 | vector_assignment | own、other_question（确定性错配，无同题自映射） |
| 层与强度 | layers、rho | 各层自身提取；多层rho/sqrt(m)匹配名义总强度 |

`src/steer_opd/method_config.py`统一解析/拒绝非法配置。T−R/R−S必须显式指定R模型；
overlap/entropy阈值、随机概率与late边界都要在训练/dev侧冻结，未提供时直接报错，不能暗用测试结果。
[预设](../configs/steering_branch_presets.json)中的null表示尚未校准，不是可以运行的默认阈值。
本轮没有自动确认R的模式/能力差，也没有启动参考模型生成。

## 2. 第一阶段入口与停止点

[第一步配置](../configs/pre_opd_overlap_tuning.json)为：

```json
{"direction":"T-S","granularity":"global","extractor":"initial","gate":"all"}
```

同题100题T/S0文本、5个候选层6/10/13/16/20、rho0.01/0.03/0.1和alpha0；
每层在本层提取方向/尺度，所有候选共用有效配对集合。默认只扫描单层位置；
多层控制内核已支持，但两层网格仍须依据第一轮结果另行冻结，未自动加入。
原32题Base dev前缀已CPU冻结为16题select与16题confirm（seed42），dev128/AIME30保持原样。
参考文本和新方向尚未生成；有效配对数不能提前写成100。

默认命令只做CPU准备，不启动GPU：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_pre_opd_overlap_tuning.py
```

未来用户明确恢复诊断后，专用入口的`--execute`才会依次生成训练参考、提取、select扫描、冻结候选、
confirm确认并输出表格/逐题CSV。结束状态为`awaiting_user_review`、`parameter_updates=0`，
不调用旧campaign.execute/acceptance/train，不进入OPD；正DeltaOverlap也不会自动开训。
训练campaign明确拒绝stage=pre_opd_overlap_tuning配置。

每候选保存原生Top16/概率、逐位置overlap/共享质量、位置桶、逐题长度、配对差值和bootstrap区间。
select按逐题等权DeltaOverlap选前2名，confirm包含这些冻结候选、旧layer13/rho0.03锚点、alpha0、
反向及3个seed随机方向；不使用confirm继续搜索。诊断固定使用同题100题方向，直接调层/层数/强度。
本文按最新计划更新，本次未调整代码或配置。
当前没有新的overlap数值，不能将CPU准备、旧Base结果或小模型测试写成新实验收益。

## 3. 分支数据与持久化

`figure8_extract.py`只读一个指定学生checkpoint，关闭steering和梯度；正文池化排除结构标记，
空白/截断/严重重复/近零方向逐题记录。`direction_bank.py`将全局和逐题原始差分、尺度与版本保存为schema2 bank。
逐题不存在可用方向时明确失败，不偷偷用全局方向补齐。当前学生T−S/R−S会先生成本轮无hook bootstrap，
计入成本；T−R只需要T/R文本。S0固定文本和本轮bootstrap分开目录。

每轮rollout绑定bank文件SHA256、vector_id、question_id与提取checkpoint/步数；
同题n4共享本轮方向，已经生成的轨迹不换新向量。当前学生逐题默认只读本批不同题，
周期刷新为保证同一刷新周期覆盖后续批次，会读取整个固定题池。

激活尺度首次在S0训练侧校准并冻结，后续当前学生刷新不重新估计尺度。
比较不同方向轴时应共用`fixed_activation_scales`和共同文本/过滤预算；未指定该项时各独立run单独估计并记录。
方向文件与pooled中间产物保留，模型/Adam/RNG仍按已有原子提交恢复；不删除共享模型/数据或外部进程。

## 4. 静态mask后端路线与历史实现

当前规范改为源学生普通rollout→教师批量评分→计算固定idx，见[静态协议](12_static_overlap_protocol.md)。
同源前缀诊断使用固定mask的批量因果前向，无需采样前probe/教师回调；固定mask调度适配尚未完成。
现有Method的gate=overlap仍运行动态probe逻辑，不能只将新配置写成overlap就声称静态迁移完成。
生产代码和旧配置本次没有修改，下面区分已核对接口与实际接入状态。

### 4.1 EasySteer可以按位置注入；HF是当前实现选择

官方[Steering指南](https://github.com/ZJU-REAL/EasySteer/blob/baecd9698f32aadef68b41536f3db477b513a0d7/docs/user-guide/steering.md)
及[API字段](https://github.com/ZJU-REAL/EasySteer/blob/baecd9698f32aadef68b41536f3db477b513a0d7/docs/api-reference/steering-specs.md)
支持逐请求ApplySpec的generation_positions、prompt_positions和窗口；已知静态idx可在提交请求前绑定。
加法direct可走图内路径，不需要外部教师逐token交互。首回答预测仍须映射到最后prompt状态，后续按t−1对齐。
当前本地上游vLLM0.32和旧EasySteer环境并不等于已验收的新fork；迁移、逐题绑定及Slerp须另验。

EasySteer这里适用于按已知mask再生成；当前固定前缀零更新诊断可用HF批量重放，不必新增二次自由生成。
若只对已经完成的源回答做学习前向注入，则属于learning-only，训练仍需要带梯度实现，
不能将EasySteer推理注入接口当成HF/训练器学习前向已接通的证据。
二次生成的mask即使来自源回答，仍只是固定位置计划；新回答前缀改变时不动态刷新overlap。

### 4.2 历史动态参考后端（当前不采用）

旧hf_batched_causal_gate在每步计算教师分布和当前步无注入probe，触发后从相同历史做正式学生前向，隔离probe上层KV。
已有动态轨迹的mask/probe/正式概率、RNG和恢复文件保留原义，不重写为静态结果。
它是正确性参考和历史耗时对象，不再要求先移植精确动态gate才能执行当前静态网格。
[历史接口审计](../audit/vllm_steering_gate_docs_2026-11-12/README.md)与[实际耗时](11_online_gate_acceleration.md)仍供追溯。

## 5. 配置与训练验收

[create_steering_variant.py](../scripts/create_steering_variant.py)按预设创建独立、authorized=false且带hold的配置，
不会启动GPU或改旧配置。示例（阈值必须换成训练/dev实际校准值）：

```bash
# 全位置、初始逐题缓存；只创建配置，不运行实验
"$STEER_OPD_PYTHON" scripts/create_steering_variant.py --preset tr_cached \
  --name tr_cached_candidate --reference assets/models/Qwen3-0.6B
```

是否允许学习、预算、R模型及门控阈值仍需按相应阶段冻结，不因创建配置而恢复旧50步排程。
训练入口默认配置现在指向同题Top16 v2；旧独立256题运行仍可显式选择历史配置，继续受暂停保护。

每个可选训练分支首批先运行实际后端验收：vLLM bank保存逐位置完整logits再与HF重放比对；
HF门控在实际受控轨迹上验证完整151665支持的双向KL，随后验证临时反向更新。
这些验收只在用户恢复对应训练阶段后执行，**零更新诊断绝不调用它们**。
同时修正了Top16验收要求全部位置capture、worker只捕获5个位置的不匹配；正式训练不保存完整logits。

目前仅CPU小模型/恢复/调度测试通过，所有新GPU后端验收和性能测量都未执行。
原实验的验收证据不能替代新分支验收。详细范围与命令见[本轮审计](../audit/method_branches_2026-11-12/README.md)。

## 6. SV与逐token记录的实际范围

2026-11-12最新用户要求为注入位置对照提前留档；此前“完整h[t]未保存”的描述对应本轮修改前状态。
零更新诊断现在默认缓存候选层的S0参考读取激活（BF16，含最后prompt状态、分词/offset/正文mask与绝对位置），
并保留各题均值差分和全局均值。只有方向所用层被缓存，当前学生每轮转储需按预算显式开启。

固定前缀与rollout记录逐tokenTop16、完整支持熵、overlap/共享质量、显式mask、方向与模型版本；
HF在线门控的probe与正式受控分布分开。训练评分清理不会删除独立的token_statistics留档。
本轮只补接口与CPU检查，GPU未运行，没有新实际缓存/结果。

第一步仍是同一全局SV逐token使用，不改为每token独立提取方向。后续改变g_t仍需对应前向/生成/训练，
留档用来避免缺失原始证据而重复已做工作，不能替代新条件实验。
字段、目录、CPU读取及复用边界见[逐token留档协议](06_token_level_records.md)。

## 7. 尚未自动完成的研究工作

固定概率随机门控已实现；按动态注入率/绝对位置分布自动匹配的在线滚动控制器未实现，不能据当前随机分支
宣称完成该公平对照。共同SFT/Compass、多seed、多配对、跨tokenizer完整训练也未加入本轮。
这些不影响当前第一步T−S/all/global的代码路径，但后续论文面板仍需独立冻结与验收。
