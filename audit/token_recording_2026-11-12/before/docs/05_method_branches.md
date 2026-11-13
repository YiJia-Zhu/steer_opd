# Steering OPD方法分支与零更新第一步

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
| 注入 | gate | all、overlap单阈值、独立signed-entropy单阈值、固定概率random、因果late |
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

## 4. 两种生成后端

**全位置分支：vLLM。** 原第一步/旧v2仍走原单全局向量路径；新增bank支持多层和按请求选本题向量。
整批请求一次enqueue，max_num_seqs与token调度预算不手动压低。
全局bank保留decode CUDA Graph；逐题bank当前禁用decode图以正确绑定每请求向量，仍由vLLM连续调度。
这是实现选择，尚未测出吞吐收益，不声称新逐题bank已GPU验收。

**在线/随机/后段门控：HF cached参考后端。** 学生与教师同驻一张GPU，按请求生成并各自复用KV。
只有overlap/entropy需要当前步未注入probe；触发时正式前向从同一旧KV出发，不接收probe上层KV。
不触发时复用probe输出，避免重复学生前向。支持限定为append-only DynamicLayer缓存，其他缓存明确拒绝。
本后端优先提供可检查的因果语义，目前不是vLLM规模的批处理实现，长序列/整批吞吐待测。

门控面板的普通OPD也使用相同HF后端/采样器、无SV/无probe，以免把后端差异误当成steering收益。
全位置面板两组都用vLLM。第一步仅诊断入口不生成OPD轨迹，不受这一训练后端选择影响。

教师在采样前产生的同前缀分数直接保存，之后CPU阶段整理为训练分数/overlap，不重复教师前向。
损失的Top16来自实际受控分布；门控probe的Top16另存，二者不混用。
逐token保存实际gate mask及probe指标，重放按同一向量/mask做因果前向，hook贯穿backward重算；
未触发位置仍计算完整既定OPD损失。门控/方向/候选/教师均停止梯度。

门控生成每32 token保存采样RNG、gate RNG与已发生轨迹/概率；恢复时只重建一次已记录受控历史KV。
每份完整回答仍先提交Top16 tensor、再提交绑定hash的JSON，避免中断时留下“答案完整但概率缺失”。

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

2026-11-12源码复核：当前提取器没有落盘完整逐token隐藏向量h[t]。每题、每角色、每层只保存
回答正文池化均值pools及逐内容token的L2范数norms；norms是一维标量序列，不能恢复隐藏向量。
全局实验的schema2 bank同时保留per_question原始均值差分及全局均值，因此保留了逐题固定方向的信息。
第一步仍只使用全局均值，没有因为保留per_question而改为逐题注入。

全局SV在有效回答预测位置重复使用同一向量；不为每个token生成一份独立SV。
门控HF分支显式保存逐预测位置steering_mask/probe指标；全位置vLLM分支按统一注入规则重建mask，
当前不在每份JSON中另外写全True列表。rollout Top16与固定前缀逐位置overlap/共享质量已有保存接口，
它们是分布/诊断日志，不是逐token SV激活缓存。本轮GPU提取和评分未运行，不能称已产生这些新文件。

未来仅比较逐题缓存或逐token门控可以使用已有方向/轨迹格式；若要改变回答内池化方式、按阶段提取或
研究逐token激活，则需要重读保存的文本，或事先增加可选的完整token激活缓存。
完整缓存还应绑定学生重新分词ID、原序列位置/内容mask/offset、层与checkpoint/hash，不能只保存
过滤后的矩阵后丢失位置映射。T/S回答长度及token内容不同，逐token差分须先定义对应方式，不能直接按序号相减。
本次只核对并澄清，没有增加大体积缓存、修改代码或启动实验。

## 7. 尚未自动完成的研究工作

固定概率随机门控已实现；按动态注入率/绝对位置分布自动匹配的在线滚动控制器未实现，不能据当前随机分支
宣称完成该公平对照。共同SFT/Compass、多seed、多配对、跨tokenizer完整训练也未加入本轮。
这些不影响当前第一步T−S/all/global的代码路径，但后续论文面板仍需独立冻结与验收。
