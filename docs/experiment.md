# 当前实验：静态双指标八方案

T为Qwen3.5-35B-A3B，Ref为Qwen3.5-0.8B，学生为Qwen3-4B。固定S0、Slerp alpha=0.1和第26层输出（36层的75%，0-based）。目前只做同源前缀、无梯度、0次OPD更新，下一步须用户明确同意后执行。

[static_dual_overlap.json](../configs/static_dual_overlap.json)是当前默认模板。[run_static_panel.sh](../scripts/run_static_panel.sh)提供手动启动入口，默认参数与模板一致。8方案为T−S/T−Ref × all/static_overlap × global/per_question，加共同无SV基线；不会自动展开层/强度网格或把两种指标扩成16方案。

注入位置轴分成两半：**4个all方案在全部回答预测位置注入，不使用overlap阈值；4个static_overlap方案才按所选overlap指标冻结注入位置。** 两边都交叉T−S/T−Ref与global/per_question。
全部8方案仍计算、记录并绘制overlap与P_overlap，用于评估干预效果。gate_metric只控制static_overlap组的位置选择；rank_metric用于方案比较，不控制all组的注入位置。零方向或零强度仍精确无干预。

| 阶段 | 执行内容与共享范围 |
|---|---|
| 训练侧题目 | 保存原100题；按seed42提前冻结70/30 select/confirm。prompt、参考文本、激活和对齐过滤后实际题数另报，划分不重抽 |
| 参考文本 | T、Ref和S各一份，按各自原生tokenizer生成。三角色文本共用质量过滤 |
| 方向 | 同一S0读取三种文本，只capture第26层prompt末与正文状态；正文原始激活先逐题池化，差分后全局/逐题构造4份bank，供8方案复用 |
| 源轨迹 | S0无SV生成，保存实际学生tokens、原生Top16、原始行为logp、完整下一字节概率与全支持熵 |
| 监督缓存 | T/Ref使用自身tokenizer批量评分同一源文本。共同UTF-8安全span、native平均logp及家族差分依据Compass A.4独立实现 |
| 静态mask | select上分别校准两种指标阈值；只用gate_metric所选的一项冻结strict-< mask，注入后不重判 |
| 八方案评分 | 统一EasySteer固定前缀前向；共同无SV基线单独评分，保留源rollout与重放差异 |
| 输出 | overlap与P_overlap分别统计和画图。rank_metric只决定一次排名，select前2候选冻结后进confirm，结果供用户检查 |

`overlap`是Rethinking的跨family Top16适配：共同span起点的完整token字节事件交集/16。`P_overlap`是完整词表→258个下一字节/EOS/控制类别→sum(min)。定义与局限见[完整方案§12.2](Steering_OPD_完整研究方案.md#122-两种跨family-overlap并行记录)。两个值均在[0,1]，差值以百分点报告。

当前gate_metric和rank_metric均为overlap；P_overlap始终记录。可选择P_overlap而不新增方案。指标缺失位置用NaN，排除统计；静态mask不触发，全位置组可触发并单独标注。span起点值向学生span广播，不声称每个异构token边界都比较到了相同文本状态。

所有请求seed统一42，Python/NumPy/PyTorch/vLLM均固定42；不使用42+i。恢复点为完整阶段或完整候选：一次引擎调用连续完成，返回后统一保存和提交；中断重算未提交整批，已完成阶段/候选复用。恢复、留档和实时进度见[实现说明](implementation.md#可复现性恢复与进度2026-11-14)。

全局向量按原始逐题差分平均后单位化，逐题零方向精确无干预；不回退别题/全局方向，不改为逐token单位化。学生在全部提取和评分期间冻结。

当前execution_hold=awaiting_user_approval。CPU准备可运行：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_compass_static_panel.py --config configs/static_dual_overlap.json
```

这只准备题目/划分和带执行锁的job，不运行模型。35B权重已由用户准备，14分片结构/索引及本地路径检查通过，启动时记录完整内容哈希；EasySteer实际GPU数值与性能待验收。解除锁时须重新生成仅CPU准备的job，避免旧冻结配置身份冲突；有模型产物的run不得修改身份后续接。

用户同意后仍先核验源概率/重放、HF与capture的同层状态与正文均值、Slerp零强度/移除恢复、首预测/t−1/EOS/padding、逐题绑定和共同输出支持。优先一张空闲GPU，35B阶段最多两张；不停止其他项目。所有vLLM入口保持默认序列调度，不传max_num_seqs或人为CUDA Graph尺寸。

## 一条命令启动

先编辑[scripts/run_static_panel.sh](../scripts/run_static_panel.sh)开头的GPU_IDS、SCHEME_IDS、RUN_NAME和超参数，然后在项目根目录执行：

```bash
bash scripts/run_static_panel.sh --check   # 只检查和打印配置，不查询或加载GPU
bash scripts/run_static_panel.sh --run     # 明确启动；再次执行同一命令即恢复
bash scripts/run_static_panel.sh --status  # 只读查看本run进度
```

GPU_IDS是nvidia-smi显示的物理编号，例如0,1；TEACHER_TP=2时至少填两张。单卡可设GPU_IDS=0和TEACHER_TP=1，但需要能容纳教师权重和引擎缓存。脚本仅在指定池内等空闲卡，单卡阶段优先列表中第一张可用卡，不跳到池外。WAIT_SECONDS控制每阶段等卡时间。

SCHEME_IDS默认all，也可填1或1,4,8。所有选择均包含共同无SV基线，仍建立相同T/Ref/S共同池及4份bank；confirm最多保留CONFIRM_TOP_N个已选方案。方案编号如下：

| ID | 差分 | 注入位置 | 方向粒度 |
|---|---|---|---|
| 1 | T−S | all | global |
| 2 | T−S | all | per_question |
| 3 | T−S | static_overlap | global |
| 4 | T−S | static_overlap | per_question |
| 5 | T−Ref | all | global |
| 6 | T−Ref | all | per_question |
| 7 | T−Ref | static_overlap | global |
| 8 | T−Ref | static_overlap | per_question |

ALPHA、LAYER、gate/rank指标、静态目标比例及生成长度/采样参数会写入实际配置；seed固定42。改变科学配置须更换RUN_NAME，不覆盖旧结果；恢复时可更换GPU_IDS。每次启动保存实际shell文本和GPU选择。也可临时覆盖变量，例如：

```bash
GPU_IDS=2,3 SCHEME_IDS=4 RUN_NAME=static_scheme4 bash scripts/run_static_panel.sh --check
```

--run先检查权重及隔离环境，然后按学生/Ref/教师分别执行小样本HF参考与EasySteer验收，两个后端分进程释放显存。验证包含原生概率、完整下一字节概率、学生同层capture、不同请求方向/静态与全位置mask、零强度/移除恢复，以及学生实际采样与固定前缀重放。验收成功才生成runs/RUN_NAME/launch_config.json及绑定源码/权重的凭据并进入面板；失败即停，不自动放宽容差或切换后端。该检查一次完成后可恢复复用；它是小样本验收，正式长序列性能仍须实测。原模板和旧仅CPU准备目录不解除hold。

输出在runs/RUN_NAME/：startup.log保存启动验收日志，console.log保存正式面板日志，acceptance/保存各角色验收，select/和confirm/保存双指标及图。--status也显示启动验收进度。脚本不下载权重，不开启OPD或后续消融。

本轮没有学生梯度更新。OPD目标、learning-only与静态mask二次生成、裸学生能力评测、刷新/消融和多seed均留后续，不把固定前缀指标改善写成学习收益。
