# Steering提取与OPD的高效后端核对

2026-11-14用户要求暂缓后续消融，先核对steering论文的提取方法及高效OPD实现。
本次只阅读本地论文、获取公开文档/源码及记录建议，没有安装库、改生产源码或启动GPU/训练。

## 1. 结论与当前任务

提取steering向量不要求HF。EasySteer已有vLLM中间层capture、指定层/逐请求位置选择，以及DiffMean/streaming提取接口；适合用作本项目的另一个提取后端。
OPD也不要求所有阶段用HF：采样和T/Ref无梯度评分可用vLLM；学生参数更新必须经过支持梯度的训练后端，verl已有FSDP和Megatron路径。
HF是模型实现/加载接口，是否高效还取决于attention内核、序列组织、批处理和训练框架；不能把使用HF模型与逐条低吞吐实现等同。

当前只有100题候选、冻结S0、Slerp0.1/层26的8方案零更新筛选；本轮没有backward，OPD训练后端的选择属于后续阶段，不自动启动。
当前所有方向的激活都由同一个Qwen3-4B学生提取，35B教师与0.8B参考只提供文本；不能拿T/R自己的hidden直接相减代替学生空间中的差分。

## 2. 论文和公开代码实际做法

| 来源与本次核对深度 | 实际方法 | 可借鉴的效率点与限制 |
|---|---|---|
| [Steering Vectors as a Training Signal](https://arxiv.org/html/2610.40998v1)，§4.1/§5.1/附录C.2；另读官方acts.py/vectors.py/train.py/generate.py | 同一初始学生分别读取参考与自身回答，正文均值后作差，生成全局/逐题向量。主设置训练前一次提取，训练中固定；每200步刷新只是附录比较 | 官方acts.py仍用HF AutoModelForCausalLM、no_grad、逐题两次前向、output_hidden_states=True收集各层；没有发现专门的高吞吐提取引擎。其生成/评测用vLLM，不能据此称提取也用vLLM |
| [Activation-Conditioned Self-Distillation](https://arxiv.org/html/2609.38342v1)，§3.2–3.4、§4.2成本、A.1 | 发现集轨迹按正确性/预算分组；提取前min(回答长度,B)个token，将每token激活单位化后按组均值，选定层/强度并固定 | 主要依赖预算截断、固定向量及成本摊销。其token加权/先单位化不同于我们的逐题正文原始均值，不应为提速直接照搬。论文Codeforces单列提取五层2.09 GPU-hours，说明提取并非没有成本；该数不外推到本项目 |
| [Steering the Teacher](https://arxiv.org/html/2610.34005v1)，§3.1、§4.4 | 只抽取选定行为位置的hidden，构造两方向，在训练中复用 | 选定行减少保留/搬运量；文中约15分钟指构造轨迹的生成，不是通用提取吞吐。约31%训练吞吐优势相对Purified OPSD，其原因包含减少额外前向，不能称比所有普通OPD快31% |

SV论文官方源码固定到[tiejin98/sv_training@198fdbd](https://github.com/tiejin98/sv_training/blob/198fdbd917e9d7234f734aa8eb754faa408bd2e6/sv_training/acts.py)。
ACSD/ST-OPSD本次依据论文指定章节，未核对其官方提取仓库实现。论文结论、公开代码和本项目建议分别记录。

## 3. EasySteer可用于高效提取的具体接口

官方[Capture guide](https://zju-real.github.io/EasySteer/latest/user-guide/hidden-state-capture/)支持：

- hs.capture：一次提交多个已有文本前缀；max_tokens=1完成prompt前向，不继续自由生成；产生的一个输出token不用作参考回答。
- layers=[26]与逐请求SelectSpec：只保留需要的层、正文token行；不把最后一个token embedding当作正文均值。
- steering=False：明确关闭引擎继承的默认干预。
- CaptureResult：保留真实层ID、request/sample身份与token位置，可用于逐题向量和复用缓存。
- [DiffMean/streaming](https://zju-real.github.io/EasySteer/latest/user-guide/extracting-vectors/)：支持token_pos="mean"及按capture批次累积；不是重新训练一个提取器。

已读源码：EasySteer前端[baecd969](https://github.com/ZJU-REAL/EasySteer/tree/baecd9698f32aadef68b41536f3db477b513a0d7)，其vllm-steer子模块正是项目此前固定的[7837b07d](https://github.com/ZJU-REAL/EasySteer-vllm-v1/tree/7837b07d21c0ddffd043a30001f2c5654290aec7)。
fork capture/session.py明确hidden_states是完整post-layer状态hidden+residual；仍须在实际Qwen3-4B上核验与HF第26层block输出的一致性。

不能省略的效率边界：

- 客户端目前只接受单worker；TP capture会返回分片并由helper拒绝，不能假定自动合并。本项目4B学生单卡提取是合适的首选验收路径。
- API将选中行返回客户端，**没有引擎内mean归约接口**。streaming提取限制的是数值工作内存，不自动限制capture缓冲或消除D2H搬运。
- 捕获正文prompt行时，上游按需跳过prefix-cache读取，因为KV并不包含被跳过的中间层激活；不能承诺同prompt的所有参考都能免费共享计算。
- 只选第26层减少捕获量，**不自动跳过后9层计算**。现有HF路径已经只执行前27个block，且只hook目标层，因此EasySteer是否更快须匹配文本/记录协议实测。
- 图捕获有条件；不能将普通decode的吞吐优势或FULL graph支持直接外推到全正文prefill capture。

迁移必须保留本项目定义：每题/每角色先对正文原始激活池化；T−S/T−Ref共用有效交集；global先平均原始差分再单位化；per_question按本题作差、零方向精确无干预。
现有代码已用同一三角色激活池构造4份bank，8方案不应重复提取。EasySteer默认token_pos=-1或ACSD的token单位化会改变定义，不能照默认值替换。
BF16逐token激活/offset/正文mask等已有留档要求继续保留；流式写盘可以控制内存，不因追求吞吐删掉原协议数据。

## 4. OPD有哪些高效实现

[CompassOPD §5.1](https://arxiv.org/html/2609.10154v1)明确写使用VERL，**vLLM用于采样和教师评分**；其学生更新采用sampled advantage，跨tokenizer按A.4对齐。
因此本项目T/Ref评分仍HF是当前接线选择，并非算法要求；可改为原生vLLM批量固定前缀评分，但不能把学生ID直接交给异构教师。

[verl当前官方OPD示例](https://github.com/volcengine/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/examples/on_policy_distillation_trainer/README.md)明确提供vLLM推理搭配FSDP或Megatron训练，teacher loop使用独立LLM服务器。
其损失有k1/k3、teacher-topk FKL等不同模式；不能把默认teacher-topk=64当作我们原Student Top16配方，也不自动获得Compass span映射、字节门控或学生Slerp训练hook。
已读FSDP损失源码的chunked logsumexp+gather路径，能避免额外完整log-softmax缓冲；注释明确它可能以时延换内存，且不等于消除所有完整logits。

[NeMo RL的OPD文档](https://github.com/NVIDIA-NeMo/RL/blob/5d28fc3c87f48026879ef52ab304391660b9a601/docs/about/algorithms/on-policy-distillation.md)也提供AutoModel训练/vLLM生成路径，可作为另一工程选项。
本次读取的该文档仍将Megatron支持标为未支持，尽管树中有相关配置；因此不依据目录名称断言该路径已经可用。本项目优先复用熟悉的verl，暂不为两卡实验另迁一套大型框架。

普通vLLM/EasySteer推理接口不执行学生optimizer/backward。后续学生更新可用PyTorch/HF+FSDP、或Megatron模型实现；使用FlashAttention/SDPA、padding-free或正确的sequence packing、动态token microbatch、按需要的checkpoint/fused loss来优化。
这些优化要保持因果边界、mask、checkpoint重算时Slerp和缓存概率一致，不能以改变目标或精度协议换出未标明的提速。

verl还提供[异步OPD调度](https://github.com/volcengine/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/docs/advance/async-on-policy-distill.md)，但文档明确one/two-step-off以部分on-policy保证换吞吐。
当前优先一张/最多两张GPU、强调checkpoint/向量/源概率严格绑定，不应自动启用陈旧策略或按异步速率复用旧向量。它只作为将来独立工程选择，本轮不新增实验。

## 5. 当前建议顺序

1. 先补已同意的完整字节概率聚合，将源rollout、教师评分和候选评分缓存串通，保持现有执行锁。
2. 为4B学生接入EasySteer capture，仅第26层、所需正文行、同一S0，构造现有4份bank；保留截断HF路径作数值参考，不重复8次提取。
3. 将35B教师/0.8B参考的无梯度概率评分迁到vLLM批量原生文本前缀，单次评分同时保存实际文本分数、必要Top16和完整字节聚合；完整词表不全部传回CPU。
4. 用户明确同意后，以同题/同文本/同层/同精度/同留档比较HF与EasySteer的提取时间、峰值显存、向量范数与夹角，并做源概率/重放检查；不设人工max_num_seqs。
5. 未来选择学习阶段时优先接verl FSDP，按实际瓶颈再考虑Megatron；sampled-span、Student Top16与Compass目标分别定义，不为后端迁移悄悄换配方。

后续消融、多seed与强基线暂不推进。上述为有源码依据的实现建议，不是已完成迁移或实测提速；完整权重/隔离环境/GPU验收仍未完成。
来源版本与hash见[调研审计](../audit/efficient_backends_20261114/README.md)。
