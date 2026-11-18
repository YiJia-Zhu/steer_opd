# 文献依据与阅读边界

本项目的来源结论与本地实现分别记录。论文有方案/效果，不代表当前配对已独立复现。

| 来源 | 本次使用的结论与核对范围 |
|---|---|
| [Rethinking OPD](../papers/2604.13016v2_rethinking_on_policy_distillation.pdf) | Top-k集合相容性与长轨迹机制；另核对v2 §2.2／2.3／6.3及固定官方commit的采样、Top-K和旧概率实现，详见[研究稿§4.1.1](Steering_OPD_完整研究方案.md#411-sampled-tokentop-k训练与top16监控)。前缀兼容匹配／概率质量是本项目跨tokenizer扩展，不能归给原文 |
| [CompassOPD](../ref/2609.10154v1_compass_opd.pdf) | 教师同家族likelihood shift；§5.1使用verl训练、vLLM采样/教师评分；A.4跨分词共同span平均原生logp与广播。本项目依据论文独立实现，未定位用户所指官方仓库 |
| [Steering Vectors as a Training Signal](../papers/2610.40998v1_steering_vectors_as_training_signal.pdf) | 同一学生读取参考/自身文本、全局/逐题方向及C.2刷新。官方acts.py使用HF逐题提取，不据其vLLM生成推断提取也用vLLM |
| [ACSD](../ref/2609.38342v1_acsd.pdf) | §3.2–3.4预算/固定激活向量；token先单位化不同于本项目原始正文逐题均值，不为效率直接替换 |
| [Steering the Teacher](../ref/2610.34005v1_st_opsd.pdf) | §3.1/4.4选定行为位置与固定方向降低成本；教师干预与原学生干预分别比较，论文成本不外推到本项目教师镜像路线 |

[EasySteer固定前端](https://github.com/ZJU-REAL/EasySteer/tree/baecd9698f32aadef68b41536f3db477b513a0d7)与其固定fork的capture/selector/logits源码已核对；指定层/行capture并非服务端mean归约，不自动截断后层，也不能用embedding接口替代block输出。

[verl官方OPD](https://github.com/volcengine/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/examples/on_policy_distillation_trainer/README.md)支持vLLM推理与FSDP/Megatron训练；默认teacher-topk损失不自动等价于旧Student Top16或Compass span目标。推理接口不执行optimizer/backward，本轮不迁移训练目标或启动异步陈旧策略。

[NeMo RL OPD](https://github.com/NVIDIA-NeMo/RL/blob/5d28fc3c87f48026879ef52ab304391660b9a601/docs/about/algorithms/on-policy-distillation.md)作为工程候选；所读版本文档未承诺Megatron支持，不依据目录名称断言可用。

固定源码位于[EasySteer清单](../ref/easysteer_runtime/manifest.json)和[实现来源](../ref/implementation_sources/manifest.json)。核心与邻近论文PDF/正文/hash仍保留在[papers](../papers/manifest.json)、[ref](../ref/manifest.json)及ref/archive；删除的是旧运行/重复记录，不删除文献。
