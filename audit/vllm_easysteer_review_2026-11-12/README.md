# vLLM批量调用与EasySteer适配复核

2026-11-12，CPU源码/环境审计；没有切换运行后端，没有启动新的GPU任务。

## 当前调用是否一次提交全部请求

是。`scripts/figure8_generate.py`一次将全部pending请求传给`llm.enqueue`，首次每步OPD为256条，
每次AIME为480条；恢复时只提交尚未完成的请求。后续`llm_engine.step()`按整个队列连续调度，
每个完成的RequestOutput立即原子保存。没有外层将请求拆为8/16/32条并等待整小批结束。
job中的`batch_size=32`是未用于拆分的遗留字段，不能把它解释为实际提交批大小。

已核对安装的vLLM0.32.0源码：

- `entrypoints/llm.py`的`generate()`明确建议全部prompts放在一个list。
- `generate()`的`_run_completion()`调用`_add_completion_requests()`再调用`_run_engine()`。
- `enqueue()`同样调用`_add_completion_requests()`，`_run_engine()`也循环调用`llm_engine.step()`。

所以当前队列调度遵循同一批量模式。区别是项目在完成请求时落盘以支持被kill后恢复。
同时运行的序列数量另由`max_num_seqs`限制，队列有480条不代表显存必须同时容纳480条。

## 尚未充分优化的部分

现有运行使用V1 runner、decode-only CUDA Graph、eager prefill；prefix caching与chunked prefill关闭，
编译mode=0，FlashInfer sampler因本地CUDA编译器/头版本不匹配而关闭。这些是当前真实设置，
不能宣称已充分优化。功能打开后的速度及带steer概率一致性均需实测，不推定全部开关必然加速。

已实施的运行参数修正是从第9步两组新rollout并发64、后续新AIME并发16，单卡显存比例0.90。
队列仍是全量一次提交。普通组第8/9步的运行观察：

| 步数 | 并发 | 生成token | 生成阶段秒数，不含启动 | token/s |
|---|---:|---:|---:|---:|
| 8 | 16 | 384935 | 226.42 | 1700.12 |
| 9 | 64 | 389066 | 144.67 | 2689.40 |

题目流和checkpoint随训练发生变化，上表为吞吐观察，不是严格固定输入/模型的因果对照。
此前长上下文小基准仅约1.17倍提速，见运行目录`recovery_20261112_replay/performance.json`。

## EasySteer可以接什么

2026-11-12查到的官方main仍为`baecd9698f32aadef68b41536f3db477b513a0d7`，
依赖其vLLM0.29.0/V2 fork，子模块commit为`7837b07d21c0ddffd043a30001f2c5654290aec7`。
获取地址、hash及源码保存在本目录；不是只依据旧记忆判断当前接口。

官方`direct`支持hidden-state加法、full-graph additive kernel、逐请求向量和独立prompt/generation位置。
本轮映射可以写为：

```python
from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, VectorSpec

spec = SteeringSpec(vectors=[VectorSpec(
    data=DirectionVector({13: delta}),
    algorithm="direct",
    scale=1.0,
    normalize=False,
    apply=ApplySpec(prompt_positions=[-1], generation="all"),
)])
```

其中`delta`为项目已按rho=0.03标定的实际向量，不能再乘一次0.03。
最后prompt状态预测首个回答token；仅写generation="all"会漏掉首个回答预测。
这是已核对的API映射示例，不是已完成GPU验收或已切换运行后端。

合理分工是EasySteer/vLLM负责steered rollout与可选提取，HF/PyTorch继续进行固定方向下的OPD学生更新。
官方`easysteer.training`冻结base、训练direct/LoReFT adapter；该接口不直接等价于本项目固定方向更新学生。
当前官方文档还明确其训练封装不支持gradient checkpointing和精确Trainer optimizer恢复，
因此不以直接调用该训练入口替换本项目已有的优化器/RNG断点。

本机conda easysteer实际上是旧0.1.0、Python3.10、Torch2.10.0、Transformers4.57.6、
旧开发版vLLM，editable路径指向另一个已有项目。当前OPD为项目隔离的vLLM0.32.0/Torch2.13。
环境存在不等于新版接口可直接导入；没有修改共享环境或邻近项目。
真正迁移须使用隔离后端，并核对151665-token公共输出支持、FP32输出logits、层输出/残差边界、
首回答mask、缓存/完整重放及断点语义，再比较相同输入的吞吐。当前没有EasySteer速度优于现实现的本地证据。

## 现有故障恢复复核

本次确认steer第8步原始分布检查通过：触发短回答回退的`016_3`两个位置完整分布KL约0.00103/0.00107，
捕获的采样log-prob与实际轨迹完全一致；已完成第8步并进入第9步。证据为运行目录
`steer/steps/0008/replay_validation.json`及`steer/latest.json`。不把运行恢复或吞吐改善当成方法有效。
