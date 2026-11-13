# max_num_seqs：官方文档、版本源码与本机默认值

2026-11-12。用户要求查证并汇报，偏好省略该参数、使用默认或较高上限。
本次没有更改正在运行的配置，也没有加载新模型或启动GPU任务。

## 结论

当前vLLM 0.32.0、单张RTX PRO 5000 72GB、离线LLM调用、默认performance_mode时，
**省略max_num_seqs会选择1024**。本机nvidia-smi给出73415 MiB=71.694 GiB，
命中源码的“显存至少70 GiB且名称不含a100”分支。对应默认max_num_batched_tokens为16384。
不是SchedulerConfig测试用128，也不是较小显存分支的256。

`resolved_defaults.json`保存了使用真实硬件数据、在CPU上调用安装版get_batch_defaults所得结果；
不是GPU启动/吞吐验收。`source_comparison.json`确认本地arg_utils及scheduler与官方v0.32.0源码逐字节一致。

## 对当前工作负载的建议

| 任务 | 一次提交请求数 | 首选 | 如需固定高上限的候选 |
|---|---:|---|---:|
| 每步学生rollout | 256 | 省略max_num_seqs，默认1024 | 256 |
| 每次AIME评测 | 480 | 省略max_num_seqs，默认1024 | 512 |
| 每个角色的方向校准 | 256 | 省略max_num_seqs，默认1024 | 256 |

表中是调度上限候选，尚未实测哪个最快。256/512覆盖本批请求数量，不是显存容量推导出的最佳并发。
请求继续全量一次入队，max_num_active_seqs保持默认None，避免另一个人为小上限。

## 为什么不能用17条满长容量直接限制为16

当前评测日志的550342个KV token / 32024长度 = 17.19条，是所有序列均达到最大长度时的容量换算。
vLLM实际按已计算token、当前要计算的新token和lookahead分配KV块，不预留每个请求的全部未来输出。
即便full_sequence_must_fit启用，该处也是检查request.num_tokens，即当前已知序列，不是输出上限。
因此最大长度容量不能当作max_num_seqs的硬上限；此前据此压到8/16过于保守，应纠正。

不过自动调度也不等于自动找到最佳吞吐。运行中KV不足时，scheduler会抢占请求，
V1默认RECOMPUTE；恢复时可能重算前缀。官方建议在频繁重算时减少max_num_seqs/max_num_batched_tokens。
因此应以同批输入的完整耗时、生成token/s和抢占/重算计数比较默认与候选，不只看GPU显存利用率。

## 实现上的关联项

当前项目不仅手写max_num_seqs，还将CUDA Graph capture尺寸/上限绑定到同一小值。
迁移到默认上限时，应同时解除这组小图尺寸限制，使用vLLM合理的自动图尺寸选择；
不能只删LLM参数，却留下最大16的图路径，或把65–480请求一概padding到1024。
这不要求同时改变科学配置、输出支持、精度、采样参数或正在进行的已冻结job。

当前开启方式仍为V1/decode-only graph/eager prefill。开启其它默认优化或切换V2是另一个运行改动，
不能把它们与只修改max_num_seqs的速度收益混为一个已验证结论。

## 来源

- [官方LLM.generate文档](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/)：全部prompts放一个list，按显存约束自动batch。
- [v0.32.0默认值源码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/engine/arg_utils.py)：get_batch_defaults及_set_default_max_num_seqs_and_batched_tokens_args。
- [v0.32.0性能文档](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/configuration/optimization.md#preemption)：KV不足时抢占/重算及调参建议。
- [v0.32.0 scheduler源码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)：动态allocate_slots和_preempt_request。

latest文档可能与安装版有差异；数值结论使用固定v0.32.0源码，不套用latest的内存默认值。
检索快照、URL、hash、CPU验证结果均在本目录。
