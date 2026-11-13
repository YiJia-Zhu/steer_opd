# 删除人为序列上限并从断点恢复

2026-11-12。用户明确要求检查其它num_seq设置、全部移除并继续实验。
正式任务已恢复：supervisor PID 1869755，单张GPU2；两组均已提交10步，继续普通组第10步AIME评测。
日志：`runs/figure8_first_run_20261112/supervisor_010_default_scheduler_20261112.log`。
以上是恢复时点记录，后续进度以真实PID、campaign.json及产物为准。

## 修改范围

- 主生成入口不再传max_num_seqs/max_num_active_seqs，不再手写CUDA Graph捕获尺寸或最大尺寸。
  token批预算也交给vLLM默认解析；保留已验收的V1、eager prefill、decode-only CUDA Graph与相同hook。
- campaign移除训练/评测分支中16/64/8等限制，移除runtime.json中的对应字段。
- 当前configs/figure8_first_run.json删除vllm_max_num_seqs与未使用的generation_batch_size。
- 验收入口与性能诊断入口均不再设序列数上限，测试请求数量仍是测试工作量，不是引擎上限。
- generate_math.py、generate_cross_family_calibration.py删除对应上限和外层小批等待，改为每个输入集合全量提交。
  generate_math的旧--batch-size参数只作兼容接收，不再切批；correct-target仍限制选出的记录数量。
- 未改第三方vLLM默认实现、论文参考源码或历史运行证据；未启动SFT/C1等其它排程。

## 恢复与数据完整性

历史checkpoint的完整config hash包含旧运行字段。新增兼容校验仅允许移除这些已废弃字段，
科学配置（数据、seed、长度、采样、损失、方向、评测预算）仍逐项严格匹配原冻结配置。
保留原config.json作为历史证据并沿用其checkpoint identity，不改已保存的模型/Adam/RNG。

生成job升级时同样只允许运行参数改变；旧job清单保存到job_history，完成答案保留原job_hash，
新清单记录接受的历史hash。完成文件的请求ID、prompt和seed均校验后复用。

切换前通过PID、启动时刻、cwd、进程组及STEER_OPD_GPU_ADMISSION核对归属，
只向本项目supervisor和当前生成进程组发送SIGTERM。旧supervisor先退出，避免它自动重试启动重复任务。
控制退出记录在controlled_stop.json，原process.json也保留退出码。

**261份已完成AIME答案逐文件SHA256验证不变**；剩余219份交新进程生成。
共享Base accuracy.json字节不变。两组第10步resume.pt通过CPU mmap检查：310个参数、Adam step10、
配置hash匹配，既有断点完整保留。之后的状态可能已经前进，不把261当作当前完成总数。

## 验收及实际运行

- 24项CPU回归通过，覆盖配置hash兼容、部分完成答案复用、科学配置变更拒绝、错误seed/来源拒绝及原训练语义。
- 当前第10步的普通/steer模型各256条rollout完成GPU验收。两组均一次提交全部256个请求。
- vLLM实际解析：max_num_seqs=1024，max_num_batched_tokens=16384，CUDA Graph最高512、自动51个捕获尺寸。
- 普通组token加权采样log-prob均差0.01641，steer组0.01679；无原0.12阈值失败。
  所有检查行的双向完整分布KL均值<0.005、最大<0.03；两组最大forward KL分别0.00757/0.00929。
- 验证时GPU2继续旧评测、空闲GPU7单独测试，最多两张卡；验证结束释放GPU7，正式恢复仅一张卡。
- 新正式日志已出现108条同时运行，确认没有残留16/64的运行上限；17:04附近观察吞吐约2300–2900 token/s。
  输出上下文会继续增长，不能把短窗口数值外推为整次评测加速比。

新运行策略及校验记录位于运行目录runtime.json与runtime_updates/20261112_default_scheduler/acceptance.json。
仍按原50步、AIME avg@16与原评测节点继续，不改变实验口径。KV不足时由vLLM正常调度/抢占重算。

## 证据文件

- before/、before_state.json：修改前文件和实际进程身份。
- validation_*_job.json、validation_*/、comparison_*.json：实模型生成与完整概率比较。
- checkpoint_compatibility.json：两组真实断点兼容性。
- controlled_stop.json、preserved_answers.json、preservation_verification.json：受控切换及答案/基线hash验证。
- resume.json：新supervisor与日志。

当前源码中仍可搜索到旧参数名称，位于“剔除历史字段”的兼容列表、回归测试和读取实际默认值的日志中；
没有自有LLM调用继续传入这些序列上限。历史JSON/日志中的旧值是已发生运行的证据，不改写。
