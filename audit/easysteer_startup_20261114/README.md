# EasySteer启动排查与小样本验收（2026-11-14）

用户停止初始化阻塞后授权GPU排查。本轮仅做诊断与小样本验收，正式八方案未启动、参数更新0；没有重置GPU或停止其他项目进程。当前代码仍使用学生读取T/R文本的差分，本记录不覆盖完整研究稿新增的R侧映射融合方向。

## 原始阻塞与修复

`static_panel_v1`的HF学生参考在14:16:23至14:16:30完成，耗时6.95秒；EasySteer于14:16:34进入初始化，最后心跳15:08:34，仍在初始化（52分钟），并未生成正式结果。

1. [fork复现日志](fork_attempt/stdout.log)的faulthandler栈停在V2 `buffer_utils.py:48`的pinned CPU `torch.zeros`，上层为UvaBuffer/states/model_runner。此前主进程已执行PyTorch CPU计算并启动心跳线程；换成spawn后立即越过该位置。统一使用spawn，初始化增加600秒限时和堆栈记录。
2. [后续编译错误](spawn_attempt/stdout.log)证明CUDA compiler与headers不兼容。隔离环境nvcc为13.5/13.2，headers为13.0；改为只读`/usr/local/cuda-13.0`工具链，并验证compiler/header/torch CUDA构建一致。未安装新包或修改系统环境。
3. 原生概率统计hook移至上游compile/warmup之后安装，dummy logits不作为实验数据；启用此前遗漏的`enable_steer_vector=True`。
4. 完整词表按1024个词元分块FP32累加后归并，避免大桶scatter丢失尾部质量；不重新归一化、不放宽概率质量容差。RPC改为显式shape/dtype二进制字段。
5. 学生BF16批形状波动影响零干预/重放检查，学生生成、capture、评分统一启用vLLM batch-invariant；自动调度和CUDA Graph保留，没有序列数上限，正式规模性能代价未测。

旧入口强制HF+EasySteer三角色对照；HF只是附加的数值参考，并非正式计算的必要阶段。当前默认`HF_PARITY=0`，仅运行EasySteer原生检查；显式设1才增加HF模型前向。Tokenizer/config读取和已有logits上的PyTorch统计不加载HF模型。

## GPU证据

| 角色/路径 | 结果与范围 | 证据 |
|---|---|---|
| Qwen3-4B学生 | 通过capture位置/token/有限状态，全位置与静态mask批量/单请求、alpha0/空mask/移除恢复、源采样重放。观察token logp比较差为0，最大byte TV约6e-8；总进程41.19秒，不能推导正式吞吐 | [report](native_validation_final/acceptance/student/easysteer/report.json)、[process](final_student_attempt/process.json) |
| Qwen3.5-0.8B Ref | 原生评分与完整概率投影通过，重复观察token logp差0，最大byte TV约1.2e-7；总进程158.20秒，包含冷启动 | [report](native_validation_steer_enabled/acceptance/reference/easysteer/report.json)、[process](native_reference_attempt/process.json) |
| Qwen3.5-35B-A3B教师TP2 | 曾完成281.81秒初始化并完成两次评分，但重复概率比较超出原严格阈值：logp平均差0.085786、最大0.313827，最大byte TV 0.045315。未将该次标为通过 | [评分日志](final_teacher_attempt/stdout.log)、[process](final_teacher_attempt/process.json) |
| 教师batch-invariant尝试 | 引擎明确报错`not supported for GDN_ATTN`，不能沿用学生策略 | [日志](teacher_invariant_attempt/stdout.log) |
| 教师再次TP2启动 | 停在16:00:22 FlashInfer autotune阶段；16:07:22停止自有作业。只能确认阻塞阶段，未证明内部具体死锁机制 | [日志](teacher_contract_attempt/stdout.log)、[process](teacher_contract_attempt/process.json) |
| TP关闭autotune、跳过图像/视频预热 | 代码已改，GPU2无计算进程、显存0但利用率持续100%，准入拒绝启动；最新路径尚未完成GPU复验 | [准入快照](teacher_no_tune_attempt/no_idle_gpu.json)、[最新状态](gpu_state.json) |

默认T/Ref验收范围已明确改为原生支持/Top16/观察token/完整投影有效性，重复漂移留作`repeat_drift`诊断；它们只评分一次并冻结同一SHA缓存，供全部学生候选复用。该策略并不证明教师跨批数值一致。学生所有严格比较保持原阈值；可选`HF_PARITY=1`仍严格比较三角色，不把失败自动改为通过。

以上S/Ref结果属于诊断当时的源码版本，各目录保留provenance和complete清单；不能直接搬到新正式run充作当前源码的完整三角色验收凭据。正式启动仍由入口生成与当前代码/权重绑定的凭据。

## 验证与剩余工作

[CPU测试日志](cpu_tests.log)：123 passed，17 warnings；[包检查](package.log)、[CPU运行时检查](runtime_cpu.log)、[手动启动预览](preview.log)均通过。机器可读汇总见[validation.json](validation.json)。

需要GPU2恢复正常，或用户指定另一空闲卡池后，复验教师TP关闭autotune/纯文本启动路径。保持shell默认GPU池2,7，不绕过准入、不自动重置或使用池外卡。正式长序列吞吐、逐token实际扰动范数、完整八方案、HF可选跨后端对照和后续OPD均未在本轮完成。
