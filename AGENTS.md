# Repository Guidelines

## Current authorization and experiment

2026-11-14用户授权实现两种overlap、双图/留档、EasySteer提取/评分与缓存集成，并清理过期MD和旧实验结果。允许代码/CPU验证及项目隔离环境准备；GPU、模型生成/评分/训练仍须用户下一步明确同意。教师权重由用户手动下载，代理只做完整性核验，不运行下载逻辑。execution_hold=awaiting_user_approval，不能自动解除。后续消融暂缓。

固定T=Qwen3.5-35B-A3B、Ref=Qwen3.5-0.8B、S=Qwen3-4B。Slerp alpha=.1，75%单层26（36 blocks，0-based）；T−S/T−Ref × all/static_overlap × global/per_question完整8方案及共同无SV基线，S0冻结、同源前缀、0更新。不得自动扩成16方案或旧40/192网格。

当前唯一配置模板configs/static_dual_overlap.json；权威说明docs/experiment.md、docs/implementation.md及用户完整研究稿。2026-11-14按用户要求移除旧实验入口/训练与动态门控/下载代码及旧配置，仅保留当前调用链和显式HF数值参考。不为历史兼容重新引入旧代码；后续OPD按独立任务实现。

2026-11-14新增用户手动启动入口scripts/run_static_panel.sh：开头定义物理GPU池、方案ID和超参数，默认--check仅CPU预览；只有用户显式--run才在新RUN_NAME下启动小样本GPU验收，三角色通过后生成带验收凭据的独立配置并执行所选方案。原模板hold保持不变，代理本轮不代用户运行--run。默认仍8方案/Slerp0.1/层26；用户显式修改参数或选择子集才改变运行配置，不自动网格搜索。已有run配置不可改；物理GPU池可在恢复时更换，仍只用指定池内的空闲卡，最多两张。

## Semantics

- overlap保留Rethinking Top16计数形式，异构词表用完整ByteLevel token字节事件相交；映射相同时直接比原生ID。明确称跨family适配。
- P_overlap是完整原生词表→256字节+EOS+其他控制类→sum(min)，不从Top16近似。两指标分别保存/统计/画图；一个gate_metric冻结mask，rank_metric显式选定，默认均overlap。
- 共同UTF-8安全span起点比较，向学生span广播；终止EOS目标/不可对齐位置记NaN、明确覆盖率。EOS作为候选概率不丢弃。
- 普通源rollout后批量教师评分，冻结strict-< mask，候选不重判idx。无在线教师/probe/KV双分支。
- 同一S0读取T/Ref/S三角色文本，原始正文激活逐题池化；4份bank共用过滤和尺度。global平均原始差分后单位化；逐题零方向零干预，不回退。
- 首预测是prompt末状态，预测t对应prompt_len−1+t。提取与注入同一block输出，norm之前；capture完整hidden+residual。
- 完整OPD、learning-only/二次生成、刷新和裸学生收益另行定义；固定前缀增量不等于学习收益。

## Runtime and efficiency

默认EasySteer固定前端baecd969/fork7837b07d、wheel0.29；与旧.venv-opd普通vLLM0.32隔离。使用scripts/bootstrap_easysteer.py准备，不改共享环境。
全量pending提交由引擎调度，不设置max_num_seqs或人工CUDA Graph尺寸。HF只作显式参考，不自动降级。
完整词表在已有logits上批量聚合，只传258类统计；映射按设备缓存。正文capture保留选定层/位置/BF16状态/offset/mask，不偷偷换池化定义。
采用上游steering指纹；完整prompt评分与正文capture遵守跳过prefix读取规则。固定mask评分当前关闭prefix caching；普通无SV生成可按支持开启。
模型内容、投影、文本/tokenizer、向量、概率文件和mask身份均须校验。权重首次流式hash，stat未变时复用，不逐token重hash；新checkpoint/源前缀不能套旧概率。
CPU通过不等于GPU验收。实际GPU运行优先一张空闲卡，最多两张；逐GPU计算进程准入，绝不停止其他项目。

## Workspace and checks

输出/缓存/临时安装目录放项目内，共享assets模型/数据只读。旧结果已按用户授权清理，不自动恢复或把旧失败/训练结果混成新配对结果。
保留用户核心研究稿、固定论文/源码与来源清单；不再通过堆叠历史前言记录当前状态。
CPU测试用source scripts/runtime_env.sh后CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q tests。包检查scripts/validate_package.py，不加载GPU模型，并检查本地模块/入口引用完整。

## Reproducibility, resume and logs

用户2026-11-14要求统一seed42（每个请求也42，不用42+i），在保留逻辑/吞吐前提提高可复现性。子进程启动前固定PYTHONHASHSEED；配置、支持、数值设置、代码/包/权重/引擎身份留档，不强制慢速确定性内核，不承诺跨GPU调度逐bit一致。
恢复粒度固定为engine_call：每角色整批参考生成、整批源生成、整个共同激活池及4份bank、每角色整批概率评分、每个完整候选（含baseline）。每次公开generate/capture调用连续运行，返回后统一处理与保存，StageCheckpoint用SHA清单和原子complete.json提交；没有标记的当前整批重算，已提交阶段/候选完整复用。不在引擎内逐请求落盘，不自定义enqueue/step调度，不保存KV/RNG。源/教师概率、BF16激活、bank/尺度/mask/指标和代码/权重内容清单保留供后续训练，新的前缀/checkpoint仍须重算对应概率与激活。
用原命令加--execute --resume，绝不绕过hold。完整缓存先CPU校验，无需GPU；仍存活的旧监督器接管等待，不重复启动。每阶段单写入锁，Ctrl+C/SIGTERM处理仅影响本项目自有子进程组。
实时转发原生tqdm，仅更新内存计数；15秒心跳写progress，阶段边界写事件。console.log与events.jsonl/progress.json记录北京时间起止、排队/活跃/墙钟时间、恢复数、前向完成数、已提交数与ETA；completed不等于committed。首条未完成或异质阶段总ETA未知时不伪造。scripts/show_static_progress.py --watch只读查看，不启动实验。
