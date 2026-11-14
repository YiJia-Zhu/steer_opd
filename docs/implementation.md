# 实现与运行说明

当前代码接通了双指标、EasySteer提取/概率评分/固定mask路径。CPU契约验收与真实GPU验收分开记录；本轮未运行新模型实验，未测提速倍数。

## 双指标与数据

[compass_alignment.py](../src/steer_opd/compass_alignment.py)定义overlap（完整token字节事件Top16交集/16）。普通token恢复精确ByteLevel BPE字节；同拼写控制token可相交，EOS控制token不随意合并，映射一致时直接比较原生ID。另记Top16跨词表覆盖率、相交概率质量和全词表共同事件质量。它保留Rethinking计数形式，是跨family适配，分词/词表差异仍会影响读数。

[byte_probabilities.py](../src/steer_opd/byte_probabilities.py)定义P_overlap：对各自完整原生输出支持做FP32归一化，将每个普通token概率加到首字节类别；EOS=256、其他控制/空输出=257，再计算双方258类概率的sum(min)。排除未定义/补齐head行，保留全部有效概率质量，不用Top16近似，不重新分配被遗漏尾部质量。

当前S原生支持151669，T/Ref各248077；均为连续已定义ID，不能沿用旧151665子集。ByteLevel以原始字节处理空格和半个UTF-8字符；其他decoder会拒绝。源EOS目标没有文本span而不评分，但EOS作为下一候选的概率参与有效边界的P_overlap。

两指标都在合法共同span起点评分，再向span内学生tokens广播。保留alignment_id、alignment_mask、原生token范围与共同字节边界；明确广播位置和未评分原因。Compass span logp按原生token平均，不称所有字符串分词路径的精确概率，也不把两个overlap指标当成OPD损失。

源rollout、T/Ref和候选评分均留Top16、sampled_logps、byte_probs[N,258]、common_vocab_mass、entropy及projection_id。投影身份绑定原生tokenizer映射、有效支持、类别语义和另一词表的共同事件；旧Top16缓存不能替代新双指标缓存。

## EasySteer路径

隔离环境已安装并通过CPU导入、Slerp注册/范数及API检查，未启动GPU；验证记录见[运行时检查](../audit/dual_overlap_implementation_20261114/runtime_cpu_check.json)。固定前端baecd9698f32aadef68b41536f3db477b513a0d7、fork 7837b07d21c0ddffd043a30001f2c5654290aec7，基座wheel为vLLM0.29.0。普通vLLM0.32环境不能冒充此fork。API源码与hash见[固定源码清单](../ref/easysteer_runtime/manifest.json)。

[bootstrap_easysteer.py](../scripts/bootstrap_easysteer.py)按官方wheel+Python overlay方式安装项目隔离的.venv-easysteer和Slerp插件，不覆盖.venv-opd或共享环境，不加载模型：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/bootstrap_easysteer.py --install
```

| 路径 | 实现 |
|---|---|
| 原生参考/源生成 | [generate_static_sources.py](../scripts/generate_static_sources.py)：全量请求一次公开LLM.generate，返回后保存整批。源采样TP1，教师文本最多TP2 |
| S0激活提取 | [easysteer_extraction.py](../src/steer_opd/easysteer_extraction.py)：一次公开easysteer.capture.capture读取完整T/Ref/S池，指定层26及prompt末/正文行；返回后核验token/位置、池化并保存BF16状态、offset/正文mask。原始capture暂留CPU内存，避免重复临时文件 |
| T/Ref概率 | [score_compass_sources.py](../scripts/score_compass_sources.py)：自身tokenizer、完整观察文本prompt_logprobs，取得真实观察token分数，T允许TP2 |
| 完整词表统计 | [engine_projection.py](../src/steer_opd/engine_projection.py)：V2 runner已有logits上scatter-add，不新增模型前向；整块logits批量聚合，只传258类及标量回CPU，不传完整词表 |
| 八组固定mask | [easysteer_static.py](../src/steer_opd/easysteer_static.py)：逐请求SteeringSpec、Slerp插件与已知prompt_positions，无在线教师/probe |

正式实验的模型前向均经EasySteer/vLLM。Tokenizer/config读取、字节与span对齐、均值池化、bank构造、指标、文件和绘图由CPU/Python/PyTorch完成；完整词表统计在vLLM worker已有logits上使用PyTorch。HF仅保留启动验收与显式hf_batched_static数值参考。旧训练/生成/动态门控入口已删除；后端路由核对见[审计清单](../audit/coarse_resume_20261114/backend_audit.json)。

所有概率均为raw logprobs，BF16模型的lm_head显式输出FP32，与HF参考一致；Qwen3.5的language_model嵌套输出头也按原生支持裁剪，归一化前排除补齐head行。投影索引按设备缓存，避免每decode步重复拷贝整个词表映射；一次引擎logit块合并统计，避免逐请求重复设备同步。source采样拒绝speculative/不支持的分片采样路径；TP评分各worker统计必须相符，缺位置/身份不明确/重算分布改变会报错。

source预测t使用状态prompt_len−1+t。固定前缀评分将完整观察回答放入prompt，额外生成一个token只用于完成前向并丢弃，预留+1上下文容量。按真实绝对位置取统计，不能把prompt概率错配到回答token状态。

capture返回完整post-block hidden+residual。原始正文状态池化排除prompt末预测状态；前者用于方向，后者留作预测位置记录。只选层26减少存储/传输，不自动截断后9层。HF参考路径已截断到前27层，因此未实测前不承诺EasySteer提取必然更快。CPU捕获预算与请求并发无关，没有人为序列上限。

## 缓存与指纹

复用上游EasySteer指纹：payload内容SHA、scale、layer、selector、algorithm、normalize参与身份，显示名称不参与。固定前缀评分关闭prefix caching；普通无SV生成按引擎/模型支持允许缓存，完整prompt_logprobs和正文capture会按上游规则跳过读取，避免缺失逐位置概率/激活。不同受干预KV不能仅因文本相同就混用。

[checkpoint_identity.py](../src/steer_opd/checkpoint_identity.py)补充模型内容版本：首次按权重shard和config/tokenizer逐文件流式SHA256，缓存于.runtime，之后按inode/大小/mtime/ctime检查并复用；最近两秒写入的文件仍重hash，避免快速同大小覆盖撞时间戳。身份纳入参考/源job、激活bank和原生评分绑定；同路径替换权重会失效，不逐token重hash。模型资产仅只读。

静态mask同时保存两种源指标、单一gate_metric、源token/hash、阈值、合法位置与mask_id。候选评分只读这些mask；单独保存source-rollout与无SV重放的两种差值。投影身份、概率文件SHA、checkpoint和bank哈希共同校验续跑。已提交评分直接复用，补图不加载模型或重算教师。

## 输出与边界

每候选comparison/overlap与comparison/P_overlap分别生成前3条的PNG/SVG/CSV及first3.pdf，清单保留源顺序/不可用原始条目。蓝线为完整同源无SV重放，橙线为固定mask前向，红点为实际注入，下方差值单位pp；缺失位置保持NaN，不能伪造为0。

summaries.json同时有metrics.overlap和metrics.P_overlap，包括题目等权配对差值/区间和token加权值。阈值在select校准，排名仅看rank_metric；confirm不反向选参。源/候选P概率及两种delta均落盘。

EasySteer逐token实际扰动范数尚未接入（relative_perturbation_available=false）；HF参考能记录。完整OPD梯度路径、GPU一致性/缓存隔离和性能仍待验收，不能称完整研究面板已复现。当前execution_hold与GPU准入在独立入口均生效，没有自动HF降级。

CPU验证命令：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q tests
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/validate_package.py
```

测试包含完整分布质量、EOS/控制/补齐行、跨分词两指标差异、交错请求与分块prefill、t−1映射、缓存篡改、正文池化、双图/汇总/续跑、8方案和执行锁；这些是CPU验证。

## 可复现性、恢复与进度（2026-11-14）

[experiment_runtime.py](../src/steer_opd/experiment_runtime.py)统一Python、NumPy、PyTorch和vLLM种子为42；**每个采样请求也固定42，已经移除42+i**。子进程启动前固定PYTHONHASHSEED=42、PYTHONUNBUFFERED=1，并使用各自解释器的CUDA工具链。数据划分、问题顺序、采样参数、原生支持、精度、模型/代码/包版本和实际引擎配置留档；未提交调用恢复时仍提交完整原请求集合。

保留vLLM自动调度、CUDA Graph及现有attention实现，不强制慢速确定性内核。GPU批形状、调度、硬件/驱动和归约可能产生数值差异，因此固定seed并不保证跨调度逐bit相同。已提交输出不重新采样；新的输出保留实际GPU与引擎配置供核对。

| 恢复单元 | 持久化与恢复方式 |
|---|---|
| 参考/源生成 | 每角色整批参考生成、整批源生成各一个恢复点。公开generate返回后保存全部回答；源token/Top16/byte_probs与回答hash绑定 |
| 激活提取 | 完整共同T/Ref/S池一次公开capture；返回后保存BF16状态、offset/正文mask/预测位置、池化均值及4份bank，整体提交 |
| T/Ref评分 | 每角色全部源前缀一次generate，返回后保存真实native logp、Top16、258类概率与投影身份，整批提交 |
| 候选评分 | baseline及每个候选各一个恢复点，每次generate覆盖全部原前缀；返回后计算双指标/delta并整体提交。画图失败仅补CPU图 |
| 全流程 | --execute --resume重新进入原配置；缓存完整的阶段可在CPU校验后跳过GPU准入；发现旧监督器仍存活则接管等待，避免重复启动 |

StageCheckpoint先保存完整阶段的文件，最后原子写入complete.json，清单绑定输入/运行时/候选名称及文件SHA256。恢复时校验已提交文件；未有完成标记的当前阶段文件不复用，整批重跑并覆盖。已提交阶段禁止覆盖，损坏或身份改变会报错。逐请求RowCheckpoint及.objects硬链接机制已移除；生产阶段仍有单写入者文件锁。

配置resume_granularity=engine_call。一次公开引擎调用连续运行，不插入逐请求落盘、检查点或自定义enqueue/step循环；指标和文件处理均在调用返回后执行。vLLM/EasySteer内部仍按自身机制调度、传输capture数据。中断可能损失当前整批或当前候选的计算，已完成阶段与候选复用。当前不保存vLLM在途KV/RNG；真实GPU恢复数值与耗时仍须验收。

provenance保存代码快照、软件/数值设置和权重内容清单；原始参考文本、BF16激活/池化均值、4份bank、共同尺度、实际源tokens与行为概率、教师span分数、静态mask、两指标与图均保留。最终artifact_index.json统一列出来源、校验和及后续训练的适用范围，复用既有大文件哈希，避免重复读取激活。当前没有optimizer/RNG训练断点，因为参数更新数为0；未来新增轨迹/学生checkpoint必须取得自己的行为概率、教师评分与激活，不能把固定源前缀候选分数冒充新on-policy数据。

日志同时实时显示并写入run/console.log；GPU子阶段另有attempts/阶段时间戳/stdout.log及process.json，记录排队、启动/结束、GPU/驱动、命令、PID、退出码与耗时。阶段输出目录有events.jsonl、最新progress.json及每次attempt快照：北京时间含UTC+08:00、首次/本次开始、已耗时、累计活跃时间、含中断等待的墙钟时间、恢复数、完成数和ETA；每15秒心跳，模型加载、评分、绘图等有分阶段时间。

终端原生tqdm和日志实时转发，不等子进程退出后才显示；进度回调只更新内存计数，15秒心跳写progress，不逐请求提交文件或事件。completed为本次前向完成数，committed为可恢复的已提交数；两者分开显示。ETA按本次完成请求估计；第一条完成前、GPU排队、异质阶段的全流程ETA均显示未知。Ctrl+C/SIGTERM记录可恢复中断；SIGKILL无法写结束事件，恢复时保留最后心跳并记录接续。CPU故障测试覆盖这两种真实进程中断。

在项目根目录只读查看进度（不会启动模型）：

```bash
python scripts/show_static_progress.py --watch
```

推荐使用[统一启动脚本](../scripts/run_static_panel.sh)，参数及方案编号见[实验协议](experiment.md#一条命令启动)：

```bash
bash scripts/run_static_panel.sh --check
bash scripts/run_static_panel.sh --run
```

--run由用户明确执行，启动时自动检查隔离环境及三角色小样本GPU一致性；验收前不启动正式面板。小样本HF数值参考是独立验收阶段，正式面板仍全部走EasySteer。验收固定阈值：观察token的logp平均绝对差≤0.05、最大≤0.2，完整下一字节概率的最大总变差≤0.03，学生捕获状态的最大相对L2≤0.03；零强度/移除恢复另用1e-5容差。这些是BF16工程验收阈值，不代表学习收益或长上下文性能，失败不自动放宽。

验收结果绑定代码、三角色权重、层、强度和教师TP；已完成验收阶段按完整调用复用，配置变化或文件损坏拒绝混接。通过后生成新run的独立配置，凭据传入每个子阶段；原模板的hold保留。再次--run即按粗粒度恢复。GPU池按物理编号限制，最多两张，不因指定卡繁忙跳到其他卡。shell文本及实际GPU选择每次单独留档。

本轮实现与CPU测试没有启动GPU或模型实验。教师权重由用户手动下载，代理不运行下载逻辑。
