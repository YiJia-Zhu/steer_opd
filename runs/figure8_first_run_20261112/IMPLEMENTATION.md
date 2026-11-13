# 首轮运行实现与恢复

2026-11-12默认调度更新：自有生成入口不再传max_num_seqs或手写CUDA Graph尺寸，vLLM实际解析1024/图最大512。
科学配置、采样预算及已有checkpoint保持不变；旧config hash通过仅剔除已废弃运行字段的兼容检查继续使用。
部分生成job保留历史hash与已完成答案字节，新进程只生成剩余请求。证据见
`audit/remove_sequence_caps_2026-11-12/README.md`；运行时详情以每个生成目录runtime.json为准。

2026-11-12性能复核后的词表说明：Base tokenizer.json定义151665项，但官方tokenizer_config另补充
四个控制token，运行时AutoTokenizer两端均151669。下述151665仍为本轮已冻结的受限输出子集，
并非运行时全部共同已定义词元；旧文“教师另有四个”不适用于运行时合并结果。
不改实际生成/评分分母或断点hash。证据见`audit/performance_ranking_2026-11-12/tokenizer_check/`。

2026-11-12用户已授权直接实现/运行。当前只执行普通OPD和T−S全位置steer两组，
各50步；SFT、T−R/动态门控与跨家族扩展不在此次运行队列。
状态以同目录`campaign.json`、`*/steps/*/train_metrics.json`、`*/latest.json`为准。

## 固定实验

- 教师Qwen3-4B Non-thinking，学生Qwen3-1.7B-Base；共享输入只读。
- 原OT3 OPD parquet去重后冻结100题OPD、256题方向、128题dev；题目及编码hash保存于
  `data/figure8_first_run_20261112/manifest.json`，无本地AIME/MATH精确规范化文本重合。
  这是精确文本检查，不宣称完整语义去污染。原30K行中检查出17,396条重复。
- batch64、n4、每批1次更新、LR1e-6、AdamW/betas(0.9,0.999)/WD0.01、constant/零warmup、
  grad clip1、token-mean、prompt≤1024、response≤7168、temperature1/top_p1。
- 冻结学生Top16，`student_p`在Top16内归一化；奖励用教师/学生完整支持上的log-prob差；
  使用原fork三维PPO surrogate。CPU测试直接加载固定原fork函数对比值/梯度。
  单GPU实现采用跨全部有效token的统一归约，原fork动态micro-batch缩放不是逐位复制。
- T−S方向由同一初始Base读取两种回答，layer13、rho0.03，全文正文每题均值/题间等权。
  生成与学习全部回答预测位置注入，方向及尺度冻结；评测完全关闭。
- AIME2025 avg@16，temperature0.7/top_p0.95、max31744，0/10/20/30/40/50步。
  固定32题dev初始Base轨迹用于无hook Overlap@16；教师固定评分可复用。
- 明确使用`<|endoftext|>`和`<|im_end|>`停止；真实prompt token已冻结，生成不继承
  Base的greedy默认或教师的top_k20默认。未以精确原文数值复现作为成功门槛。

## 共同词表修正

首次未限制输出head的256条rollout中，2条分别采到151866、151927，它们没有Base
tokenizer定义。88条已评分回答上的原始无定义行平均质量约4.52e-6、最大约0.00288。
这批没有用于参数更新，完整留在`pre_common_support/`，见`vocabulary_support_audit.json`。

正式配置统一使用151665个共同有定义且ID一致的词元（0…151664），生成、学生评分、
教师评分、训练head和评测均在该支持上归一化。输入embedding仍保留原尺寸，以便教师
原生非thinking prompt可使用自己的控制token。教师另有四个已定义控制输出及padding行
不进入公共输出分布。原始排除概率质量在评分缓存中另外记录，区分excluded和undefined。
这项必要的接口修正对两个组相同，已明确列为相对原始全head配方的变化。

## 速度与数值验收

vLLM 0.32负责连续批处理生成，每个请求完成立即原子落盘；采用eager prefill和
decode-only CUDA Graph。受控hook在图捕获前安装，生成位置由原始prompt长度决定，
包含首回答预测位点，处理prefill/重算；decode图的有效行全部注入。
vLLM的延迟residual加法在被注入位置先合并，再加delta，与HF block输出接口一致。

训练侧使用HF/PyTorch、BF16前向，FP32 master权重/梯度累积/Adam及输出head。
当前torch的`mm(out_dtype=float32)`缺少自动反向，使用经float参考梯度核验的显式linear
反向。无梯度评分与训练重放使用相同head定义。自定义HF舍入norm仅作过诊断，正式运行
仍使用vLLM原生norm/激活，配置中的`normalization=native`明确记录。

验收文件在`acceptance/`：alpha0与移除hook恢复；HF缓存/整段重放；vLLM/HF逐位置
Top20/采样log-prob及部分位置完整词表KL；真实普通/非零方向forward/backward参数更新。
完整分布KL阈值与HF自身cache/replay一致：每条均值<0.005、所测位置最大<0.03；
采样log-prob均值误差<0.06。低概率尾部的未加权最大log-prob差保留诊断，不冒充完整分布误差。
正式rollout另逐条检查采样ID合法及HF重放差异；训练检查更新前评分/重放误差和梯度有限。
这些是实现检查，不是accuracy提升证据。

本机nvcc13.2与CUDA头13.0混用导致可选FlashInfer sampler JIT失败，因此显式关闭
该采样器，使用vLLM原生采样和已安装FlashAttention；没有修改共享环境或下载模型。

## 一张卡与可恢复性

每阶段经`run_on_idle_gpus.py`进行两次空闲GPU检查：无compute PID、显存≤128MiB、
利用率≤1%。当前使用物理GPU2；每个子进程只见一张卡。其它卡上的原有进程保持不动。
所有阶段串行复用GPU，不执行原shell的全局ray stop，不向外部进程发信号。

回答和teacher/student评分按条保存；每次完整更新保存BF16模型、FP32 master/Adam、
Python/NumPy/CPU/CUDA RNG。先写pending目录，全部成功后原子提交，再更新latest指针。
checkpoint提交后才清理本运行可重建评分缓存和旧optimizer；保留评测模型与rollout。
未提交更新被kill后重算该步，已完成回答/评分不重做。文件锁阻止重复supervisor。

在项目根恢复：

```bash
source scripts/runtime_env.sh
"$STEER_OPD_PYTHON" scripts/run_figure8_campaign.py --config configs/figure8_first_run.json
```

查看实时进度：

```bash
.venv-opd/bin/python scripts/figure8_status.py
```

配置hash不匹配时拒绝接着训练，避免无意混合实验。运行时版本在GPU阶段检查。
常见子进程SIGKILL/SIGTERM会有限重试；整个supervisor被kill时用同一命令恢复。
恢复的未完成新采样仍受GPU数值/批处理影响，不声称跨任意硬件逐bit复现生成。
