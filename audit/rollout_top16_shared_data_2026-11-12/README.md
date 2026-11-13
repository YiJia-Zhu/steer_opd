# Rollout Top16与同题数据变更（2026-11-12）

用户要求先停止实验，再保存rollout完整Top16交教师评分，并将SV提取与OPD改为同一题集。
本次没有启动GPU任务。旧campaign暂停，两组各10/50步保留；停止过程见
[停止证据](../user_stop_top16_20261112_185502/)。旧断点/方向/答案不改名为新设置。

## 实施范围

- 新配置：`configs/figure8_shared_top16_v2.json`，数据与run为`figure8_shared_top16_v2_20261112`。
- 原100道OT3题（ID/顺序/题文/编码完全相同）同时作为SV与OPD题池；dev128、AIME30不变。
- S0分别读取T和S0自己的校准回答，仍提取全局固定T−S，layer13/rho0.03；生成与学习注入，评测关闭。
  文本生成/实际有效配对数尚未产生；空白/截断/重复等过滤规则不变，100是候选题数。
- 两条新训练曲线均从原Base开始；不在旧steer第10步换方向，不混入旧vanilla的HF候选路径。
- T−R、在线逐题重提与动态门控未实施，SFT/跨家族任务继续暂停。

## 逐token候选与训练

vLLM整批enqueue，保留默认调度，不施加num_seq上限；训练rollout用logprobs=16、raw_logprobs。
每个有效回答预测位置保存16个ID（int32）、原生log-prob（FP32）与采样token log-prob。
它们以151665冻结输出子集为分母，候选本身不再次归一化。vLLM可能返回第17个Top16外采样token，
按实际Top-K rank和是否为额外采样项处理，边界并列也不挤占真实候选。

`rollouts/topk/<key>.pt`先原子写入，随后提交回答JSON；row/tensor hash相互绑定。
恢复时拒绝缺失、损坏、越界、非有限概率及问题/seed/轨迹改变；中断前未提交的回答仍为pending。
新路径直接由教师读取学生候选，跳过单独HF学生无梯度评分。带梯度学习前向仍保留，
按保存的候选/概率构造原student_p冻结奖励和三维PPO surrogate；hook贯穿checkpoint backward。
学生rollout、教师评分、学生学习各一条模型前向路径，学习的梯度重算另计；不声称精确两次底层kernel执行。

教师在同一次logits上取得自身Top16及学生候选分数。每步`rollout_overlap.json`记录token加权交集、
学生/教师共享质量、token数与前缀/hook来源。训练后可重建的教师评分缓存按原规则清理，
学生rollout Top16与聚合overlap保留；候选概率不冒充完整分布，未伪造学生原始head排除质量。

训练轨迹overlap来自各组自己的生成，steer组有hook，不能替代无hook固定dev前缀主指标。
无hook固定前缀的Overlap@16和AIME accuracy仍决定是否符合预期；即时overlap不当作收益证据或门控实现。

## 概率与断点校验

旧HF无梯度评分路径保留原最大log-prob误差0.2检查。新跨后端路径在同一次学习前向验证
Top16条件分布及“16个候选+其余事件”分布，两方向KL均要求每回答均值<0.005、最大值<0.03。
这些是可观察缩减分布，不是完整支持KL。候选逐项log-prob最大误差继续记录为诊断，
没有把跨后端的低概率候选误差误当成同一后端重复前向误差。

新campaign仍先执行真实HF/vLLM完整151665支持分布验收，并验证保存Top16和实际rollout logits的
一致性（每个预测位置都要有capture，误差<1e-4），再做真实反向更新验收。上述GPU检查尚未运行，
长序列/整批吞吐也未测，不承诺提速倍数。旧GPU通过记录不能替代新接口验收。

paused_by_user状态和execution_hold配置均阻止默认启动；显式恢复参数只供用户以后恢复实验时使用。
CPU子进程实测默认启动被拒绝，状态文件字节未变，没有产生GPU attempt。

## 原Base结果复用

CPU复核模型/支持集/采样/长度/seed/诊断协议及dev/AIME题目编码均一致后，复制原Base评测、固定dev前缀
与教师固定评分，1031个文件逐个SHA256一致。不复用任何已训练权重；两组第0步共用同一份Base结果。
原答案/job provenance完整保留，来源和hash清单见新run的`initial_evaluation_reuse.json`；
执行脚本见[reuse_initial_evaluation.py](reuse_initial_evaluation.py)。本次没有新增GPU评测。

## CPU验证

48项通过，覆盖真实原fork surrogate梯度、学习forward使用保存行为奖励、候选额外采样项/并列rank、
缺失/非有限/越界/损坏、写入中断、教师单次评分及恢复聚合、Top16/其余事件概率偏差、
无额外学生评分的campaign调度、暂停保护、同题复用与dev/test隔离、历史恢复兼容。
pytest退出0，有两项第三方SWIG弃用warning；进程退出时multiprocess资源追踪析构打印已知环境异常，
没有影响测试结果（XML的failures/errors均0）。

命令（CUDA_VISIBLE_DEVICES为空）：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q \
  tests/test_rollout_topk.py tests/test_generation_runtime.py \
  tests/test_replay_validation.py tests/test_resumable_opd.py \
  tests/test_campaign_reporting.py tests/test_training_semantics.py \
  --junitxml=audit/rollout_top16_shared_data_2026-11-12/cpu_tests.xml
```

[validation.json](validation.json)记录CPU结果、数据检查、暂停guard及代码hash；
[before/](before/)保存本次修改前代码与文档快照。当前只完成代码/数据/CPU准备，不构成训练收益。
