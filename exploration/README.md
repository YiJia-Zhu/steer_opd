# exploration：固定数据的SV方向探索

本实验由用户2026-11-17授权，在物理GPU0／1进行；参数搜索继续使用2／3卡。共享src、scripts、configs和运行中源码保持原样。exploration/snapshot保存参数实验冻结版本287af959…的独立副本，仅在副本中绑定项目根路径与插件元数据位置；运行入口是exploration/launch.py，配置是exploration/config.json。

保持教师Qwen3.5-35B-A3B、学生Qwen3-4B、原1000题、700／300划分、原师生回答和固定学生前缀、Top16指标v4、同一冻结无SV基线及seed42。无模型参数更新、无新回答、无OPD；只探索方向构造和注入策略。所有SV设计使用原1000题池，包含Confirm的激活，和原global方案一致；Confirm仍是内部复核。评分使用相同EasySteer环境、TP2、BF16权重／FP32头、教师不启用batch-invariant；CPU指标使用独立8进程，保留全部字段并无损gzip。

先做原Select中的固定64题筛选，各候选评分完整原回答，不能把64题的结果当成700／300题结果。只有Select64参与候选选择：优先取比重放原方案至少多0.05pp、且P_overlap不降的前两项；若没有，取有正增益的最好一项；无正增益则停止扩展。入选项随后按原Select700和Confirm300完整验证。Confirm不参与新候选排名。

| 名称 | 实际干预／目的 |
|---|---|
| no_sv_replay | 无SV控制，检验冻结基线重放差异 |
| reference_replay | 层29／Slerp .1／all／global，复现同64题的旧结果 |
| negative_direction_control | 原global反向，排查符号和“任意旋转都涨”的解释 |
| official_answer_input_mask | 原方向和强度，只将注入从预测状态位置换到官方回答输入位置；单独标记mask控制 |
| nuisance_orthogonal | 从global差分去除两种回答共同激活均值的投影，随后单位化；其余仍层29／Slerp .1 |
| covariance_whitened | 用两域各自中心化后的题间pool协方差，加0.1×平均特征方差的ridge，对global作逆平方根调整并单位化；层29／Slerp .1 |
| knn_shrink_global | 按教师自身回答pool的中心化余弦找64个邻题，排除自己；0.5global＋0.5邻题raw差分均值，单位化；层29／Slerp .1 |
| multilayer_19_29 | 复用19和29层各自global；两层同时注入，每层alpha=.1/sqrt(2)。仅使alpha平方和与单层.1相同，未声称实际扰动范数或算力严格相同 |
| source_adaptive_protect | 原global、层29；按冻结源overlap逐共同span设置alpha：≤9/16为.3，10–12/16为.2，≥13/16为0；不可对齐目标为0。一次提交多个互不相交的位置选择，不在线重判，不按题顺序退火 |

这些是待验证的设计假设，不是保证提升的新结论。共同mask／数据／模型之外的变化均在condition中留档。高重合保护策略来自旧结果的位置诊断，是事后提出的探索；不得当成预注册独立测试结论。目标是整体题目平均至少+2个百分点，不能用低overlap子集的+2.7pp替代整体目标。

GPU前先核验原bank／capture／结果的提交SHA，以及题目、前缀和tokenizer支持。首两个GPU控制必须满足平均overlap重放差绝对值≤0.1pp、观察token的logp MAE≤.05，否则停止；这不是跨批逐位相同声明。所有进程只使用通过空闲准入的0／1卡，不停止其他任务。探索子进程的cwd为项目/exploration，独立租约为exploration/.runtime/gpu_leases.json，避免参数任务的项目两卡额度阻止本次用户已授权的并行。原参数准入器按项目根cwd识别进程，因此不会把探索0／1卡计入其两卡额度；物理GPU的占用检查仍保留。每阶段留存完整候选提交、原概率与全部指标；不保存逐token隐状态。保留60GiB磁盘给参数实验，另为探索单阶段留5GiB，空间不足停止扩展并保留已提交结果。

输出沿既有结构放入outputs/step_2_sv、step_3_overlap、step_6_summary/exploration/<同模型和数据路径>/st_direction_design_n1000_seed42_v3。各阶段日志为step3对应pilot／select／confirm/stage.console.log，总进度为step6/progress.json，comparison.csv明确pilot_only。源码身份与输入SHA留档，修改探索代码后必须新建身份；共享参数进程的源码hash前后核对。

```bash
source scripts/runtime_env.sh
"$STEER_OPD_PYTHON" exploration/launch.py --run
"$STEER_OPD_PYTHON" exploration/launch.py --status
```

首次v1仅排队未调用GPU；v2已提交前7项后因缺少多向量声明停在新保护策略。v3只在探索快照声明steer_multi_vector=True，重新执行无SV／旧方案控制；通过后以原提交SHA引用v2已有候选，继续余下保护策略及多层联合。旧提交和身份不改。
