# Compass式跨tokenizer接口与8方案执行流程

2026-11-13用户要求参考CompassOPD代码仓库补齐接口与流程。
截至本轮，论文v1及GitHub检索没有定位到官方仓库，已向用户请求URL；不能声称做过官方源码移植或逐项复现。
当前依据[CompassOPD附录A.4](https://arxiv.org/html/2609.10154v1#A1.SS4)独立实现，来源范围见[检索记录](../audit/compass_span_adapter_20261113/reference_status.json)。

## 1. 本轮模型和范围

T=Qwen3.5-35B-A3B，Ref=Qwen3.5-0.8B，S=Qwen3-4B。
固定Slerp alpha0.1、75%单层，即学生36层中的0-based26；T−S/T−Ref × all/static_overlap × global/per_question完整8组，另加共同无SV基线。
仍是固定源回答前缀、0参数更新筛选；本轮不运行OPD、SFT或旧失败任务，也不自动扩展40/192配置。

已读取35B教师官方tokenizer/config元数据，固定revision为`59d61f3ce65a6d9863b86d2e96597125219dc754`，没有下载权重。
实际词元映射：S有151669个定义ID，T/Ref各248077个，T与Ref的完整映射hash相同。
新协议各自使用原生已定义输出支持，排除head填充行；不复用旧151665受限子集或跨模型整数ID比较。
所有角色在新配置中显式`enable_thinking=false`，模板、prompt、生成预算和概率支持独立留档。

## 2. Compass式span似然评分

[compass_alignment.py](../src/steer_opd/compass_alignment.py)恢复ByteLevel token的精确字节，保留实际采样的学生token序列，
由教师和Ref各自重新编码同一回答文本，按S/T/Ref共同且UTF-8安全的字节边界构造最小共同span。
不会把单独decode出的Unicode替代字符当成真实字节，也不通过未验证的Unicode归一化制造匹配。

每个模型对同一实际span使用自身token log-prob的平均值；一对一span退化为单token原始log-prob。
T与Ref的平均分之差广播到对应学生token，另保留学生原生逐token sampled_logps。
学生锚点位移若以后用于学习，应读取原生逐token学生概率，不使用分析中的span平均值替代。
平均log-prob不是完整字符串事件概率；本轮也没有将它当作跨词表完整KL。

控制token、末尾EOS及无法一致对齐的位置有显式alignment mask，不填造似然。
末尾EOS不属于可见文本span，因此不参与本指标排名；全位置注入组仍可在其预测状态注入并留档。
无有效span的源回答记录剔除原因；所有组使用同一有效位置，报告题数/token数与覆盖率。

[compass_scoring.py](../src/steer_opd/compass_scoring.py)同时支持Qwen3和Qwen3.5 dense/MoE的text decoder，
多模态checkpoint只读取文本路径，T/Ref分别批量评分。35B阶段最多2张准入GPU，其他阶段优先1张。
缓存绑定原生prompt/回答、checkpoint、词表、输出支持与alignment ID，候选评分不再调用教师。

## 3. 静态overlap是本项目适配

Compass A.4只定义实际span的似然对齐，没有定义跨tokenizer的原生Top16分布。
本项目使用显式命名指标`compass_aligned_byte_top16_overlap`，不将其标作作者指标或原生token-ID Overlap@16：

1. 在共同span起点，取S和T各自的原生Top16。
2. 普通token映射为精确的下一token续写字节，EOS候选映射为共同终止事件，其他控制/未定义候选不伪造匹配。
3. 比较两侧可比续写事件集合，交集大小除以16；记录两侧有效事件覆盖率和共享原生概率质量，不做Top16内部概率重归一化。
4. 将span起点值广播到该span的学生位置；随后冻结`alignment_valid AND O_src<tau`作为静态mask。
5. 所有候选重用源mask，候选新overlap仅为结果，不改变idx。

该指标依赖tokenizer分段与span粒度，不是完整教师分布的概率输运，也不是每个学生token处的原生即时Overlap@16。
它与旧原生指标分表报告。若用户提供仓库中不同的共同动作定义，需按真实源码复核并另记协议版本。

## 4. 完整数据与执行入口

配置：[configs/compass_static_eight_20261113.json](../configs/compass_static_eight_20261113.json)。
入口：[run_compass_static_panel.py](../scripts/run_compass_static_panel.py)。

流程为T/Ref/S参考生成→共同S0提取→4份bank→普通学生源rollout保存Top16→T/Ref原生文本概率评分→
训练侧阈值校准→完整8组select→冻结前2名后confirm→停止供检查。

- 原100题ID/内容沿用；用新学生重新构造prompt，不沿用旧Base prompt或旧55题有效数。
- 预先固定训练题70% select、30% confirm，seed42；质量/激活/对齐过滤后报告实际数量，每侧少于8题即拒绝，不事后重洗划分。
- confirm是带本题参考向量的预留训练题检查，不宣称独立dev泛化；AIME完全不参与。
- 参考T/Ref/S生成temperature0.7、top_p0.95、max12288；另作一次普通S源生成，temperature1、top_p1、max7168，每题1条。
- [prepare_compass_banks.py](../scripts/prepare_compass_banks.py)由同一S0读取三角色文本，缓存token激活，使用共同过滤/尺度构造T−S/T−Ref的全局与逐题4份bank。
- 全局先平均原始差分；零方向精确不干预，不回退其他题或全局。Slerp方向保持单位长度，旧加法尺度不乘到alpha上。
- 只用select有效位置校准20%目标触发率，记录实际可达率和精确阈值，不复用旧0.375或在候选间调阈值。
- 主排名为题目等权配对增量，另报有效token加权、置信区间、覆盖率、扰动和实际mask。
- 图只请求源清单前3条；无法评分的原样本注明不可用，不换后面的样本冒充前3条。不可评分位置保留NaN/有效性标记。
- 所有阶段使用独立目录和可恢复缓存；改变来源/hash则拒绝复用，不恢复旧失败训练。

默认CPU准备及检查：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_compass_static_panel.py --config configs/compass_static_eight_20261113.json
```

权重和GPU路径验收后，加`--execute`；各GPU子阶段内部都经过空闲GPU准入，最多2张，不终止其他项目。
默认显式使用已存在的HF批量参考后端。EasySteer需选择`screen_backend=easysteer`及对应`screen_python`，没有静默后端降级；新版fork仍未在本地部署/验收。
本轮没有实际GPU评分，提供的代码不能当作已完成35B模型数值或吞吐验收。

## 5. 当前验证与未完成事项

已完成CPU原始词表核对、100题准备、原生prompt/4-bank/8-group job组装，以及实际Qwen3/Qwen3.5 tokenizer缓存链路测试。
Qwen3.5 dense/MoE文本架构用缩小配置和Transformers自身Torch参考算子验证，未使用GPU FLA内核。
CPU完整回归结果及hash见[实现审计](../audit/compass_span_adapter_20261113/README.md)。

当前预检为`missing_assets`：已查资产路径缺Qwen3.5-35B-A3B完整权重；下载的是tokenizer/config元数据。
新8组生成/评分尚未开始。EasySteer GPU部署、一致性和速度验收仍未完成；官方Compass仓库的逐项源码对照等待URL。
