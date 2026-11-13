# 静态mask、Slerp和EasySteer代码实现

2026-11-13用户要求修改代码，现已接入静态mask与Slerp，并增加EasySteer适配。
本轮配置仍是Slerp0.1、75%层26、8方案及共同无SV基线，全部零更新。

## 已实现

- [static_masks.py](../src/steer_opd/static_masks.py)：源回答/checkpoint/词表/概率hash绑定，FP64严格阈值比较，冻结mask及首预测位置映射。
- [operators.py](../src/steer_opd/operators.py)：保持范数的Slerp，零方向/零强度精确不干预；反向共线采用确定性切线。
- [static_screen.py](../src/steer_opd/static_screen.py)：批量固定前缀因果前向、逐题向量、padding隔离、实际mask和逐位置扰动；没有在线教师、probe或KV双分支。
- [score_static_teacher.py](../scripts/score_static_teacher.py)：完成的无SV源回答上一次性批量教师概率评分；当前原生适配要求相同tokenizer映射。
- [run_static_overlap_screen.py](../scripts/run_static_overlap_screen.py)：CPU准备、显式HF/EasySteer后端、共同基线、配对统计、原子恢复及默认前3条PNG/SVG/CSV/PDF。
- [easysteer_static.py](../src/steer_opd/easysteer_static.py)、[easysteer_plugin.py](../src/steer_opd/easysteer_plugin.py)：每请求固定selector及自定义`steer_opd_slerp`算法。

新`static_overlap`标签现在已有实现。旧`overlap`保留动态语义，旧入口会拒绝新静态/Slerp配置，避免误执行。
Slerp从bank取得单位方向，alpha控制转角，不使用旧加法的激活尺度或rho。默认加法序列化与旧配置保留兼容。

## EasySteer状态与位置语义

固定核对的fork是`ZJU-REAL/EasySteer-vllm-v1@7837b07d21c0ddffd043a30001f2c5654290aec7`。
上游支持已知selector、每请求spec和raw logprobs，但没有内置Slerp，因此增加项目插件。
插件仅支持split模式，不宣称完成full-graph Slerp kernel。项目插件入口元数据已装在`.runtime/easysteer-plugin`，不改共享环境。
父进程及worker通过通用插件注册同一算法，使用V2 runner；GPU准入保留，但不套用旧OPD vLLM0.32的版本锁。

首回答预测t=0映射最后prompt状态，二次生成的t>0映射decode位置t−1。
固定前缀评分将`prompt+source_response[:-1]`作为完整输入，selector使用绝对prompt位置；
prompt_logprobs取得前N−1个预测，单个输出分布取得最后预测，采样出的token丢弃，不生成另一份回答。
空mask/零方向/零强度使用`steering=False`，显式禁用继承的默认steering。

当前`.venv-opd`仍是普通vLLM0.32，没有这个新版fork；旧conda EasySteer也不等于新版API。
已完成固定上游API/selector的CPU契约验收，尚未部署或验收本地GPU引擎，没有提速倍数。

## 输入与使用

新评分job须显式给出`stage=static_overlap_screen`、`optimizer_updates=0`、学生/checkpoint和指标身份、
输出支持、源行清单、教师缓存、冻结阈值、4份S0 bank及8个condition。
`source_dir`可读取项目已提交的无SV rollout Top16，校验row/tensor hash；教师文件须绑定相同源token、支持及checkpoint。
改变源回答/概率/bank/配置会拒绝复用旧manifest。完整阶段缓存可CPU补图，不加载模型。

默认只CPU准备：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_static_overlap_screen.py --job runs/<new-run>/job.json
```

模型路径与输入验收后，通过`scripts/run_on_idle_gpus.py`准入再加`--execute`评分。
`backend=hf_batched_static`使用现有环境；`backend=easysteer`需隔离环境安装支持的fork。
[8项选择清单](../experiments/next_pair_slerp_scheme_screen_20261113.json)仍是规格，尚缺新配对数据/bank/已验收指标，不能直接当作job。

## 新配对与验证边界

Qwen3.5-35B-A3B/0.8B→Qwen3-4B的共同概率事件计算器仍未实现。原生适配明确拒绝不同tokenizer ID映射，
不能通过改写身份或沿用旧151665支持规避。冻结外部事件指标的mask能力不等于已具备该指标的概率计算器。
已检查的共享目录及项目链接仍未发现35B教师；新8组GPU评分未启动，旧run和配置保留。

静态调度省去旧逐token教师/probe/分支重算的在线依赖；倍数需匹配模型、长度和概率协议实测，不能直接套用旧任务2:48:30。
CPU全套165 passed、1 skipped；之后新增准入保护另做局部回归。
覆盖tiny Qwen3批量/逐条一致性、角度/范数/梯度重算、首预测/EOS/padding、8方案、hash/阈值精度、缓存/图及真实上游接口。
完整证据见[实现审计](../audit/static_mask_slerp_implementation_20261113/README.md)。
