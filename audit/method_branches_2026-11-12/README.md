# 方法分支与独立零更新入口实现（2026-11-12）

用户授权完成代码分支，第一步继续T−S、全回答预测位置注入、S0全局均值。
实现与CPU准备已完成；没有启动GPU、下载权重/数据、生成新校准文本或恢复OPD。
[使用与边界](../../docs/05_method_branches.md)；[第一步配置](../../configs/pre_opd_overlap_tuning.json)。

## 保持不变

- `configs/figure8_shared_top16_v2.json`与修改前快照字节一致，100题direction/opd、dev128/AIME30不改。
- 旧两组10步及Base评测保留；旧campaign和v2仍paused_by_user。
- 当前第一步只做即时overlap：T−S、S0、global、all，OPD更新数=0。
- 独立诊断默认CPU准备，已冻结select16/confirm16；尚未运行参考生成/方向提取/评分。
  新run也为paused_by_user，没有GPU attempt或训练checkpoint。

## 实现

| 部分 | 实际入口/模块 | 内容 |
|---|---|---|
| 参数分支 | method_config.py / steering_branch_presets.json | T−S/T−R/R−S，global/per_question，initial/current，周期刷新，独立gate模式 |
| 提取 | extraction.py / figure8_extract.py / direction_bank.py | 同一学生读选定角色文本；多层共同有效配对；全局先均值后归一化；逐题bank与版本/hash |
| 快速生成 | figure8_generate.py / vllm_worker.py / vllm_steering.py | 原始全局vLLM路径保留；bank支持全局多层及按请求选逐题向量；默认调度，不压num_seq |
| 门控 | controlled_rollout.py / figure8_generate_controlled.py | 采样前同前缀教师与当前步无hook probe，历史受控KV隔离；未触发复用probe；记录实际mask |
| 评分/学习 | figure8_stage.py / resumable_core.py / steering.py | 在线教师分数CPU整理复用；按每题bank和记录mask重放；所有有效回答token仍有loss；backward重算保持hook |
| 中断 | controlled_rollout.py / rollout_topk.py | 每32 token保存两个RNG/轨迹/概率；恢复一次重建历史KV；完整回答以tensor先、JSON后原子提交 |
| 真实模型验收 | figure8_branch_acceptance.py | 实际vLLM bank logits/HF门控全支持双向KL与临时更新；仅在恢复训练阶段调用，当前未运行 |
| 诊断 | run_pre_opd_overlap_tuning.py / pre_opd_overlap_stage.py / overlap_tuning.py | 独立0更新入口，select网格、冻结confirm候选、逐题CSV/表格/区间/共享质量/位置与长度桶；结果后退出 |
| 配置创建 | create_steering_variant.py | 必须提供冻结阈值/R等信息；独立run/data/config，authorized=false及execution_hold，不自动启动 |

全位置普通/steer两组继续使用vLLM。门控面板两组均用同一HF采样器（普通组无SV/probe），
避免把后端差异算成效果；这一参考后端按请求执行、KV复用，尚未做整批吞吐优化/测量。
逐题vLLM bank目前使用eager decode来确保每请求向量绑定；全局bank可保留decode图。
上述新GPU路径均未验收，不声称达到原vLLM吞吐或方法有效。

每题向量绑定question_id、bank/vector_id、提取checkpoint/step与文本hash，训练前不会换掉已有轨迹方向。
S0校准尺度冻结，current重提保持该尺度；跨方向公平比较可显式共享fixed_activation_scales。
无效/近零逐题方向明确失败，不自动用全局补齐。T−S/R−S的当前学生分支计入无hook bootstrap成本。

新增门控的overlap/signed-entropy阈值、随机概率、late边界没有根据测试集或猜测赋默认值。
尚未实现自动在线滚动/绝对位置匹配随机门控，也未加入SFT/Compass、多seed、多配对或跨tokenizer完整训练。
这些属于后续研究面板，未因代码分支实现而宣称完成。

本轮也修复了一个验收接口问题：当Top16验收要求每个位置的完整logits时，worker现在捕获全部位置，
不再只捕获旧固定5个位置。生产rollout不因此保存完整logits。

## CPU验证

80项通过，failures/errors为0。实际小Qwen3模型测试包括：三种差分的同空间分解、全局/逐题与等强度，
probe与正式KV分支隔离、六种gate设置下cached生成与记录mask全重放一致、混合mask的checkpoint backward梯度，
采样/Gate RNG和KV中断恢复、在线教师复用、后端调度及普通组匹配、原Student Top-K surrogate、
独立诊断无backward/optimizer且结果后停止、select/confirm不重叠、旧hash/断点兼容与第一步设置保护。

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" scripts/run_pre_opd_overlap_tuning.py
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q \
  tests/test_method_branches.py tests/test_rollout_topk.py tests/test_resumable_opd.py \
  tests/test_generation_runtime.py tests/test_replay_validation.py \
  tests/test_campaign_reporting.py tests/test_training_semantics.py \
  --junitxml=audit/method_branches_2026-11-12/cpu_tests.xml
```

pytest退出0；有第三方SWIG弃用warning，进程退出时另有既有multiprocess资源追踪析构异常打印，
XML的failures/errors为0，没有修改共享环境。
[validation.json](validation.json)记录文件hash、暂停/数据/第一步设置检查和GPU未运行状态；
[before/](before/)保存本轮开始时的代码/文档快照。CPU实现验收不作为正式模型训练收益。
