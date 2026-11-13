# 完整方案对照覆盖复核（2026-11-12）

用户要求再次核对完整研究稿中备选/分支/对照是否真正实现。
结论：核心开关已接入，完整对照面板尚未齐全。不是所有文档对照都有实验入口，也不能把CPU算子测试当作GPU/学习验收。
逐项清单共29项：[机器可读清单](coverage.json)、[阅读版](../../docs/07_control_implementation_matrix.md)。

## 主要缺口

- 固定概率random仅为Bernoulli抽取；等注入率/次数与位置分布匹配的随机对照未实现。
- 当前all面板默认vLLM、门控面板HF；面板内普通组虽匹配，但跨gate一因素消融缺共同后端调度。
- 三方向、全局/逐题等内核已有；共同文本、共同有效交集、共享尺度及范数夹角的公平比较还没有统一流程。
- 阈值/后段边界标定、两层网格自动选型、多训练seed、同实际计算预算及机制续写/正确性对照未完整编排。
- rollout-only/learning-only/末期撤去、模板/长度/共同正确集合、Compass/Entropy-Aware目标、共同预热、跨tokenizer训练等不在已完成范围。
- 独立D_vec有历史划分/旧T−S实现，但新的冻结dev和预算的T−R独立全局对照未形成当前面板；当前第一步不加入旧/新题集比较。
- SFT继续按用户早先要求暂缓，不因审计新增训练/下载。

## 审计中修正的三个问题

1. `overlap_tuning.evaluate_candidate`现在按真实row mask记录。旧绑定记录若仅mask日志写错，可CPU更新元数据，原概率/指标不重算。
2. 复用bank除了文件SHA，还核对bank/metadata方法、提取checkpoint与step。防止把旧全局/旧rho/旧提取者bank拷入新run后跑错实验。
3. `resolve_method`显式拒绝未实现的injection阶段/日程；旧loss-only helper不能被当作当前受控策略实验。

这三项是现有实现的正确性修正，没有补齐缺失对照，也没有改变当前第一步参数、配置文件、数据或运行状态。
没有启动GPU，没有生成新正式缓存/SV/指标，没有重启OPD。

## CPU验证

99项通过，failures/errors均0。新增7例覆盖：三种未实现阶段拒绝、bank方法/checkpoint/step三种错配拒绝，
以及真实小Qwen3双层混合mask的留档与无模型重算的旧元数据修复。

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q \
  tests/test_control_coverage.py tests/test_token_recording.py tests/test_method_branches.py \
  tests/test_rollout_topk.py tests/test_resumable_opd.py tests/test_generation_runtime.py \
  tests/test_replay_validation.py tests/test_campaign_reporting.py tests/test_training_semantics.py \
  --junitxml=audit/control_coverage_review_2026-11-12/cpu_tests.xml
```

退出0；第三方SWIG弃用warning/既有multiprocess析构打印不改变XML的0失败/0错误。未修改共享环境。
[validation.json](validation.json)记录源码hash、配置与暂停状态；[before/](before/)保存本轮修正前快照。
所有新GPU接口仍未实测。第一步继续T−S/all/S0-global、0次OPD、原100题，结果后停供用户检查。
