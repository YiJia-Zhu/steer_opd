# 逐token留档与注入位置对照（2026-11-12）

用户要求核对完整方案中的注入位置，并明确记下未来逐token对照的留档需求，避免因缺数据重复运行。
完整方案第7.3–7.5节定义的是g_t：全位置、overlap、因果后段、独立熵差和匹配随机对照，
同轨迹固定SV。第9.7节现补保存/复用要求；[字段协议](../../docs/06_token_level_records.md)列出文件与边界。

## 已实施

- 提取接口支持BF16/FP32按题/角色/层保存原始激活、分词/offset/正文mask和位置映射，包含最后prompt状态。
  零更新诊断默认开启BF16，普通bank训练按预算显式开启；不默认转储全部训练轮次。
- 池化值、每题原始方向与全局均值继续保存；raw cache先写、pooled后提交，SHA256校验。
  reader用mmap读取，可离线重做池化；元数据被kill后可CPU修复，不重复模型读取。
- 固定前缀统计保存学生/教师Top16、完整支持熵、熵差、overlap/共享质量、mask、位置及bank身份。
  旧教师Top16缺熵时仅补一次同checkpoint前向，保存后各候选复用；不是Top16条件熵。
- HF受控rollout记录正式策略、教师与probe的熵；vLLM raw logits处记录逐位置全支持熵标量，
  返回位置序列必须与输出token逐一一致，mask显式保存。真实GPU验收将与完整捕获logits比对熵，当前未执行。
- 教师评分另写持久化token_statistics目录，保存teacher Top16、逐位置统计和probe区分；
  原有临时scores清理不删除该目录或rollout Top16。
- 第一阶段代码不再安排遗留旧/新题池条件；只用同100题扫层/强度（15候选+alpha0），方向对照保持原计划。
  原v2及pre_opd配置字节、数据和select/confirm划分保持不变；legacy_direction旧字段不再调度。

这些记录可复用相同checkpoint/文本/前缀/干预历史的提取与分析，不能替代新位置策略的前向/训练，
也不能将完整带hook重放当作当前步未注入probe。未缓存层或更新后学生需要新读取。
严格注入率/绝对位置匹配随机在线控制器仍未实现，不因已有普通随机gate而宣称对照完成。

## 验证与执行状态

92项CPU检查通过，包含原80项与12项新增测试：缓存重池化和t−1映射、结构marker位置保留、
缓存损坏拒绝、无模型元数据恢复、全支持熵与Top16条件熵区分、教师熵仅计算一次、
vLLM标量熵的位置/支持/来源校验、熵hash绑定、训练原始统计保留与probe区分、无旧/新题池条件。
命令与XML：

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" -m pytest -q \
  tests/test_token_recording.py tests/test_method_branches.py tests/test_rollout_topk.py \
  tests/test_resumable_opd.py tests/test_generation_runtime.py tests/test_replay_validation.py \
  tests/test_campaign_reporting.py tests/test_training_semantics.py \
  --junitxml=audit/token_recording_2026-11-12/cpu_tests.xml
```

退出0，XML失败/错误均0；第三方SWIG弃用warning与既有multiprocess析构打印未影响测试。
没有运行GPU/训练，没有正式模型的新参考文本/激活/SV/指标，也没有回填旧10步缺失记录。
全部campaign继续paused_by_user；第一步仍T−S/all/S0-global、0次OPD，同题100，结果后停供检查。
新增大缓存尚未生成；磁盘复核约124GB可用，BF16缓存具体体积随有效文本长度而变。

[validation.json](validation.json)保存代码hash、配置/数据不变检查及CPU/GPU状态。
[before/](before/)保存代码/文档快照；完整研究稿before副本由删除本轮唯一新增9.7节无损重建，
保留原CRLF，其余内容未改，未把重建副本冒称为事前快照。

用户明确的“提前留逐token证据”要求已追加到项目AGENTS与Codex ad_hoc记忆，不编辑生成的记忆文件。
