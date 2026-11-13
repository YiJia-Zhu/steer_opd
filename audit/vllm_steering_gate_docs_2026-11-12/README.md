# EasySteer位置注入与在线overlap门控边界

2026-11-12，官方固定版本文档与本地源码核对。仅CPU/网络读取及文档留档；未加载模型、安装环境、修改训练代码/配置或恢复GPU实验。

## 官方来源

EasySteer固定commit：`baecd9698f32aadef68b41536f3db477b513a0d7`。
其vLLM fork固定commit：`7837b07d21c0ddffd043a30001f2c5654290aec7`。
原Steering指南与fork源码保存在`../vllm_easysteer_review_2026-11-12/`。
本目录`retrieval.json`记录补取的API、引擎、quickstart、performance文档的URL、长度和SHA256。

- [官方Steering指南](https://github.com/ZJU-REAL/EasySteer/blob/baecd9698f32aadef68b41536f3db477b513a0d7/docs/user-guide/steering.md)
- [官方API字段](https://github.com/ZJU-REAL/EasySteer/blob/baecd9698f32aadef68b41536f3db477b513a0d7/docs/api-reference/steering-specs.md)
- [官方性能说明](https://github.com/ZJU-REAL/EasySteer/blob/baecd9698f32aadef68b41536f3db477b513a0d7/docs/user-guide/performance.md)

## 已确认

1. EasySteer fork支持`SteeringSpec`、`VectorSpec`、`DirectionVector`、`ApplySpec`与`llm.generate(..., steering=spec)`。
2. `ApplySpec`支持prompt位置、generation位置/窗口、token ID条件及排除项；可逐请求传不同spec。
   首回答预测必须处理最后prompt状态；答案位置到hook输入状态有t−1关系，实际引擎索引仍需验收。
3. 单向量direct可使用in_graph执行，选择buffer随调度更新，不必因位置筛选全面关闭CUDA Graph。
4. 请求在admission时解析并保留其有效配置；`set_default_steering`只影响新请求。
   所查公开接口没有直接接外部教师分布、逐token计算overlap后修改在途请求的现成回调。
5. 当前项目上游vLLM0.32的`entrypoints/llm.py`与`sampling_params.py`未发现该EasySteer API；
   项目通过`src/steer_opd/vllm_worker.py:SteeringWorker`和`vllm_steering.py`自定义hook实现全位置注入。
   本机旧conda easysteer环境不等于上述固定版本，不能据环境名宣称API已可用。

## “先算overlap再给index”的适用范围

- 固定checkpoint、固定前缀、匹配的干预历史：已有overlap可用于该前缀的位置诊断或重放。
- 预先给定的后段/随机/固定mask：可直接由位置selector执行；这是已知位置注入。
- 自由生成精确在线gate：必须在实际受控前缀上先算当前步probe/教师分布，决定注入后再采样。
  一旦steer改变token或历史KV，原无hook轨迹后续overlap不再是当前步指标。
  checkpoint更新后也不能沿用旧学生分布。固定旧mask可以作为独立变体，不能默认为在线主方案。
- probe保留此前受控历史，仅本步无注入；不能每一步另起一个全程无hook学生作为等价替代。

## 当前慢在哪里

`scripts/figure8_generate_controlled.py`对pending请求逐条调用HF `generate_response`，尚无vLLM连续批处理。
`controlled_step`每步获取教师q；在线gate还获取学生probe，g=0复用probe，g=1从同一旧历史另做正式学生前向。
KV分支通过复制append-only DynamicLayer容器隔离，不是每步完整拷贝所有历史KV tensor。

简化学生步成本约为`(1+r)*C_student`，r为触发比例；这不是整个OPD时间倍数，也未计prefill、输出层、同步及其他开销。
教师原本就是普通OPD的一项成本；在线gate将其从rollout后的批量teacher forcing移到逐token关键路径，
分数已保存复用，不能误算成原教师评分之外又完整重复评分。向量加法不是主要成本。
EasySteer可改善执行效率，但精确在线gate的probe、教师交互与KV选择仍需专门集成。
未测正式模型门控吞吐，不声称具体加速倍数。

## 执行状态

用户倾向后续门控生成也使用EasySteer。当前只确认接口与集成边界；未完成后端迁移。
第一步仍T−S/all/S0-global、0次OPD，实验保持暂停；不会为此自动启动训练或改成loss-only。
