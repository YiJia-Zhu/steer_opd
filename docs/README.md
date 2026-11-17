# 文档索引

- [Ablation实验计划](ablation.md)：overlap（原100/1000题四方案完成，新增教师S−T待验收）→param→baseline，含当前4方案、双指标与执行边界。
- [Overlap完整算例](overlap.md)：教师/学生分词、Top16最大一一前缀匹配、Top16原始概率最大兼容质量与静态mask的逐步计算。
- [实现与运行说明](implementation.md)：EasySteer、缓存、逐token留档、绘图和验证。
- [主线代码框架逻辑](代码框架逻辑.md)：教师S−T全局方向准备、vLLM／EasySteer／verl分工、逐批OPD循环、初始与训练生成的关系、缓存及耗时口径。
- [完整研究方案](Steering_OPD_完整研究方案.md)：保留研究定义与后续阶段；§12.2定义两种跨family指标。
- [文献来源](sources.md)：论文结论、阅读范围及实现依据。
- [TODO](../TODO.md)：当前已完成事项和待用户同意的执行任务。

过期调参网格、旧动态运行日报与重复审计已清理；原四组固定Slerp0.1/学生层26；显式开关新增S−T教师层29的四组镜像消融，当前仅代码／CPU验证，不自动启动实验。
