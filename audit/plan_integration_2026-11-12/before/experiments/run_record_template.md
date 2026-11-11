# 单次运行记录

状态：待填写。复制此模板后记录，不在结果出现前填预期数值。

## 配置

- 实验 ID / run ID / 日期：
- 配对 C1/C2/C3/R1 / 面板A原始或B共同SFT / 方法组ID：
- 教师完整模型名 / revision / 模式：
- 教师家族参考R / revision / 模式（如使用）：
- 学生完整模型名 / revision / 初始 checkpoint：
- 总参数 / 文本主干参数 / 激活参数 / 可训练参数 / 等规模误差：
- 冻结学生锚点pi_0 / 原论文对应 / 本项目扩展与adapted项：
- 训练仓库 / commit / 环境：
- GPU / dtype / 量化：
- 学生和教师 tokenizer / native chat template：
- 任务指令 / 输出格式：
- 原生span采样或有限Candidate-RKL / K和位置数 / sum或mean评分：
- 候选学生原始保留质量 / 有效候选数 / 位置与span覆盖 / 丢弃原因：
- 候选文本是否互为前缀 / 教师分数不可累加为覆盖率的说明：
- tokenizer、空格、UTF-8、EOS与稳定编码边界检查：
- 蒸馏目标 / 归一化 / 学习率 / batch / seed：

## 数据与向量

- 校准、正式 OPD、开发、测试 ID 文件与哈希：
- 原始行数 / 独立题数 / 跨 split 去重 / AMC strict40 或 legacy83：
- 教师示范筛选 / 验证器 / 覆盖率：
- T/R/S各自文本缓存路径 / 完整三文本题目ID / 不完整或错误参考处理：
- 同教师多学生是否共享同题和同一份T/R文本；共同校准与原生题源分别标记：
- 激活读取层 / 位置 / pooling / mask：
- 方案1/2 / v_total、v_cap、v_offset / 统一三文本池化与代数分解误差：
- 向量文件 / 范数 / 校准样本数 / 随机方向种子：
- 注入操作 / 层 / 精确位置 / 强度 / 实际扰动范数：
- hook 启用阶段 / 辅助更新数 / 撤去安排：
- gradient-checkpoint 重算与 packing mask 检查：

## 成本与评测

- 更新数 / 学生生成 token / T、R、pi_0生成及评分 token分别计数：
- T/R/S学生prefill调用数与token / 方向与SFT搜索预算 / 加载与merge成本：
- 对齐预算口径（更新数、GPU-hours、教师 token）及不等价之处：
- 校准搜索与正式训练分别的 GPU 时间 / 峰值显存：
- 评测配置 / checkpoint / hook 是否关闭：
- 任务指标 / 长度 / 重复 / 停止行为：
- 截断率 / avg@n 或 pass@n / 种子 / 置信区间：
- 共享词表原生Overlap或异构扩展指标（明确名称）/ 共享质量 / 粒度与熵口径：
- 参数未更新时移除hook恢复检查；更新后H/H+1/H+20无hook结果分别记录：
- 原始日志、逐题结果和 checkpoint 路径：
- 与对照的差异 / 失败现象 / 下一步：
