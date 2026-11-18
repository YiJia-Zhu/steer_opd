# 重复生成相关文献笔记

相关本地诊断实验已按用户授权删除；此处仅保留文献整理。

## 3. WSP仓库实际怎样修复循环

核对仓库版本`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`的[实验8说明](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)、[50-pass最终报告](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md)和[逐次工具代码](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl)。下载的代码仅阅读，未执行。

他们修改的是Qwen3.5-0.8B，thinking开启、24层、hidden size1024。原评测已经使用temperature0.6、top_p0.95、top_k20、presence_penalty1.5，输出上限32768。因此，不能把结果解释成简单添加重复惩罚，或通用vLLM重复生成软件补丁。

50-pass方案的关键步骤：

1. 为4道练习题人工配对“正常推进的推理”和“不断重复同一疑问的推理”，避免把结束思考或最终答案标记直接当作区分特征。
2. 比较原模型的中间激活。最终方向来自每条配对上下文末36个位置的残差均值之差，再跨4对求均值和单位化；对应MLP层15/19后的残差输出16/20。这里没有PCA。
3. 用“循环与正常上下文的MLP激活差 × 该神经元向循环方向写入的权重”给神经元排序，分别选择64个。最终选择使用原始fwd4；另做迁移上下文检查。
4. 只编辑层15和19的`mlp.down_proj.weight`选中列，冻结方案后在独立练习题及最终benchmark上验证。

最终公式为：

```text
W_new = W_old - 6 × outer(u, mask64 × (uᵀ W_old))
```

`u`为循环方向，`mask64`选择64个神经元输出列。系数6会把选中列沿u的分量变成原来的−5倍，属于反向增强干预；不能称为普通强度1的正交投影。两层共涉及131072个权重位置，BF16舍入后130188个值实际改变，矩阵相对改变量约10.64%和8.11%。

他们还比较了不同剂量、单层干预、等范数随机正交方向、正常/重复分支的条件似然，以及是否诱导过早结束。最终报告给出的迁移测试中，重复分支每token logprob约下降0.103，正常分支仅下降0.009。最终MATH-500均值准确率7.9%→48.1%，达到长度上限的比例93.0%→33.1%；还有残余循环和其他失败。

这些证据支持“存在可定向削弱的循环机制”。作者另做的[Gemma后续实验](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/README.md)中，较少循环的模型未出现类似提升，MATH变化不到约1个百分点。因此该Qwen结果主要体现释放被循环阻塞的表现，不能外推为权重编辑普遍增加数学能力。

### 一个影响结论的重要外部复核

[Issue #4](https://github.com/RikiyaT/lm_understand-logs/issues/4)的独立复核者报告：原始循环设置约7.2%；不改权重、改用Qwen3.5-0.8B的非thinking推荐设置并设置presence_penalty2.0后，2000个样本准确率44.0%，达到上限仅0.3%。这是第三方实验，核对了其报告和记录说明，本轮没有复跑。

该对照同时改变thinking、温度和惩罚，无法单独归因于某一参数，也没有否定原权重干预在原设置下有效。但它提醒我们：研究循环时应先比较合理采样基线，不能只以极易循环的解码设置衡量方法收益。Issue中的“decided-only”49.8%采用另一分母，不能混入普通准确率比较。

我们学生为36层、hidden size2560的Qwen3-4B，结构也不同。WSP的层号、向量、64个神经元和系数6均不能直接照搬。可借鉴的是**同模型、同问题的正常/循环配对和可逆干预对照**。我们当前教师S−T方向比较的是师生回答，不是专门提取的循环方向。


## 4. 论文和技术资料怎样解释重复

| 资料与阅读范围 | 与当前问题相关的发现 | 对我们结论的边界 |
|---|---|---|
| [Learning to Break the Loop / DITTO，NeurIPS 2022](https://arxiv.org/abs/2206.02369v2)，摘要及正文§2–3 | 上文出现同一句话后，模型更倾向继续复制；重复次数增加会进一步提高复制概率。用人工重复上下文训练，使后续重复概率衰减，并混合正常训练保护合理复用。 | 主要实验是GPT2/BART、开放生成及摘要，不是当前数学OPD；本地按循环轮次观察到类似现象。 |
| [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319v2)，摘要及正文§5.1–5.2 | 显式对不希望生成的重复候选优化`-log(1-p)`；在模型自身生成的上下文上做序列级负训练，并混合原训练目标。 | 正常公式、变量和必要论证会重复，不能把所有重复token都处罚；数学场景需重新验证。 |
| [When Speed Kills Stability](https://richardli.xyz/post/rl-collapse-training-inference/)，重点阅读§3.2、§4和§5 | vLLM与训练前向的概率差可能放大低概率token的误差；长轨迹会积累偏移。其序列级重要性修正/筛选比部分token级方案更稳定。 | 其典型退化伴随熵/梯度爆炸，我们主要是熵下降、梯度变小；概率不一致是已测到的风险，尚非循环的已证实首因。 |
| 本地Rethinking OPD v2正文§6.1 | 教师在长学生前缀后的纠偏收益下降，学生状态会逐渐偏离教师自然轨迹。 | 论文的3K/7K甜点和10K/15K崩溃不能直接作为我们的阈值；我们已使用7168上限。 |
| [Qwen3-4B官方模型卡](https://huggingface.co/Qwen/Qwen3-4B#best-practices)，本地实际模型README | 明确提示无限重复风险；non-thinking推荐T0.7、top_p0.8、top_k20，重复严重时建议presence_penalty1.5。 | 我们原协议为non-thinking、T1、top_p1、不截top_k、无presence惩罚；S0/S60协议一致，所以采样设置本身解释不了训练后的增量退化。 |

资料共同指向一个机制：模型把历史重复当成很强的复制线索；进入后局部预测越来越确信，局部高概率未必对应全局任务进展。我们当前教师也顺着循环前缀给出高概率，逐token匹配就很难帮助学生离开。

