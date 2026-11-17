# S−T实现与sv_training的一致性检查

当前检查没有发现逐题差分符号、单位化、global均值构造或抽查的向量绑定错误；但当前S−T overlap实验不构成sv_training论文的严格复现。已确认注入位置和池化精度等实现细节不完全相同。现有证据不能把global优势直接归因于某一个差异，也不能证明端到端GPU行为已经与官方逐位等价。

本次只做源码阅读、已有产物检查、随机小型模型的CPU前向和数值对照，没有生产教师／学生的新生成或GPU评分、参数更新或生产代码修改。机器可读结果见[JSON](sv_training_consistency_20261117.json)，可复查脚本见[检查脚本](check_sv_training_consistency_20261117.py)。

## 1. 对照来源与阅读范围

- 论文：[Steering Vectors as a Training Signal in LLM Post-Training](https://arxiv.org/abs/2610.40998v1)。本次核对§4.1、§4.2.1／4.2.3、§4.3、附录B、C.2及D的相关段落；未声称重新复现全部论文实验。
- 官方仓库固定commit：[198fdbd917e9d7234f734aa8eb754faa408bd2e6](https://github.com/tiejin98/sv_training/tree/198fdbd917e9d7234f734aa8eb754faa408bd2e6)。本次重新获取steering.py、trainer.py及相关源码；acts.py、generate.py、train.py、vectors.py与项目此前保留的固定副本SHA完全一致。
- [下载清单及SHA](../.runtime/sv_training_audit_20261117/download_manifest.json)。正文实现以公开代码为准；论文附录D对raw／normalized的部分叙述不能覆盖代码中Slerp强制单位化的实际行为。

## 2. 先核对当前结果

以下为Confirm、题目平均Δoverlap，单位pp：

| alpha | all／global | all／per_question | static／global | static／per_question |
|---|---:|---:|---:|---:|
| .05 | +.4245 | +.3061 | +.1534 | +.1251 |
| .1 | +.7077 | +.4362 | +.2989 | +.2467 |
| .2 | +.7277 | +.0446 | +.5426 | +.4460 |

“强度越大越好”和“all最好”不能推广到全部方案：all／per_question在.2明显回落，该强度下static／per_question优于all／per_question。.1的all／per_question有277／300题改善，global有300／300题改善；global在264题上超过per_question，per_question在35题上超过global，1题持平。逐题SV有作用，只是该指标下平均效果较弱。

## 3. 方法设置差异

| 项目 | 官方sv_training | 当前S−T |
|---|---|---|
| 被读取／被注入模型 | 学生 | 教师 |
| 差分 | H_S(教师／GT回答)−H_S(学生自身回答) | H_T(学生回答)−H_T(教师自身回答) |
| 注入的文本 | 教师／GT参考文本，参与学生训练 | 固定学生源文本，无参数更新 |
| 主指标 | 训练后撤去SV的测试准确率 | 即时overlap／P_overlap |
| 默认层位 | 代码L//2+1，0-based | 教师40层中的29，75% |
| headline对照 | global Mean-SV使用add、alpha=.5；per-sample使用Slerp、alpha=.1 | global和per_question均使用Slerp，逐档比较相同alpha |
| bootstrap生成 | 默认greedy | 学生temperature=1；教师temperature=.7、top_p=.95 |
| 数据／参考质量 | GSM8K／MATH等；蒸馏训练默认保留通过verifier的教师回答 | OpenThoughts数学子集；按用户要求保留未验真的全部可用回答 |
| SV上下文 | prompt+回答最多2048，回答token均值 | 相同总上下文与正文均值规则 |
| 注入／评分上下文 | SFT records跳过超过2048的训练样本 | 沿学生完整回答评分，学生生成上限7168 |

因此，论文的per-sample Slerp优势不能直接当作“教师空间、0更新、overlap”下逐题必优于global的预期。官方主表也没有直接比较同alpha的global Slerp和per-sample Slerp。

## 4. 已有产物的实际检查

- 检查全部1000个教师空间方向：每个逐题raw方向等于该题保存的两份pool之差；符号为学生回答pool减教师自身回答pool；没有零方向。
- 将保存的两端pool送入官方mean_vector_from_acts，结果与已保存global raw方向逐位相同。global采用先对原始差分求平均、再单位化。
- 全部1000个Slerp单位方向与官方Injection的归一化结果逐位一致；共同激活尺度不参与Slerp强度的额外缩放。
- Select／Confirm覆盖全部1000个唯一题目ID，均能在逐题bank查到。按两个split等距抽查all/global与all/per_question共64条已保存记录，vector_id及condition均与本题的预期payload一致。此项为保存记录抽查，不是对全部GPU内部payload传输的重新验收。
- 固定EasySteer代码向Slerp传入完整hidden+residual，修改以delta形式写回hidden；capture也保存相同完整block输出。这一代码路径没有把MLP分支误当完整隐状态。

## 5. 已确认的实现口径差异

### 注入位置整体差一个token

官方[steering.py](https://github.com/tiejin98/sv_training/blob/198fdbd917e9d7234f734aa8eb754faa408bd2e6/sv_training/steering.py#L195)按labels != -100选回答输入位置；trainer的labels在prompt之后才有效。当前[selector_for_predictions](../src/steer_opd/static_masks.py)按预测token的位置选prompt_len−1+t。

以prompt长5、回答长4为例：官方选[5,6,7,8]，当前选[4,5,6,7]。当前因此包含prompt末位置，直接影响第一回答token预测；官方回答mask不包含该位置。当前规则符合项目此前明确的预测位置协议，但与官方代码不等价。如果声称严格复现，应先独立核对／对照这两种口径，不能悄悄改写已完成实验。

### BF16均值与FP32均值

官方acts.py先对BF16 hidden求mean，再转FP32；当前先把captured BF16状态转FP32，再mean。随机小型CPU Qwen测试中，中间block的FP32版本完全一致；BF16版本在当前结果再round至BF16后与官方均值一致。把1000题现有pool按官方均值精度round后重建方向，与当前方向的平均余弦为0.999977，最小0.999888。这表明量化差异在本组向量上较小，但没有据此声称Top16排名绝不受影响。

### Slerp几何相同，数值实现不逐位相同

两边都把h旋向单位SV方向，alpha是角度比例；当前没有把Slerp误写成先乘向量幅度的线性相加。官方使用acos／sin插值；当前使用atan2／切向量形式，附加范数修正及零向量／反向共线保护。128×2048随机CPU对照、alpha=.05／.1／.2：FP32相对L2误差约1.1e−7，BF16约1.07e−4至2.30e−4。普通位置几何吻合，不能称逐位相同；共线退化处理也不同。

### 100%层位的final norm边界

官方acts.py读取hidden_states[layer+1]；HF Qwen的hidden_states[-1]包含最终归一化，而block forward hook读取最终归一化之前的输出。随机CPU Qwen测试确认最后一层与中间层有不同的边界语义；当前HF Qwen3.5的capture_outputs默认也把最后hidden state绑定到last_hidden_state。当前实验用29／40层，不受最后一层边界影响。下一轮100%层位必须明确保留block输出口径，或另做官方边界对照，不能把norm后的提取与norm前的注入混为同一空间。

## 6. global更好的可能原因及证据边界

1000个逐题方向与global的平均余弦为0.774、中位数0.783，说明多数方向有较强共同成分，同时含有题目特异成分。平均后会减少部分题间差异；在所有题上做共同分布适配时，这可能更适合当前overlap目标。逐题方向与global越不一致，global相对逐题的优势倾向越大：这一相关性在Select为−0.382、Confirm为−0.293。相关关系支持上述解释，尚未证明题特异成分都是噪声，或global一定有更高下游学习收益。

“从本题提取”也不保证逐token指标提高：当前方向压缩了整段回答的均值差分，对同题所有位置复用；它不是每个位置的overlap或标签概率梯度。特别是提取只读2048总上下文，评分却覆盖完整回答，后续位置的偏移未必与早段均值方向一致。

现有text_quality另记录教师853／1000条触达生成长度上限；教师969／1000、学生738／1000条在SV提取时触发2048总上下文截断。这不等于回答错误，但与官方经过验证的参考训练文本存在明显数据条件差异，不能默认题特异差分都提供正确的局部修正。

优先的下一步应是冻结其余条件，对照官方回答输入mask与当前预测mask，统一池化／Slerp数值口径，并利用已有sampled_logps报告参考文本NLL变化；若要复现论文结论，还需单独恢复学生T−S、训练与撤去SV的最终评测。以上额外GPU对照和训练均未运行。当前先保留已有结果，不能用这份静态／CPU检查宣称已消除全部实现或方法差异。

## 7. 复查命令

```bash
source scripts/runtime_env.sh
CUDA_VISIBLE_DEVICES='' "$STEER_OPD_PYTHON" audit/check_sv_training_consistency_20261117.py
```
