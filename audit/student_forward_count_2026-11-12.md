# 当前OPD为何有额外学生评分前向

2026-11-12。用户指出常见OPD可只有rollout与学习两段，要求解释当前无梯度重放。
本次核对源码，没有修改正在运行的训练目标、候选集合来源或执行流程。

当前流程是学生rollout → HF学生无梯度评分 → 教师评分 → HF学生带梯度更新。
普通OPD与steer OPD均如此；SV并不要求额外多一遍学生评分。

原因是Student Top-K16接口需要每个有效回答预测位置的16个学生候选ID和对应概率，
而当前vLLM rollout配置logprobs=0只记录被采样token的log-prob。
教师阶段先依赖这些ID，之后学习前向在同一固定候选上计算当前概率及surrogate。
该无梯度前向还用于检查HF与vLLM实际rollout的概率一致性。

原Rethinking fork同样如此：

- `verl/verl/trainer/ppo/ray_trainer.py:1111`在教师compute_rm_score前调用学生compute_log_prob，
  注释明确First forward, get student top k ids and log probs。
- `verl/verl/workers/actor/dp_actor.py:695`在torch.no_grad内调用_forward_micro_batch取得Top-K。
- rollout的vllm_rollout_spmd.py:269设置logprobs=0，并注释let actor to recompute。
- 更新时on_policy分支使用当前log_prob_for_loss.detach()作为PPO old；奖励仍使用冻结学生Top-K权重。

所以不能把所有传统OPD都概括为严格两次；两段式实现可行，这个参考fork选择了重算。

保持目标的可选优化：

1. rollout直接返回并持久化Top16 ID/log-prob，再给教师评分，省掉学生HF无梯度前向。
   需要核对vLLM/HF候选/概率、停止与预测位置、归一化支持及通信/落盘成本，不能只保留采样token概率。
2. 学习前向取得Top16后将候选与奖励停止梯度，教师评分，再使用保留的同一学生计算图反向。
   当前每批只一次optimizer更新，在参数未变时可设计为同一目标；需要显存与梯度等价性验证。
   这要求保留学生激活期间运行教师，不能仅删除现有student阶段。

第10步旧计时：普通组学生评分22.23秒、steer组26.98秒（含阶段准入/加载等），
并非整轮训练的主要成本。当前尚未实施以上融合或rollout Top16迁移，原实验继续。
