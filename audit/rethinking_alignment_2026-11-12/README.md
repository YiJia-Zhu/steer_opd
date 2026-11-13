# Rethinking OPD基础配方对齐

日期：2026-11-12。用户授权统一反向KL与原仓库默认；训练、数据生成和下载继续暂停。
[默认配置](../../configs/rethinking_opd.json)为当前入口，[逐项比较](comparison.json)提供机器证据。
修改前文件保存在`before/`，没有改写原fork或共享资产。

## 对齐结果

固定仓库为`ref/ref_rep/Rethinking-OPD`，commit
`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。
比较解析原`on_policy_distillation.sh`并组合其继承Hydra YAML，不执行shell或Ray。
79项训练默认、6项离线评测设置无未解释差异。参考关键源码/配置同时检查无tracked修改；
一个已有未跟踪flash-attn wheel不作为训练依赖已经兼容的证明。

| 项目 | 统一后的默认 |
|---|---|
| 目标实现 | 原fork Student Top-K16，only_stu，student_p奖励与三维PPO surrogate |
| Advantage | token_reward_direct |
| batch / mini-batch / 微批 / n | 64 / 64 / 每GPU1 / 4 |
| 优化器 | AdamW，LR1e-6，betas0.9/0.999，weight decay0.01 |
| 学习率 / 正则 | constant，warmup0，无额外KL/熵正则 |
| clipping | PPO0.2，dual-clip3，grad_clip1 |
| 归约 / 题序 | token-mean，shuffle=False |
| 训练采样 | temperature1、top_p1、top_k=-1、repetition_penalty1 |
| 长度 | prompt1024、response7168；内置验证7168且禁用 |
| 原基准轮次 / 保存 | 1 epoch / 每20步；50步是小预算覆盖例外 |
| 引擎 | 原fork FSDP，dynamic batch、remove-padding、activation offload按原配置 |
| 资源默认 | 原8GPU、TP1、fp32参数/FSDP设置，rollout BF16；未实际占用GPU |
| 离线评测 | avg@16、temperature0.7、top_p0.95、max31744、non-thinking、两个原生stop marker |
| 规则评分 | 直接调用原独立评测的grade_answer_verl，不默认换math_verify |

论文以子集重归一化反向KL解释；源码冻结归一化student_p加权的原生log-ratio奖励，
经PPO surrogate更新。本项目复用原代码，不将它改成教师Top-K前向KL，也不宣称
与直接对子集KL求导完全等价。全词表KL现在只作为另列消融。

## 代码与配置

- [协议模块](../../src/steer_opd/rethinking_protocol.py)组合原配置并隔离Python导入根，
  避免被`scripts/runtime_env.sh`中的`vendor/verl`覆盖；旧loss-only hook环境移除。
- [默认入口](../../scripts/launch_pilot.py)默认仅CPU预览，需要明确execute且恢复授权才考虑运行；
  受控steering组始终因实现未完成而拒绝执行，不以普通OPD冒充。
- [预览](preview_final/launch.json)及[组合配置](preview_final/composed_config.json)可审阅，未导入训练器或初始化Ray。
- [审计工具](../../scripts/audit_rethinking_alignment.py)可重复检查原shell/YAML，而非自我比较配置常量。
- 原始入口归档在[修改前脚本](before/scripts/launch_pilot.py)。其他R1历史工具不作为当前F1入口。

## 保留的明确差异

1. 用户选择100道OT3 OPD题、50个trainer step；不是论文约30K题1 epoch的覆盖量。
   数据须显式循环到3200次prompt访问，不能drop_last后只训64题。尚未生成新划分。
2. 模型按Figure8选Qwen3-4B Non-thinking→真正1.7B-Base，原通用shell活动默认是R1/JustRL/DAPO。
   本地SFT组/下载暂缓。首轮只比较steer相对传统直接OPD，论文SFT数字仅作用户近似参照。
3. AIME2025是本轮唯一测试项，未自动增加论文AIME24/AMC23或MATH-500。
4. 固定seed42、明确non-thinking模板参数；模板/停止事件和教师新增词元仍须逐token验收。
5. 方向、overlap门控、受控生成与一致重放是本项目核心改动；相关后端尚未实现。
6. 日志仅本地console、独立本地Ray会话，不执行原shell的全局ray stop/start。
7. 安装的torch/Transformers/vLLM/verl版本不同于原实验，未验证旧fork GPU依赖兼容性。
   原8×A800不能由当前机器上“配置为8卡”替代，资源调整须记录。

## 检查与限制

10项CPU测试通过，覆盖配置对外部源码、原fork实际导入、执行阻止、原PPO梯度方向、
non-thinking/原规则评分及已有hook语义。原函数小例子的解析梯度与源码一致，且有别于
teacher-forward-KL；这不验证真实模型的受控生成或训练收益。
进程退出时仍观察到已有multiprocess RLock清理异常提示，测试退出码0；不据此宣称GPU环境就绪。

完整材料校验继续发现一个原有历史缺口：
`audit/docs_cleanup_2026-11-10/concurrent_source_snapshots/12_cross_family_research_contract.md`缺失。
未伪造历史快照，未放宽其hash检查。当前方案/链接和配置对齐检查与这个历史完整性问题分开记录。

恢复前仍须完成：独立方向/dev与100题数据准备、EOS/额外词元协议、真实GPU运行环境、
受控rollout及梯度重放一致性。当前只完成基础配方和CPU预览，未运行实验或下载。
