# 模型、数据与下载清单

**2026-11-10更新：** 当前运行与下载暂停；原论文对应优先的模型/数据方案见
[第14篇](14_paper_aligned_models_and_data.md)。本文保留11月7日调查与下载profile，
其中提议的Phi/Llama新配对不再作为正式主线，不能据旧profile自动补下载。

核对日期：2026-11-07。这里把**论文原设置、当前可下载权重、我们建议的新配对**分开。已查论文正文/附录、官方 GitHub 启动脚本和 Hugging Face 元数据；尚未下载权重或复现成绩。

机器可读清单：[download_manifest.json](../experiments/download_manifest.json)；默认仅预览的下载脚本：[download_assets.py](../scripts/download_assets.py)。metadata 与接口检查保存在 [调查目录](../audit/experiment_planning_2026-11-07/)。

## 1. Rethinking OPD 实际用了什么模型？

论文：[2604.13016v2](https://arxiv.org/abs/2604.13016v2)，[本地正文](../papers/sources/2604.13016v2.txt)。以下按实验目的列举主要组合，箭头为教师→学生。

| 位置/目的 | 教师 | 学生 | 需要注意 |
|---|---|---|---|
| §3.1，能力不等于可教性 | Qwen3-4B，non-thinking；对比 Qwen3-4B-Base-GRPO | Qwen3-1.7B-Base | GRPO 教师由 Base 自训；官方 Qwen3-4B-Base 只能作为初始化 |
| §3.2，R1 谱系 | DeepSeek-R1-Distill-Qwen-7B；对比 Skywork-OR1-Math-7B | DeepSeek-R1-Distill-Qwen-1.5B | 名称相近也可能有不同 OPD 效果；需核对教师训练谱系 |
| §3.2，Qwen3 non-thinking | Qwen3-4B；对比 Qwen3-4B-Non-Thinking-RL-Math | Qwen3-1.7B，non-thinking | 后者经 DeepMath 57K RL；不能直接以官方 4B 替代 |
| §4 / §5.2，机制和模板 | JustRL-DeepSeek-1.5B；另用 R1-Distill-Qwen-7B 比较 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL 是 RL 后教师；同为 1.5B 也可能有任务优势 |
| §5.1 / C.1，SFT 冷启动 | Qwen3-4B，non-thinking | Qwen3-1.7B-Base | OpenThoughts3 数学题中采 200K，让此教师生成后 SFT；不是直接取原数据 assistant 回答 |
| B.3，放大学生 | Skywork-OR1-Math-7B；对比 R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Qwen-7B | 后续扩展，首轮不必下载 14B |

原 OPD 训练主要使用 DAPO-Math-17K。§5.1 的恢复实验另用 OpenThoughts3 题目：200K 教师生成 SFT，后续约 30K 去除 SFT prompts 的题做 OPD。SFT 使用 LLaMA-Factory，1 epoch、LR 1e-5、sequence length 14,336。我们用 128/512 校准样本的方案是小预算研究，不能称为这个 200K 设置的完整复现。

默认 OPD 表 2：batch64、rollout n4、Student Top-K16、LR1e-6、temperature1、top_p1、prompt1,024/response7,168。评测 AIME24/AIME25/AMC23，avg@16；评测数据清单仍需按 [06](06_experiment_chapter.md) 的 strict/legacy 口径区分。

模板统一指的是任务指令，例如 boxed 答案格式；不是要求所有模型使用同一 chat template。附录中的 8×A800-80GB 是特定 GRPO 教师构造资源，不能当作本方案所有训练的已验证需求。

## 2. Simple-OPD 实际用了什么模型？

这里的 Simple 指 **Simple-OPD: Demystifying Warm-up for On-policy Distillation**，[2608.06802v1](https://arxiv.org/abs/2608.06802v1)，[本地正文](../ref/sources/2608.06802v1.txt)。

| 位置/目的 | 教师 | 学生 | 公开权重与区别 |
|---|---|---|---|
| §3–§4，主要 warm-up 分析 | DAPO-Math-17K RL 后的 Qwen3-8B-Base | Qwen3-1.7B-Base | 官方启动脚本指定 `RedKiKi/Qwen3-8b-base-rl-dapo-17k`，本次 API 核实公开可访问；不是原始 `Qwen/Qwen3-8B-Base` |
| §5.1，不同 OPD 目标 | DAPO GRPO 后的 Qwen3-4B，non-thinking | Qwen3-1.7B，non-thinking | 不应以原始官方 4B 权重声称复现该 GRPO 教师 |
| §5.2，thinking | Qwen3-4B-Thinking-2507 | Qwen3-0.6B，thinking enabled | 两个官方 HF 模型均可下载，是建议的 P1 |
| §5.3，不同基础模型 | JustRL-DeepSeek-1.5B | DeepSeek-R1-Distill-Qwen-1.5B | 是建议的 P0 |

训练框架是 verl＋vLLM；当前公开 repo 说明其基于 G-OPD/verl v0.6.1。warm-up 从 OPD 题目分布采样教师 CoT；主要方案为低秩 LoRA warm-up，合并后做 OPD。论文比较 full SFT、LoRA ranks16/32/64 及不同 warm-up 步数；其最佳步数不直接等于我们的最佳步数。

原 OPD 表 4：batch128、n1、LR1e-6、prompt2,048/response8,192、temperature1、top_p1。warm-up batch16，full SFT LR5e-6，LoRA LR5e-5。评测包括 MATH-500、AIME24/25，部分组有 AMC23；MATH avg@4，AIME avg@16。OOD 包括 IFEval、GPQA-Diamond、HumanEval，以及 MMLU-Pro 的 physics/chemistry/history。论文称 all experiments 用 8 GPUs，当前读到的设置不能据此确定我们新配置的单卡显存要求。

## 3. 建议下载模型与空间

下表数值是 HF 元数据中 `.safetensors` 文件总大小，十进制 GB，**不是训练显存**；不含 tokenizer、checkpoint、optimizer、KV cache 和下载缓存。revision 已固定在清单里。

| 模型 HF ID | 角色 | 权重约 GB | 访问与下载批次 |
|---|---|---:|---|
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | P0 学生 | 3.55 | 公开；第一批 |
| [hbx/JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B) | P0 教师 | 3.55 | 公开，模型卡链接 JustRL 发布仓库；第一批 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | P1 学生 | 1.50 | 公开；第二批 |
| [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | P1/P3 教师 | 8.05 | 公开；第二批，可复用 |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | P2/P2b 学生 | 3.44 | 公开；第三批 |
| [RedKiKi/Qwen3-8b-base-rl-dapo-17k](https://huggingface.co/RedKiKi/Qwen3-8b-base-rl-dapo-17k) | P2 教师 | 16.38 | 公开；Simple 启动脚本指定 |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | P2b non-thinking 教师 | 8.05 | 公开；需要显式设置模式 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | P3 异构学生 | 6.43 | manual gated，需账号获准；本次匿名读取 config 为401 |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | P3 另一学生候选 | 7.64 | 公开，可避免等待模型访问；仍需异构输出接口 |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Rethinking 教师对比 | 15.23 | 可选；非首批 |
| [Skywork/Skywork-OR1-Math-7B](https://huggingface.co/Skywork/Skywork-OR1-Math-7B) | Rethinking 教师对比 | 15.23 | 可选；非首批 |

`thunlp/JustRL-DeepSeek-1.5B` 当前是 manual gated；首轮清单采用公开 `hbx/JustRL-DeepSeek-1.5B`。未做两者权重逐文件一致性验证，复现实验应明确记录采用哪个发布版本。没有确认的自训 RL checkpoint 不按相近名称擅自匹配。Simple 主教师的公开 ID 已由其启动脚本支持；具体实验 checkpoint 的精确对应仍应由模型表现、模型卡及作者记录进一步核对。

### 3.1 两个首轮配对的接口检查

已实际下载小型 config/tokenizer 文件检查，未下载权重：

| 配对 | 架构与学生空间 | 输出词表映射 | 结论 |
|---|---|---|---|
| R1-1.5B / JustRL-1.5B | 均 Qwen2ForCausalLM；28 层，hidden1536，vocab151936 | tokenizer.json 的 vocab＋added_tokens 映射哈希相同 | 适合共享 ID 的首轮；chat template/停止与思考标记仍需单独固定 |
| Qwen3-0.6B / 4B-Thinking-2507 | 均 Qwen3；学生28层/hidden1024，教师36层/hidden2560；vocab151936 | vocab＋added_tokens 映射哈希相同 | 向量只在 hidden1024 的学生中提取/注入；教师维度不同不影响文本方案 |
| Qwen / Phi-3.5 | 学生 Phi3，32层/hidden3072，vocab32064 | 词表不同 | 不能沿用 token-ID KL；先落实 [跨 tokenizer 计划](06_experiment_chapter.md) |

检查记录：[model_interface_checks.json](../audit/experiment_planning_2026-11-07/model_interface_checks.json)。映射相同不代表教师与学生思考模式一定相容。

### 3.2 两篇方法参考的补充设置

主参考 **Steering Vectors as a Training Signal** 的公开 quick-start 使用 `Qwen2.5-1.5B-Instruct` 学生，监督蒸馏参考为 thinking 模式 `Qwen3-8B`；示例从学生读取教师与自身文本构造方向。这可用于 E2 代码复现，但不是在线 OPD 模型组合，首轮清单不要求再下载这套模型。源码依据见 [固定版本 README](https://github.com/tiejin98/sv_training/blob/198fdbd917e9d7234f734aa8eb754faa408bd2e6/README.md#quick-start)。

CompassOPD 附录 A.1 的主教师/参考是 **Qwen3.5-35B-A3B / Qwen3.5-0.8B**，non-thinking；学生为 Granite4.1-3B、Qwen3-4B、OLMo-3-7B-Instruct-SFT。其教师家族泛化还使用 Qwen3-30B-A3B-Instruct-2507 / Qwen3-0.6B，以及 Ministral-3-14B-Instruct-2512 / Ministral-3-3B-Instruct-2512。它们是强教师/较弱参考配对，不是教师/学生配对。其 SFT 初始化使用教师重新回答的 OpenThoughts-114k，和 Rethinking 的 OpenThoughts3-1.2M 不是同一个数据集。

该原版跨 family 复现较重，论文使用8×H200；当前只准备其 [本地论文](../ref/2609.10154v1_compass_opd.pdf) 与实验对照要求，尚未把这些大模型加入首批下载清单。小模型上的 Compass adapted 需明确标注移植设定；直接比较原版时全部方法共用相同 SFT 初始化，见 [06](06_experiment_chapter.md)。

## 4. 数据下载与使用规则

| HF ID | 配置/切分 | 下载后怎么用 | 优先级 |
|---|---|---|---|
| [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | default / train | `prompt` 已带指令；答案在 `reward_model.ground_truth`，ID 在 `extra_info.index`。当前179万原始行，先核实独立题数、去重、排除评测、再划分 | 首批 |
| [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | default / test，500 | 使用 problem/answer/unique_id；保留 subject/level | 首批 |
| [project-numina/aimo-validation-aime](https://huggingface.co/datasets/project-numina/aimo-validation-aime) | default / train，90 | URL 年份筛2024，得到30题；旧 AI-MO 链接已重定向 | 首批 |
| [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) | AIME2025-I、AIME2025-II，各 test15 | 两配置合并30题，保存分卷 ID | 首批 |
| [project-numina/aimo-validation-amc](https://huggingface.co/datasets/project-numina/aimo-validation-amc) | default / train，83 | URL 年份筛2023，得到40题；83题 legacy 单独报告 | 首批 |
| [google/IFEval](https://huggingface.co/datasets/google/IFEval) | 仓库 `ifeval_input_data.jsonl` | 使用官方指令检查器，不用数学答案提取器 | 第二阶段 OOD |
| [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | test；validation另存 | 筛 physics/chemistry/history，固定 prompt 与选项评分协议 | 第二阶段 OOD |
| [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | gpqa_diamond | auto gated；只用 diamond 配置，获准后下载 | 扩展 |
| [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | openai_humaneval / test164 | 功能正确性评测，固定解码与执行环境 | 扩展 |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | train，1.2M | 按 domain 选择数学，取 conversations 中题目，再让选定教师生成；全库压缩下载约28.2GB | 复现 Rethinking 200K SFT 时再下 |
| [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) | train，103,022 | question/final_answer；先与 DAPO 去重。约2.14GB压缩数据口径，不是 usedStorage | 后续分布变化实验 |

这些数据仓库的 split 名为 train，不代表它们在本项目里可用于训练；AIME/AMC 全部保留为测试。题目划分基于内容与来源，不基于 split 的字面名字。

## 5. 下载命令：建议先 P0，之后按阶段追加

当前实际下载仍暂停，以下为工具使用说明。脚本默认**只打印清单和命令**，不会创建 assets 或下载权重。
恢复下载后可加 `--execute`；使用 HF 本地登录状态处理 gated 模型和数据。
模型默认存放于 `/home/zhenning/data/huggingface_models/<模型名>`，通过 `--models-root` 修改共享模型目录。
`--root` 控制项目内模型软链接、数据和 receipt 的根目录，默认仍为本项目 `assets/`。

```bash
cd /home/zhenning/data/working_tmp/steer_opd
python scripts/download_assets.py --list
python scripts/download_assets.py --profile pilot-r1
```

实际下载首批两模型和五个训练/评测来源：

```bash
python scripts/download_assets.py --profile pilot-r1 --execute
```

只下模型，或者增加第二组：

```bash
python scripts/download_assets.py --profile pilot-r1 --kind model --execute
python scripts/download_assets.py --profile paper-thinking --execute
python scripts/download_assets.py --profile simple-main --execute
```

脚本执行依赖 `huggingface_hub`。新模型调用 `snapshot_download`，固定 commit revision，成功后建立项目软链接并写 receipt；已有共享模型直接复用，不改写文件，并在 receipt 标明其 revision 未重新核实。模型只选择 safetensors 及配置/tokenizer/小型代码文件，不执行模型代码；dataset 取公开数据文件。**下载脚本不筛年份、不去重、不生成训练 parquet**，这些仍按清单的 `dataset_settings` 和 [实验章节](06_experiment_chapter.md) 执行。

默认路径示例：

```text
/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B/
/home/zhenning/data/huggingface_models/JustRL-DeepSeek-1.5B/
assets/models/DeepSeek-R1-Distill-Qwen-1.5B -> 共享目录中的同名模型
assets/models/JustRL-DeepSeek-1.5B -> 共享目录中的同名模型
assets/datasets/BytedTsinghua-SIA/DAPO-Math-17k/
assets/receipts/
```

若手动下载，可查看脚本预览打印的 `hf download ... --revision ... --local-dir ... --include ...`。不要把文件大小当训练显存；正式资源规划见 [06](06_experiment_chapter.md)。

## 6. 论文设置的可复核来源

- Rethinking：正文 §3–§5，附录 B/C、表2/3；[本地论文](../papers/2604.13016v2_rethinking_on_policy_distillation.pdf)。
- Simple：正文 §3–§5、附录 warm-up / hyperparameters；[本地论文](../ref/2608.06802v1_simple_opd.pdf)。
- Simple 教师 ID：[固定版本启动脚本](https://github.com/Utaotao/Simple-OPD/blob/48234d0b9cabf9d0e01a4e8f0e5b965ab751cde6/verl/examples/warm_up/Simple-OPD.sh#L12-L14)。
- Rethinking 公开默认配对：[固定版本示例](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl_example/opd.sh#L15-L16)。
- 各模型 revision/访问状态/权重大小：[结构化清单](../experiments/download_manifest.json) 中每项 `metadata_file` 对应本次 API 记录；这是发布元数据验证，不是成绩复现或权重等价性证明。
