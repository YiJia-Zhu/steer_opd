# Steering-assisted OPD

实验统一命名为**ablation**，按`overlap → param → baseline`推进，唯一执行计划为[ablation.md](docs/ablation.md)。原100题overlap已完成；1000题原四方案也已完成，当前在GPU6、7补跑1000题教师空间S−T镜像消融（验收通过，正在提取SV），参数实验、OPD训练和baseline对比后续另行开展。

固定Qwen3.5-35B-A3B教师、Qwen3-4B学生、T−S方向、Slerp0.1与0-based层26。比较all/static_overlap × global/per_question四组及共同无SV基线，学生冻结、0更新。同一S0读取教师回答与自身无SV源回答，按sv_training规则截取prompt+回答前2048 token，对保留回答token逐题池化后作差，构造全局/逐题两份bank；仅保存池化均值/范数与SV，不保存逐token隐状态。

同时记录Top16前缀兼容最大一一匹配overlap与Top16原始概率最大兼容质量P_overlap。共用vLLM返回的16个ID/logprob，不重新归一化，不缓存完整词表。定义见[指标说明](docs/overlap.md)，metric_version=4。模型前向均走EasySteer/vLLM；默认HF_PARITY=0，显式设1才加HF参考。教师生成上限12288、学生源生成上限7168，2048仅用于SV读取。

当前唯一模板为[ablation.json](configs/ablation.json)，保留execution_hold=awaiting_user_approval。既有GPU0、1上的1000题四方案已完成；本轮教师S−T补充实验已启动，仅方案5–8，运行配置独立，原结果保留。默认RUN_NAME=ablation、PARAM_NAME=n1000_slerp0.1_layer26_seed42_v1、STOP_AFTER=panel、CONFIRM_TOP_N=4，四方案在700题select和300题confirm均评分。global提取池包含全部1000题，confirm属于训练侧复核。

[数据准备脚本](scripts/prepare_ablation_data.py)从本地OpenThoughts3数学子集扩展并冻结1000题。已完成100题和1000题的全部数据、回答、SV、概率、指标和图已真实移动到outputs的step0／1／2／3／6，参数名分别为`n100_slerp0.1_layer26_seed42_v1`和`n1000_slerp0.1_layer26_seed42_v1`。原文件SHA保持，旧runs／data／logs目录已移除，其余无用实验结果已删除。各批分析和对照CSV在step6，两批综合分析使用`n100_vs_n1000_slerp0.1_layer26_seed42_v1`。

手动入口为[scripts/run_ablation.sh](scripts/run_ablation.sh)，开头可设GPU、方案与超参数。CPU数据准备后，`bash scripts/run_ablation.sh --check`预览，`--run`重新验收并执行或恢复，`--status`只读看进度。新结果按[输出结构](输出结构.md)保存于`outputs/step_<编号>_<阶段>/ablation/<模型组合>/<训练数据集>/shared/<PARAM_NAME>/`，原生EasySteer/vLLM tqdm保留，外层仅打印阶段。每个公开引擎调用返回后整批提交，失败从最后完整阶段恢复。

[文档索引](docs/README.md) · [完整研究方案](docs/Steering_OPD_完整研究方案.md) · [实现说明](docs/implementation.md) · [TODO](TODO.md)

仓库仅保留当前静态实验调用链、显式HF参考和相应测试；旧训练/动态门控及模型数据下载代码已删除。文献和固定源码在papers/和ref/，共享模型/数据只读，教师权重仅核验。清理范围见[删除清单](audit/repository_cleanup_20261114/manifest.json)。退火保留为后续真实训练批次消融，本阶段四组均为固定强度。

新增教师消融：TEACHER_ABLATION=1启用方案5–8，教师读取学生回答与自身回答，在教师层29计算“学生−教师”均值差后注入教师，沿固定学生推理评分。原T−S方案1–4注入学生层26，默认入口仍保留原四组。仅CPU预览：`TEACHER_ABLATION=1 bash scripts/run_ablation.sh --check`。两种方向分别在各自隐空间构造，联合比较共用无SV基线；教师组增益不代表学生学习收益。模板hold保留，当前仅独立补跑S−T四组，不进入OPD。

改变alpha会复用数据、回答、SV和原生概率；改变层位只重算对应SV及受影响下游。所有复用先校验科学依赖和文件SHA，新的前缀或checkpoint仍须重算对应产物。入口用`PARAM_NAME`区分参数，`RUN_NAME`只表示`ablation`／`param`／`compar`；step4训练与step5测试尚未实现，不自动执行。阶段输入、引用与失效规则见[输出结构](输出结构.md#7-参数变更与复用规则)。

迁移细节及目录位置见[输出结构](输出结构.md#8-已完成100题1000题的实际迁移)，逐文件SHA和删除清单见[迁移记录](audit/stage_output_migration.json)。当前默认参数对应已完成历史实验；`--status`查看原结果，执行新配置须使用新PARAM_NAME，防止覆盖历史结果。
