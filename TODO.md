# TODO

当前实验等待用户下一步明确同意，不运行GPU、模型生成/评分或训练。固定T=35B-A3B、Ref=0.8B、S=Qwen3-4B，Slerp0.1/层26，8方案+基线、0更新。

- [x] 同时实现overlap（Rethinking跨family Top16适配）与P_overlap（全概率下一字节sum(min)），缓存完整概率并校验投影身份。
- [x] 两种指标分别记录源/候选/delta、题目等权与token加权汇总，自动输出前3条的两套PNG/SVG/CSV/PDF。
- [x] 接入EasySteer原生生成、S0指定层capture、T/Ref批量评分及固定mask前向，无人工seq上限；保留HF显式参考。
- [x] 使用上游steering配置指纹，补模型内容hash与离线缓存绑定，保留逐token激活/位置/mask。
- [x] 每请求seed=42，固定Python/NumPy/PyTorch/vLLM与子进程hash seed；保留代码/环境/权重/引擎身份。
- [x] 恢复改为整批阶段/完整候选提交：一次公开generate/capture连续运行，返回后保存和原子提交；未提交整批重算，已完成阶段/候选复用。移除逐请求检查点与自定义调度循环；CPU故障测试覆盖SIGTERM/SIGKILL及提交边界。
- [x] 实时tqdm/console.log、结构化事件/心跳、阶段起止/耗时/ETA与只读进度查看；保留后续训练需要的中间产物。
- [x] 清理旧运行结果、重复Markdown与历史审计快照，保留核心研究稿、必要输入/源码及删除清单。
- [x] 隔离运行环境已安装，CPU导入/Slerp注册/范数/API及固定fork文件hash检查通过；实际GPU与模型架构支持另验。
- [x] 再核对当前8方案队列的后端：生成、提取、T/Ref评分和候选评分均走EasySteer/vLLM；CPU处理、已有logits的PyTorch统计和显式HF参考单独说明。
- [x] 集成手动启动sh：开头设置物理GPU池/方案ID/超参数，--check预览、--run验收后执行或恢复、--status查看；指定池外GPU不使用，改参数须新RUN_NAME。
- [x] 精简为当前实验调用链：删除旧pilot/Figure8/full-scheme、OPD训练/动态门控、模型数据下载脚本与旧配置；裁掉共享文件中的无用训练逻辑，保留原始数据/权重、文献和当前验收测试。
- [x] 用户手动准备35B教师权重；已补项目链接，14分片结构/索引与固定config/tokenizer元数据检查通过。未运行下载；完整权重内容哈希在启动时记录。
- [ ] 用户同意后验收源概率/重放、HF与EasySteer的同层capture及方向、Slerp0/移除恢复、首预测/t−1/EOS/padding、逐题绑定和共同原生支持。
- [ ] EasySteer逐token实际扰动范数留档；当前明确relative_perturbation_available=false。
- [ ] 用户手动--run后先取得三角色GPU验收凭据，再做所选方案select/冻结confirm和双指标图，停供检查。默认8组和overlap指标，不扩成16组；本轮未实际运行GPU验收。
- [ ] 实测时间/显存/吞吐；提前不承诺提速倍数。

[当前配置](configs/static_dual_overlap.json) · [实验协议](docs/experiment.md) · [实现说明](docs/implementation.md)

完整OPD目标/梯度路径、撤去SV后独立能力、刷新/匹配控制、多seed/强基线、层/强度扩展与CompassOPD官方仓库源码复核均留后续，当前不自动执行。原Student Top16学习配方不因新增P_overlap自动改变。
