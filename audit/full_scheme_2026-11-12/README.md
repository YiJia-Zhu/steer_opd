# 完整面板实现与当前验收状态

状态快照：2026-11-13 02:02（Asia/Shanghai）。协议见[完整队列](../../docs/08_full_scheme_execution.md)。

现有Qwen配对的分支与队列代码已补充，尚无本轮GPU验收、激活提取、参数更新或效果结论。
precheck与完整监督器均存活且PID/start-ticks一致；precheck等待空闲GPU，完整监督器等待precheck。
8张GPU均有计算进程和显存驻留。未停止其它项目进程，当前队列串行使用1张真正空闲卡。
实时状态应读取run目录，不能把本快照当作之后的完成证明。

- CPU检查：128 passed、1 skipped、10 warnings，6.36秒；见`cpu_tests.log`。
- 79个Python文件AST检查通过；本快照重新核验79份源文件SHA256，均与测试清单一致，见`source_manifest.json`。
- 模型、tokenizer与冻结数据输入SHA256见`input_fingerprints.json`；哈希不等于GPU数值验收。
- GPU门控KV隔离、长前缀完整分布重放和临时反向检查仍未运行，不声明已通过。
- 通用包校验因旧归档缺失失败：`audit/docs_cleanup_2026-11-10/concurrent_source_snapshots/12_cross_family_research_contract.md`。
  见`package_validation.log`，未补造快照或跳过原哈希要求。
- 清单`panel_preview.json`仅为预览：55个潜在正式run、49种方法、上限合计3400次更新，另最多9个10步选型试验。
  真实参数和清单在实际选参后冻结，不表示这些实验已完成。
- EasySteer精确在线教师交互未接入；当前门控使用项目批量HF实现。本地SFT和跨tokenizer训练扩展未启动。

存储：整盘约3.4TiB可用；项目路径的`statvfs/df`仅约116GiB可用（本次快照值）。
两路径解析到相同目录，同一XFS挂载启用`prjquota`，目录project ID21003且继承P属性。
直接读取XFS配额管理字段被系统拒绝，未声称获得其hard/soft明细。可写额度以项目路径实测为准。
每GPU阶段保留40GiB余量，不足时等待；不改系统配额、不删旧run、不向项目外写新结果。

机器可读证据：`validation_summary.json`、`runtime_status.json`、`storage_quota_latest.json`。
旧的`storage_quota.json`保留首次核对时点，剩余空间随共享文件系统使用情况变化。

恢复命令：

```bash
source scripts/runtime_env.sh
"$STEER_OPD_PYTHON" scripts/run_full_scheme.py --execute
```

当前监督器已经运行，无需重复执行；被中断后再使用同一命令。
