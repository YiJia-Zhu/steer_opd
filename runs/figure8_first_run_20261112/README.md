> 2026-11-12最新状态：用户已停止实验，两组各10/50步与steer评测266/480保留。当前不续跑；同题Top16改动使用独立v2版本，不改写下方历史结果。

# 首轮OPD实验进度

已完成更新：{'vanilla': 10, 'steer': 10}；目标每组50步。

主评测关闭steering；论文SFT数字仅作近似参照。

| 组 | step | AIME avg@16 | 无hook Overlap@16 |
|---|---:|---:|---:|
| vanilla | 0 | 1.04% | 59.2932% |
| vanilla | 10 | 1.88% | 59.4920% |
| steer | 0 | 1.04% | 59.2932% |
| steer | 10 | 待评 | 59.4766% |

恢复：在项目根source scripts/runtime_env.sh后，运行

```bash
"$STEER_OPD_PYTHON" scripts/run_figure8_campaign.py --config configs/figure8_first_run.json
```

每次自动重新检查空闲GPU，优先单卡；不会终止其它进程。
