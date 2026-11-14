# Qwen3.5-35B-A3B 下载排查与完成记录

2026-11-14（Asia/Shanghai）。目标目录：`/home/zhenning/data/huggingface_models/Qwen3.5-35B-A3B`。本次用户明确授权运行下载并排查，实验的 `execution_hold` 保持原样。

下载已完成。官方 revision 为 `59d61f3ce65a6d9863b86d2e96597125219dc754`，共 27 个文件、71,927,084,782 字节，包含 14 个权重分片。所有 LFS 文件通过官方 SHA-256 校验，其他文件通过 Git blob SHA-1 校验；权重索引引用的分片全部存在。随后 `hf download --dry-run` 返回需要下载 0 个文件。

开始排查时，只有 `model.safetensors-00013-of-00014.safetensors` 未完成，大小为 5,367,839,544 字节。其余 26 个文件都已通过内容哈希校验。磁盘当时约剩 46 GiB，无遗留下载进程，容量足够补齐这个分片。

证据与判断：

- 最新 Xet 日志（11:51 开始）记录了 3 次 `IncompleteMessage` 连接中断，目标为 `us.aws.cdn.hf.co`。前一晚的日志还记录了 5 次 HTTP 403；对应失败请求中的签名链接在请求失败时已经过期。
- Xet 配置显示读取超时为 300 秒、最多尝试 5 次，自适应下载并发上限为 64。长时间等待和重试与进度停滞现象相符，但现有证据不能进一步确认断连来自代理、上游还是客户端，也不能证明并发是原因。日志里的 `no_retry_on_429: false` 是配置，不代表发生了 429；`Reached end of reconstruction` 的 416 也未作为卡住证据。
- 经当前代理访问 HF 模型 API 成功，约 0.41 秒；不经代理直连 HF 主站在 8 秒后连接超时。普通 HTTP 对缺失分片的 1 MiB Range 请求成功返回 206。因此保留现有代理，仅对本次下载进程禁用 Xet。
- 本机 `huggingface_hub 1.23.0` 的 `_download_to_tmp_and_move` 为每次尝试创建新的 UUID 临时文件。重启下载会复用已完成文件，但不会接着旧 `.incomplete` 文件的位置下载。该结论来自当前安装源码，不能泛化到所有版本。

实际使用的下载命令：

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  /home/zhenning/data/miniconda3/bin/hf download Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --local-dir /home/zhenning/data/huggingface_models/Qwen3.5-35B-A3B \
  --max-workers 1
```

缺失分片于 12:11:19 至 12:15:06 下载完成，命令耗时 227.227 秒，平均约 23.6 MB/s，退出码为 0。下载期间持续有进度，未记录到 HTTP 重试。最后分片重新计算 SHA-256；其他文件复用了本次排查中计算的哈希，并确认 inode、大小、mtime 未变。

原有 5 个遗留 `.incomplete` 文件合计 8,434,995,709 字节，仍保留在目标目录的下载缓存中；它们不属于上述 27 个完整文件。本次没有修改共享 Python 环境、代理配置或实验配置，没有加载模型或执行 GPU 任务。

留档：

- `remote_manifest.json`：官方文件大小、内容哈希与 revision。
- `xet_diagnostic_summary.json`：旧日志的状态码、失败记录和签名过期时间；URL 查询参数已去除。
- `diagnosis.json`、`http_probe.json`：初始目录情况与连接测试。
- `http_download_status.json`、`http_download.log`：实际命令、进程、时间与退出码。
- `initial_integrity.json`、`final_integrity.json`：逐文件校验结果。
- `post_download_dry_run.log`：HF CLI 确认无待下载文件。
