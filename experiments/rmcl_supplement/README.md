# RMCL 单预测器补实验源码

本目录保留 2026-10-15 服务器包的 Python、shell、协议和任务索引。这里不包含数据、离线 ChemBERTa、checkpoint、reference 预测、原完整包 manifest 或 source_inventory。

可直接查看任务（只使用标准库，不训练）：

```bash
cd experiments/rmcl_supplement
python -B run.py list
```

训练、preflight、analyze、representations 以及原测试中的数据/权重检查，需要先将原完整包中 `data/`、`pretrained/`、`reference/`、`checkpoints/`、`manifest.json` 和 `source_inventory.csv` 恢复到本目录。不要将这些大文件提交到 Git。

`run.py verify` 校验原完整包清单；本源码包请在仓库根目录使用 `python -B tools/verify_source_package.py`。原完整包操作说明保留为 `ORIGINAL_FULL_PACKAGE_README.md`，其中对两个 ZIP 及离线材料的描述只适用于原完整包。

补齐输入后，`run.py train --plan all` 将启动同环境十二次训练；`--plan missing` 使用历史结果补齐五次训练。当前任务清单不表示训练已完成。CMA/RSCL 的定义、历史环境限制和缺失基线证据请阅读 `protocol.json` 及原说明。
