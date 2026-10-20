# RMCL 单预测器补实验源码

本目录保留 2026-10-15 服务器包的 Python、shell、协议和任务索引。这里不包含数据、离线 ChemBERTa、checkpoint、reference 预测、原完整包 manifest 或 source_inventory。

可直接查看任务（只使用标准库，不训练）：

```bash
cd experiments/rmcl_supplement
python -B run.py list
```

训练、preflight、analyze、representations 以及原测试中的数据/权重检查，需要先将原完整包中 `data/`、`pretrained/`、`reference/`、`checkpoints/`、`manifest.json` 和 `source_inventory.csv` 恢复到本目录。不要将这些大文件提交到 Git。

`run.py verify` 校验原完整包清单；本源码包请在仓库根目录使用 `python -B tools/verify_source_package.py`。

准备输入后，`run.py train --plan all` 定义四组、三个种子的十二项训练；`--plan missing` 按任务索引选择五项训练。CMA/RSCL 的定义、种子和模型选择规则见 `protocol.json`。
