# RMCL 补实验服务器包

本包用于补齐单个 SMILES–Morgan CCI 预测器的消融和表征证据。打包阶段未训练模型，也未生成新实验图片。原项目文件保持不变。

## 两个压缩包

- `RMCL_supplement_server_20261015_core.zip`：代码、SMRT 数据、离线 ChemBERTa、已有预测、实验配置、审计和操作说明。可用于新增训练。
- `RMCL_supplement_server_20261015_checkpoints.zip`：C/D/E 七个已有消融 checkpoint，以及主单预测器 exact1234。表征分析需要此包。

两个 ZIP 解压到同一个父目录，会合并为同一个 `RMCL_supplement_server_20261015` 文件夹。不要互相覆盖解压目录。两个包中的文件职责不重叠。需要约 6 GB 保存解压输入；建议至少再预留 15 GB 给新训练和导出的结果。

## 先明确实验定义

- CMA 指历史 `cross_attention_residual` 配置：SMILES query、Morgan key/value，加原始分支特征拼接。它不是独立的投影对齐损失。
- RSCL 指跨视图 SupCR、相似性项，以及融合表征 SupCR 的联合目标。使用连续 RT 排序、原始 ratio 形式，非固定阈值分类。
- Base 的预测头输入为 SMILES 与 Morgan 拼接；CMA 配置的预测头还接收跨视图结果。因此该比较衡量融合配置整体，不能声称隔离了注意力机制或模型容量。
- 历史 concat 模型没有融合投影头。本包的 F 类 Base + RSCL 在原拼接特征上增加仅服务于损失的投影头，从而保留两项 SupCR。它不会改变 Base 的回归前向路径。实现与历史代码分开，并有合成测试检查这一契约。

## 默认待补任务

| 变体 | seed 2026 | seed 777 | seed 3407 |
|---|---|---|---|
| Base，无 CMA/RSCL | 已有 C | 新训练 C | 新训练 C |
| Base + CMA | 已有 D | 已有 D | 已有 D |
| Base + RSCL | 新训练 F | 新训练 F | 新训练 F |
| Full，CMA + RSCL | 已有 E | 已有 E | 已有 E |

默认 5 次新训练。主成绩 exact1234=31.0575 s 不混入上述受控种子表。

注意：历史运行的环境、恢复方式和 RDKit 随机枚举状态并非完全一致。新脚本按 epoch 固定 RDKit RNG，保存优化器、scheduler 和其余 RNG 以便续跑。历史补齐模式是透明的回顾性比较，不应称为环境完全一致的严格重跑。如果论文要求同一环境下的完整四组实验，应使用 `--plan all` 从头完成 12 次训练，不复用历史指标。

## 服务器操作

以下命令在解压后的包目录执行。只运行到你需要的步骤；解压、校验、查看任务不会开始训练。

```bash
cd RMCL_supplement_server_20261015
python3 -B run.py verify --weights
```

推荐 Python 3.10–3.12，使用服务器已经能正常使用 GPU 的 PyTorch 环境。此前服务器记录为 PyTorch 2.7.1+cu128；本包不会自动替换 PyTorch。依赖安装需要服务器联网；预训练模型已在包内，训练时不需要下载模型。

```bash
bash setup.sh
.venv/bin/python -B run.py list
.venv/bin/python -B run.py preflight
```

先用单个任务验证训练速度；这条命令会真正启动训练：

```bash
.venv/bin/python -u -B run.py train --variants rscl --seeds 2026
```

补齐默认五次训练并继续表征分析，可在 tmux 中执行：

```bash
tmux new -s rmcl
bash start_missing.sh
```

`Ctrl+B` 后按 `D` 可离开 tmux；重新进入使用 `tmux attach -t rmcl`。中断后执行原命令即可从最后完整 epoch 续跑。强制杀进程可能留下 `worker.lock`，必须确认对应任务已经退出，再手动移除该任务目录中的锁文件。不要在同一任务上并发运行两个进程。

同环境完整重跑选项：

```bash
.venv/bin/python -u -B run.py train --plan all
```

`--plan all` 会在 `results/runs` 中创建自己的结果，不修改 `reference` 或历史 checkpoint。已完成的新任务会跳过。若更改协议、依赖或代码，请用另一份解压目录，避免混合运行。

多 GPU 可以按任务划分，例如两张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -B run.py train --variants base
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -B run.py train --variants rscl
```

## 表征与结果整理

```bash
.venv/bin/python -B run.py analyze
.venv/bin/python -B run.py representations --samples 1500 --batch-size 16
.venv/bin/python -B pack_results.py
```

表征命令会读取已有/新训练 checkpoint 做推理，输出固定分子子集的嵌入、原空间定量指标及两类 t-SNE 图。样本选择不使用 RT；RT 仅用于上色和分析指标。默认覆盖三个种子。

Figure 1 比较同一 CMA 模型的 SMILES pooled 与 cross-view pooled；Figure 2 比较 D/E 对应种子的融合表征。t-SNE 面板分别拟合，不能跨面板比较坐标、簇间距离或把图形外观当成因果证据。定量指标包括嵌入距离与 RT 距离的 Spearman 相关、近邻 RT 差异、投影匹配相似性/检索率及特征方差。Base 的未训练投影不应解释为已对齐空间。保存结果无论正向还是负向都必须报告。

CPU 数据分析不需要 checkpoint 包；表征分析需要。所有新结果写入 `results`。训练仅用 train/val 选模型，test 只在完成选择后评估。

`--epochs` 仅供用户主动诊断，输出目录带 `_diagnosticN`，不会自动进入正式结果汇总。包中不把短跑或合成测试作为科学实验结果。

## 强基线的证据限制

GCN、1D-CNN、GNN-RT 保存预测可以按分子身份核查。DeepGCN 的已有文件不通过当前划分核查，汇总脚本会拒绝将其严格排名。Graphormer 只有历史指标而缺少所指预测文件。归档 Transformer–LSTM 只有 RT 与预测，缺少分子身份，不能凭行数补造 ID。

项目当前没有恢复该 26.2352 s Transformer–LSTM 结果的完整原始训练工程，因此本包不声称可以精确重跑它。包中有单 SMILES 分支代码和已有材料，可用于明确命名的项目内基线，但不能冒充那项历史强基线。要复现原强基线，仍需找回原训练代码、配置和带身份的测试输出。

## 泛化结果无需重训

`reference/generalization` 和 `reference/predret` 保存单预测器材料，`run.py analyze` 从逐样本数据重算。Scaffold 使用已完成 seed1234；PredRet 分开冻结预测与目标系统微调。监督微调不是零样本泛化。十折 MAE 的均值与样本 SD 不等于 pooled MAE 或置信区间。外部 2197 条中有两条分子在 SMRT 训练中出现，来源记录保留在协议中。

## 溯源与核查

- `source_inventory.csv`：原始绝对路径、包内路径、来源哈希与派生列说明。
- `reference_index.json`：已有消融和 checkpoint 的对应。
- `protocol.json`：模块定义、统计口径、环境与限制。
- `manifest.json`：两个包内文件的 SHA-256。
- `validation`：打包时的静态、数据与合成契约检查；不是新增训练结果。

本包不含 PureGIN checkpoint、PTDF 或 Top6 集成执行流程；不会修改原项目。新模型的当前真实性能、RSCL 非稳定收益、基线缺口和表征证据缺口均不能通过命名替换消除。
