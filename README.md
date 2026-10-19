# SMILES-Morgan-HSCL

用于 LC–MS 保留时间预测的研究源码：SMILES/Morgan 双分支、监督对比学习、PureGIN 及决策融合。

这是 **2026-10-19 整理的源码包**。包含模型、训练、数据处理、评估、消融、泛化和本地预测界面。实验数据、模型权重、预训练文件、逐样本预测及历史运行输出需要另行取得；仅克隆本仓库无法重现已归档数值。

## 版本与结果口径

| 代码路径 | 用途 | 已归档结果（不是本次重新训练） |
| --- | --- | --- |
| `code/src/` | SMILES–Morgan 编码、融合和 SupCR 损失 | 单预测器结果应按对应配置核对 |
| `code/train_pure_gnn_for_ensemble.py`、`scripts/run_final_fusion_ccrl_gnn.py` | Scheme A：CCRL 集成 + 三个 PureGIN 决策融合 | SMRT test MAE 25.8893 s，RMSE 52.2262 s，R² 0.9124 |
| `experiments/rmcl_supplement/` | 2026-10-15 单预测器 CMA/RSCL 补实验源码 | 主单预测器 seed1234 归档 MAE 31.0575 s；补实验任务不代表已完成训练 |

Scheme A 固定定义：`0.30 * pred_ensemble + 0.24 * g2026 + 0.18 * g3407 + 0.28 * g777`。
残差校正保留为辅助分析，不能替换该公式中的 `pred_ensemble`。RMCL 单预测器和 Scheme A 集成的性能不能混用。详见 `docs/` 和 `configs/scheme_a/`。

## 安装

推荐 Python 3.10–3.12。先创建并启用虚拟环境，在该环境内安装适合硬件的 PyTorch（历史环境记录见下文），再安装其余依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# 在已激活环境内先安装适合硬件的 PyTorch
python -m pip install -r requirements.txt
```

`requirements.txt` 来自补实验记录，未声明跨环境逐位一致。旧 Scheme A 环境记录为 Python 3.9.25、torch 2.3.0+cu118、RDKit 2022.09.5；补实验服务器记录为 torch 2.7.1+cu128。不同 RDKit 随机枚举及运行环境可能影响结果。`requirements-optional.txt` 仅用于引用 PyG 的基线骨架。

## 无需机器学习依赖的快速检查

```bash
python -B tools/verify_source_package.py
python -B code/evaluate_rt_predictions.py --predictions examples/predictions_demo.csv --method synthetic-demo
```

示例只有三条人工构造的 RT 数值，不含真实分子或实验预测。预期 MAE=20 s，RMSE≈21.6025 s，R²=0.93；用于检查评估入口，不能作为科学结果。

## 数据与模型文件

| 本地路径 | 需要的材料 |
| --- | --- |
| `data/unified/train.csv`、`val.csv`、`test.csv` | 原固定划分；字段为 `sample_id,smiles,rt,split`，RT 单位为秒 |
| `pretrained/ChemBERTa-zinc-base-v1/` | 具有 config、tokenizer 和模型权重的 Hugging Face 兼容本地目录；原模型标识为 `seyonec/ChemBERTa-zinc-base-v1` |
| `checkpoints/` | 与运行配置匹配的 CCRL 和 PureGIN checkpoint |
| `outputs/` | 若重算历史融合，需原 CCRL/PureGIN validation/test 预测和配置 |

SMILES 与 Morgan 必须共享同一行的分子、标签和划分。不能为两个模态单独随机切分。原数据和第三方模型的取得、许可与引用由其来源规定；本源码包不重新分发这些文件。原复现包的材料清单保留在 `docs/original_reproduction_README.md`，其中提到的权重和预测不在此 GitHub 源码包中。

## 双分支训练入口

在仓库根目录补齐数据及预训练模型后，可运行一次前向和损失检查：

```bash
python code/src/train_multimodal.py --smiles-backbone custom --smiles-model pretrained/ChemBERTa-zinc-base-v1 --smoke-test
```

移除 `--smoke-test` 会启动训练。所有训练参数可通过 `--help` 查看；默认参数不等于所有归档实验的精确配置。单模态入口为 `code/src/train_smiles_only.py` 和 `code/src/train_morgan_only.py`。源码副本已将这两个训练器的项目根路径修正到仓库根目录；原工作目录中的源码未改动。

## PureGIN 与融合

```bash
python code/train_pure_gnn_for_ensemble.py --seed 2026 --output-dir outputs/pure_gin_seed2026
```

其他两个种子为 3407、777。完整历史 Scheme A 还需要匹配的 CCRL 集成输入。`scripts/run_final_fusion_ccrl_gnn.py` 读取 `outputs/top6_pool_target_power_ensemble/predictions_{val,test}_calibrated.csv` 中的 `pred_ensemble`，以及 `outputs/pure_gin_seed*/predictions_{val,test}_gnn.csv`。该脚本会重新进行验证集权重搜索并写出结果，执行前应保留原输出副本。`code/verify_reproduction.py` 适用于补齐全部原材料后的完整复现目录，会检查缺失的 checkpoint 和预测文件；源码包完整性使用 `tools/verify_source_package.py`。

## 补实验、泛化和预测界面

- `experiments/rmcl_supplement/README.md`：单预测器补实验源码所需的外部输入。
- `code/generalization_*.py`、`code/prepare_predret_*.py`、`code/predret_*.py`：冻结预测、scaffold 和 PredRet 协议。目标系统监督微调须与零样本预测分开报告。
- `code/literature_reimplementations/`：项目内基线或骨架，文件名带 skeleton 的实现不是完整文献复现。
- `rt_dashboard/`：本地预测界面；补齐权重后执行 `python -m rt_dashboard.server`，也可在 Windows 运行 `run_rt_dashboard.bat`。

部分历史脚本依赖原输出树、另行取得的文献工程或历史配置，不能在仅有源码的克隆中直接运行。已省略三份绑定旧机器绝对路径的启动/汇总脚本、文稿生成器和第三方代码副本。整理范围与改动哈希见 `SOURCE_INVENTORY.json`。

## 校验、引用与权利

`SOURCE_MANIFEST.json` 记录包内文件的 SHA-256；`tools/verify_source_package.py` 校验文件清单、哈希及 Python 语法。此次整理未重新训练模型。完整科学复现必须使用原数据划分、配置、权重与对应环境。

仓库尚未指定开源许可证；公开可见不等于授予额外使用许可。作者、论文 DOI 和正式引用信息应在确认后补充。第三方依赖和预训练模型保留各自许可；见 `THIRD_PARTY_NOTICES.md`。
