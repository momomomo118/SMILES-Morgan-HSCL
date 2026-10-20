# -*- coding: utf-8 -*-
"""Build a source-only GitHub snapshot without changing experimental inputs."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'deliverables/github/SMILES-Morgan-HSCL'
RECORDS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(relative, text):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((text.rstrip().replace('\r\n', '\n') + '\n').encode('utf-8'))


def copy(source, relative=None, transform=None):
    relative = relative or source.relative_to(ROOT).as_posix()
    data = source.read_bytes()
    result = (transform(data.decode('utf-8-sig')).encode('utf-8') if transform else data).replace(b'\r\n', b'\n')
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result)
    RECORDS.append(dict(path=relative, source=source.relative_to(ROOT).as_posix(),
                        source_sha256=digest(data), sha256=digest(result),
                        change='portable paths; LF line endings' if transform else ('LF line endings' if result != data else 'verbatim')))


def portable_training(text):
    text = text.replace('PROJECT_ROOT = Path(__file__).resolve().parents[1]',
                        'PROJECT_ROOT = Path(__file__).resolve().parents[2]')
    if 'SMILES_BACKBONE_PRESETS = {' in text:
        start = text.index('SMILES_BACKBONE_PRESETS = {')
        stop = text.index('\n}', start) + 2
        text = text[:start] + '''SMILES_BACKBONE_PRESETS = {
    "chemberta_zinc": PROJECT_ROOT / "pretrained" / "ChemBERTa-zinc-base-v1",
    "chemberta_10m_mtr": PROJECT_ROOT / "pretrained" / "ChemBERTa-10M-MTR_pretrained",
    "chemberta_77m_mtr": PROJECT_ROOT / "pretrained" / "ChemBERTa-77M-MTR_pretrained",
}''' + text[stop:]
    else:
        text = text.replace('PROJECT_ROOT\n            / "smiles_baseline"\n            / "RT-TR-main"\n            / "RT-TR-main"\n            / "seyonec"',
                            'PROJECT_ROOT\n            / "pretrained"')
    return text


def main():
    if DEST.exists() and '--refresh' not in sys.argv:
        raise SystemExit(f'Refusing to overwrite existing package: {DEST}')
    if DEST.exists():
        old = json.loads((DEST / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        for item in old['files']:
            if digest((DEST / item['path']).read_bytes()) != item['sha256']:
                raise SystemExit('Refusing to overwrite a modified draft: ' + item['path'])
    DEST.mkdir(parents=True, exist_ok=True)
    excluded = {'run_lowcost_ablation.py', 'make_smrt_literature_comparison.py',
                'run_fast60_local_tuning.ps1'}
    for p in sorted((ROOT / 'code').rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in excluded:
            continue
        if p.suffix not in {'.py', '.sh', '.txt'}:
            continue
        transform = portable_training if p.parent == ROOT / 'code/src' and p.name in {'train_multimodal.py', 'train_single_modal.py'} else None
        if p.name == 'generalization_frozen_20260926.py':
            transform = lambda text: re.sub(r"default='[A-Z]:/Users/[^']+/envs/torch_clean/python.exe'", 'default=sys.executable', text)
        copy(p, transform=transform)
    script_names = [
        'run_final_fusion_ccrl_gnn.py', 'run_final_fusion_five_branch.py',
        'run_strict_pipeline_ablation.py', 'paper_ablation_runner_20260825.py',
        'paper_ablation_audit_20260825.py', 'audit_simplified_mcrl_20261011.py',
        'evaluate_simplified_mcrl_20261011.py', 'report_simplified_mcrl_20261011.py',
        'plot_final_prediction_scatter.py', 'generate_signed_error_histogram.py',
        'probe_runtime_env.py',
    ]
    for name in script_names:
        transform = (lambda text: re.sub(r'& [A-Z]:/Users/[^ ]+/python.exe', 'python', text)) if name == 'report_simplified_mcrl_20261011.py' else None
        copy(ROOT / 'scripts' / name, transform=transform)
    for p in sorted((ROOT / 'rt_dashboard').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.md', '.js', '.css', '.html'}:
            copy(p)
    copy(ROOT / 'run_rt_dashboard.bat', transform=lambda text: '@echo off\nsetlocal\ncd /d "%~dp0"\npython -m rt_dashboard.server --open --warmup\nendlocal\n')
    for name in ['model_mechanism.md', 'predret_condition_workflow.md']:
        copy(ROOT / 'docs' / name)
    copy(ROOT / 'README.md', 'docs/original_reproduction_README.md')
    copy(ROOT / 'FINAL_RELEASE_SCHEME_A/model_definition.md', 'docs/scheme_a_model_definition.md')
    for name in ['metrics_final.json', 'weights.json']:
        copy(ROOT / 'FINAL_RELEASE_SCHEME_A/metrics' / name, 'configs/scheme_a/' + name)
    copy(ROOT / 'data/unified/README.md')
    supplement = ROOT / 'deliverables/RMCL_supplement_server_20261015'
    for p in sorted(supplement.rglob('*')):
        rel = p.relative_to(supplement)
        if not p.is_file() or '__pycache__' in rel.parts:
            continue
        if rel.parts[0] in {'code', 'scripts', 'tests'} and p.suffix == '.py':
            copy(p, 'experiments/rmcl_supplement/' + rel.as_posix())
        elif len(rel.parts) == 1 and (p.suffix in {'.py', '.sh'} or p.name in {'protocol.json', 'requirements.txt', 'reference_index.json', 'experiment_matrix.csv'}):
            copy(p, 'experiments/rmcl_supplement/' + rel.as_posix())
    copy(Path(__file__), 'tools/build_source_package.py')

    write('.gitignore', '''# Credentials and local configuration
.env
.env.*
!.env.example
*.pem
*.key
# Python and editor files
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.idea/
.vscode/
# Data, checkpoints, pretrained models and generated results
data/**
!data/unified/
!data/unified/README.md
checkpoints/
pretrained/
outputs/
results/
logs/
work/
tmp/
experiments/rmcl_supplement/data/
**/checkpoints/
**/pretrained/
**/results/
**/outputs/
**/reference/
*.pt
*.pth
*.bin
*.safetensors
*.pkl
*.pickle
*.npz
*.npy
*.zip
*.log
''')
    write('.gitattributes', '* text=auto eol=lf')
    write('requirements.txt', '''# Python 3.10-3.12. Install a suitable PyTorch build separately.
# Versions recorded in the RMCL supplement; not a universal environment lock.
numpy==1.26.4
pandas==2.3.3
scipy==1.13.1
scikit-learn==1.6.1
transformers==4.40.1
tokenizers==0.19.1
safetensors==0.4.5
rdkit==2025.9.2
tqdm==4.67.1
matplotlib==3.9.4
''')
    write('requirements-optional.txt', '''# Graph baseline skeletons require a PyTorch-compatible PyG installation.
# Core PureGIN uses native PyTorch; PyG is not needed for that branch.
torch-geometric
''')
    write('examples/predictions_demo.csv', 'sample_id,true_rt,pred_rt\nsynthetic_1,100,110\nsynthetic_2,200,180\nsynthetic_3,300,330')
    write('README.md', '''# SMILES-Morgan-HSCL

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
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
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
''')
    write('experiments/rmcl_supplement/README.md', '''# RMCL 单预测器补实验源码

本目录保留 2026-10-15 服务器包的 Python、shell、协议和任务索引。这里不包含数据、离线 ChemBERTa、checkpoint、reference 预测、原完整包 manifest 或 source_inventory。

可直接查看任务（只使用标准库，不训练）：

```bash
cd experiments/rmcl_supplement
python -B run.py list
```

训练、preflight、analyze、representations 以及原测试中的数据/权重检查，需要先将原完整包中 `data/`、`pretrained/`、`reference/`、`checkpoints/`、`manifest.json` 和 `source_inventory.csv` 恢复到本目录。不要将这些大文件提交到 Git。

`run.py verify` 校验原完整包清单；本源码包请在仓库根目录使用 `python -B tools/verify_source_package.py`。

准备输入后，`run.py train --plan all` 定义四组、三个种子的十二项训练；`--plan missing` 按任务索引选择五项训练。CMA/RSCL 的定义、种子和模型选择规则见 `protocol.json`。
''')
    write('THIRD_PARTY_NOTICES.md', '''# Third-party components and input artifacts

This source snapshot does not vendor third-party literature repositories, pretrained weights, or datasets. Python dependencies retain their own licenses.

- ChemBERTa backbone identifier used by this project: `seyonec/ChemBERTa-zinc-base-v1`. Obtain a compatible local model/tokenizer from its upstream source and follow its terms.
- SMRT and PredRet: use the original dataset sources, citations and permissions; fixed project splits are separate input artifacts.
- PyTorch, Hugging Face Transformers, RDKit, NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm and optional PyTorch Geometric are external dependencies.
- Historical adapters that refer to `external_literature_code/` require those original projects to be obtained separately. Their code and licenses are not replaced by this repository.

No new license is assigned to the project's source by this packaging operation. Author-approved licensing and publication metadata remain to be supplied.
''')
    write('tools/verify_source_package.py', '''"""Verify the source snapshot using only the Python standard library."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise SystemExit('Integrity failure: ' + item['path'])
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=item['path'])
    expected = {x['path'] for x in manifest['files']} | {'SOURCE_MANIFEST.json'}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and not any(x in {'.git', '__pycache__'} for x in p.relative_to(ROOT).parts)}
    unexpected = actual - expected
    missing = expected - actual
    if missing or unexpected:
        raise SystemExit('File set differs: missing=' + str(sorted(missing)) + ', extra=' + str(sorted(unexpected)))
    print('Source package PASS:', len(manifest['files']), 'files; hashes and Python syntax checked.')


if __name__ == '__main__':
    main()
''')
    write('SOURCE_INVENTORY.json', json.dumps(dict(
        source_date='2026-10-19', scope='source-only; no datasets, weights or experimental outputs',
        omitted_legacy_scripts=sorted(excluded), files=RECORDS), ensure_ascii=False, indent=2))
    files = []
    for p in sorted(DEST.rglob('*')):
        if not p.is_file() or p.name == 'SOURCE_MANIFEST.json' or any(x in {'.git', '__pycache__'} for x in p.relative_to(DEST).parts):
            continue
        data = p.read_bytes()
        if p.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(p))
        files.append(dict(path=p.relative_to(DEST).as_posix(), bytes=len(data), sha256=digest(data)))
    write('SOURCE_MANIFEST.json', json.dumps(dict(format=1, files=files), ensure_ascii=False, indent=2))
    print(json.dumps(dict(directory=str(DEST), files=len(files)+1,
                         bytes=sum(x['bytes'] for x in files)), ensure_ascii=False))


if __name__ == '__main__':
    main()
