# -*- coding: utf-8 -*-
"""Validate and report completed inference/reuse experiments, without training."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/generalization_20260926'
FROZEN=BASE/'predret_frozen_verified'
REUSE=BASE/'scaffold_reuse'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def table(df):
    def fmt(x):return f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                      ['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    complete=json.loads((FROZEN/'completion.json').read_text())
    assert complete['status']=='complete' and complete['source_reproduction_passed']
    protocol=json.loads((FROZEN/'frozen_protocol.json').read_text(encoding='utf-8'))
    reuse_protocol=json.loads((REUSE/'protocol.json').read_text(encoding='utf-8'))
    for p,h in {**protocol['input_sha256'],**reuse_protocol['inputs']}.items():assert digest(ROOT/p)==h,p
    d=pd.read_csv(FROZEN/'predret_predictions.csv')
    raw=pd.read_csv(ROOT/'data/external_predret/transfer_10_systems_cleaned/all_cleaned.csv')
    assert len(d)==2197 and d.sample_id.is_unique and set(d.sample_id)==set(raw.sample_id)
    calc=.30*(.30*d.pred_exact1234+.25*d.pred_dropout015+.10*d.pred_lowfusion+.20*d.pred_aug050+.15*d.pred_exact2025)+.24*d.pred_gin2026+.18*d.pred_gin3407+.28*d.pred_gin777
    assert np.allclose(calc,d.pred_MCRL_frozen,rtol=0,atol=1e-9)
    scores=pd.read_csv(FROZEN/'predret_metrics.csv')
    for system,part in d.groupby('system'):
        saved=scores[(scores.subset=='all')&(scores.system==system)&(scores.model=='MCRL_frozen')].iloc[0]
        assert abs(saved.mae_s-abs(part.pred_MCRL_frozen-part.rt).mean())<1e-8
    adapted_audits=[]
    for p in sorted((ROOT/'server_results/predret_compare_strict_20260909/proposed').glob('*/fold_*/metrics_full_ensemble.json')):
        m=json.loads(p.read_text(encoding='utf-8-sig'))
        assert m['blend_selection_split']=='inner_validation' and m['outer_test_used_for_weight_selection'] is False
        adapted_audits.append(dict(file=str(p.relative_to(ROOT)),sha256=digest(p)))
    assert len(adapted_audits)==100
    (BASE/'validation_checks.json').write_text(json.dumps(dict(status='passed',n_predictions=len(d),
        n_systems=d.system.nunique(),fixed_fusion_formula_passed=True,original_input_hashes_unchanged=True,
        source_reproduction=json.loads((FROZEN/'source_reproduction_checks.json').read_text()),
        adapted_validation_only_audits=adapted_audits,new_neural_training_epochs=0,
        zero_shot_target_fit=False,scaffold_optimized_on='validation only'),indent=2),encoding='utf-8')
    per=scores[(scores.subset=='all')&(scores.system!='POOLED')]
    pivot=per.pivot(index='system',columns='model',values='mae_s')
    per_table=pivot[['MCRL_frozen','MCRL_target_adapted_OOF','SMRT_training_median']].reset_index()
    per_table.columns=['System','Frozen MCRL MAE (s)','Adapted OOF MAE (s)','Source-median MAE (s)']
    frozen=per[per.model=='MCRL_frozen'][['system','n','mae_s','rmse_s','r2','spearman']]
    frozen.to_csv(BASE/'Table_PredRet_frozen_generalization.csv',index=False,encoding='utf-8-sig')
    per_table.to_csv(BASE/'Table_PredRet_zero_shot_vs_adapted.csv',index=False,encoding='utf-8-sig')
    rm=pd.read_csv(REUSE/'metrics.csv')
    ret=rm[(rm.split=='test')&(rm.model=='single_CCI_configuration')][['partition','n','mae_s','rmse_s','r2','spearman']]
    ret.to_csv(BASE/'Table_retrospective_scaffold_generalization.csv',index=False,encoding='utf-8-sig')
    macro=pd.read_csv(FROZEN/'predret_system_macro_metrics.csv')
    fm=macro[(macro.subset=='all')&(macro.model=='MCRL_frozen')].iloc[0]
    am=macro[(macro.subset=='all')&(macro.model=='MCRL_target_adapted_OOF')].iloc[0]
    pooled=scores[(scores.subset=='all')&(scores.system=='POOLED')&(scores.model=='MCRL_frozen')].iloc[0]
    unseen=scores[(scores.subset=='unseen_smrt_all')&(scores.system=='POOLED')&(scores.model=='MCRL_frozen')].iloc[0]
    r=ret[ret.partition=='random'].iloc[0];s=ret[ret.partition=='scaffold'].iloc[0]
    ci=pd.read_csv(REUSE/'scaffold_cluster_bootstrap.csv')
    scaffold_ci=ci[(ci.partition=='scaffold')&(ci.model=='single_CCI_configuration')].iloc[0]
    wins=int((pivot.MCRL_frozen<pivot.SMRT_training_median).sum())
    text=f'''# 补充泛化实验报告（2026-09-26）

本次新增神经网络训练为 **0 轮**。复用已有 checkpoint、验证集/测试集预测和已完成的 PredRet 迁移结果；新增冻结推理、融合权重的验证集优化、重叠审计和置信区间计算。没有修改原始论文、checkpoint 或历史预测文件。先前准备的 Top6 scaffold 长训练没有启动。

命名约定：Multi-View Contrastive Representation Learning（MCRL）指完整框架；Cross-View Contrastive Interaction（CCI）指其中的跨视图交互模块。下文的“CCI-based 预测模型”指包含 CCI 和回归头的预测器，并非把 CCI 模块等同于完整 MCRL。表中的单 CCI 配置是 MCRL 的简化候选架构。

## 1. 完整 MCRL：SMRT → PredRet 冻结零样本评估

使用完整最终配置：CCI-based Top6 回归集成（6 个候选中 5 个非零权重）与 3 个 PureGIN 分支。源模型参数、目标变换和两级融合权重保持固定；没有使用任何 PredRet RT 标签训练、校准或选权。RT 单位为秒。

固定外层权重：CCI 集成 0.30；GIN2026 0.24；GIN3407 0.18；GIN777 0.28。CCI 集成内部权重为 exact1234 0.30、dropout015 0.25、lowfusion 0.10、aug050 0.20、exact2025 0.15、exact777 0。没有残差校准。

评估覆盖 {len(d):,} 条记录、10 个系统。按 canonical isomeric SMILES 审计，{int(d.seen_smrt_train.sum())} 条与 SMRT train 重叠、{int(d.seen_smrt_val.sum())} 条与 val 重叠、{int(d.seen_smrt_test.sum())} 条与 test 重叠。排除全部 SMRT 数据中出现的分子后，还剩 {complete['unseen_all']:,} 条；另提供仅排除 train+val 的子集（{complete['unseen_development']:,} 条）。这里的“未见”针对所检查的 SMRT 数据，不能排除 ChemBERTa 无监督预训练语料中的分子。

{table(per_table)}

完整冻结模型的系统宏平均 MAE 为 **{fm.mae_s:.3f} s**，全部样本 pooled MAE 为 **{pooled.mae_s:.3f} s**；排除全部 SMRT 重叠分子后的 pooled MAE 为 **{unseen.mae_s:.3f} s**。相对固定 SMRT 训练集 RT 中位数基线，冻结模型在 {wins}/10 个系统上 MAE 更低。系统宏平均与 pooled 指标的样本权重不同，不可混用。

右侧 adapted 列复用已有 100 个外层测试折的 OOF 预测，系统宏平均 MAE 为 {am.mae_s:.3f} s。该配置使用目标系统训练标签、内层验证选权，并包含不同的候选集合和 RDKit 预测分支，因此这两列用于展示“冻结部署”和“目标系统适配”两种场景，**不能将差值全部归因于迁移训练或 CCI 模块**。这里统一重算 pooled OOF 指标，可能与历史“10 折指标平均”略有差异。

完整 MAE、RMSE、R²、Spearman 及两个无重叠子集见 `predret_frozen_verified/predret_metrics.csv`。各系统 MAE 的 95% bootstrap 区间见 `frozen_mae_bootstrap_ci.csv`；对已固定的模型进行样本重采样，未涵盖重新训练产生的方差。

## 2. 复用实验：同结构的随机划分 / scaffold 对照

两边均采用 **1 个 seed 2026 的 CCI-based RT 预测模型 + GIN2026/3407/777**。独立使用各自验证集优化非负、和为 1 的融合权重，再固定权重计算测试结果。模型 tensor 形状以及记录的主要超参数已逐项核对。此配置不是论文最终 Top6 MCRL 配置。

{table(ret)}

scaffold 相对随机划分的 MAE 增加 {s.mae_s-r.mae_s:.3f} s（{100*(s.mae_s/r.mae_s-1):.2f}%）。scaffold 测试集包含 78 个无手性 Bemis–Murcko 骨架，按骨架分组 bootstrap 的 MAE 95% 区间为 [{scaffold_ci.mae_ci_low_s:.3f}, {scaffold_ci.mae_ci_high_s:.3f}] s。

train / val / test 的样本 ID、分子、无手性骨架交集均为 0；划分包含同一份完整 SMRT 样本。详细结果和每条预测见 `scaffold_reuse/`。

**解释限制：**随机划分的学习率调度 patience 为 2，scaffold 为 3；scaffold 从自身 epoch-11 最优 checkpoint 续训时重置了优化器，续训日志在第 24 个已完成 epoch 后中断。两者也未证实采用完全一致的历史特征环境。因此只能作为同结构的回顾性泛化对照，不能声称训练条件和收敛预算完全受控，也不能单独证明 CCI 的因果收益。没有把原来的 25.8893 s（最终 Top6 配置）直接当作这组单 CCI 配置的随机对照。

## 3. 为保证复用可靠性做的兼容性核验

本机旧环境不能直接复现所有历史预测。CCI 指纹使用隔离的 RDKit 2025.09.2 后通过源域检查；GIN 图/描述符使用隔离的 RDKit 2026.03.3 原生函数后通过检查。GIN 差异定位到部分取代芳香氮的氢键受体计数。没有通过改变分子特征去优化 PredRet 测试误差，也没有对历史预测加偏移补偿。

所有 8 个非零权重分支分别对固定 64 条 SMRT 源域样本进行复现核验，容差预先设为最大绝对差 < 0.05 s。检查结果见 `predret_frozen_verified/source_reproduction_checks.json`。这是对现有归档的兼容性验证，并不证明原训练时使用的具体 RDKit 版本。原有 Python 环境未修改，新依赖只放在本项目 `work/` 下。

早期未通过核验的输出已收纳至 `diagnostics/`，仅为诊断证据，**不可作为正式实验结果引用**。本次正式冻结结果只使用 `predret_frozen_verified/`。

## 4. 论文可用的英文描述

**Protocol.** To assess cross-system generalization without target-domain adaptation, we froze the SMRT-trained MCRL checkpoints, source-domain target transformation, and both levels of ensemble weights, and evaluated the resulting predictor on 2,197 records from 10 PredRet systems. No target-system retention times were used for training, calibration, model selection, or weight optimization. Results were additionally reported after excluding compounds present in the SMRT development data and in the complete SMRT dataset. Target-adapted out-of-fold results were reported separately because their data access and candidate ensemble differed from the frozen configuration.

**Results.** Frozen MCRL achieved a system-macro MAE of {fm.mae_s:.3f} s and a pooled MAE of {pooled.mae_s:.3f} s across the 10 systems. The pooled MAE was {unseen.mae_s:.3f} s after excluding all compounds shared with SMRT. These measurements characterize the limits of direct cross-system deployment and should be distinguished from results obtained with target-system adaptation.

**Retrospective scaffold analysis.** Using the available single-CCI predictor and three PureGIN branches on each split, validation-only decision fusion yielded test MAEs of {r.mae_s:.3f} s for the random split and {s.mae_s:.3f} s for the scaffold-disjoint split. This retrospective comparison matches the candidate architecture, but learning-rate scheduling and the scaffold training continuation differed; it is therefore not a fully controlled, equal-budget comparison of the final Top6 MCRL configuration.

## 5. 复现与文件

- 冻结评估脚本：`code/generalization_frozen_20260926.py`。
- 独立 GIN 特征进程：`code/generalization_gin_features_20260926.py`，只接收 sample_id 与 SMILES，不接收 RT 标签。
- 历史结果复用：`code/generalization_reuse_20260926.py`。
- 报告与结果校验：`code/report_generalization_20260926.py`。
- 正式协议、输入 SHA-256、分支核验、原始预测、指标均保存在对应结果子目录。
- 完整 Top6 scaffold 重训脚本仅处于准备状态；根据“最小训练量”的要求，未执行。

本轮产出足以新增“冻结跨系统评估”和“同结构回顾性 scaffold 分析”。严格同预算的完整 Top6 scaffold 对照仍没有被完成，不能在论文中写成已完成。
'''
    (BASE/'补充泛化实验报告.md').write_text(text,encoding='utf-8')
    # Optional publication-oriented static figures; data tables remain canonical.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    systems=per_table.System.tolist();y=np.arange(len(systems))
    ax.hlines(y,per_table['Adapted OOF MAE (s)'],per_table['Frozen MCRL MAE (s)'],color='#b8c1cc',lw=2,zorder=1)
    ax.scatter(per_table['Frozen MCRL MAE (s)'],y,s=55,label='Frozen MCRL',zorder=3)
    ax.scatter(per_table['Adapted OOF MAE (s)'],y,s=55,label='Target-adapted configuration (OOF)',zorder=3)
    ax.set_yticks(y);ax.set_yticklabels(systems);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlim(10,1200);ax.set_xticks([10,30,100,300,1000]);ax.set_xticklabels(['10','30','100','300','1000'])
    ax.set_xlabel('MAE (seconds; logarithmic scale)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9);ax.grid(axis='x',alpha=.2)
    ax.set_title('PredRet: frozen deployment and target-system adaptation',pad=40)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(BASE/f'PredRet_generalization.{ext}',dpi=220)
    plt.close(fig)
    print('REPORT AND VALIDATION COMPLETE',BASE,flush=True)

if __name__=='__main__':main()
