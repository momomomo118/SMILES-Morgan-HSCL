"""Build the evidence-based simplified MCRL report after evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/simplified_mcrl_audit_20261011'

def link(relative,label=None):
    return f'[{label or relative}]({(ROOT/relative).as_posix()})'

def main():
    current=pd.read_csv(OUT/'reevaluated_test_metrics.csv').set_index('model')
    historical=pd.read_csv(OUT/'historical_current_split_metrics.csv')
    meta=json.loads((OUT/'checkpoint_metadata.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((OUT/'audit_summary.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'historical_prediction_reproduction_checks.json').read_text(encoding='utf-8'))
    single=current.loc['target_power_exact_repro_seed1234']
    ensemble=current.loc['CCI Top6 ensemble without PTDF']
    full=current.loc['Full MCRL Scheme A']
    comparisons=[]
    for name,r in [('Simplified single CCI predictor',single),('CCI Top6 ensemble without PTDF',ensemble)]:
        record=dict(model=name)
        for key in ['mae_s','rmse_s','medae_s','mape_pct','r2','pearson']:
            record[key]=float(r[key]);record['full_'+key]=float(full[key]);record['delta_'+key]=float(r[key]-full[key]);record['relative_change_'+key+'_pct']=float((r[key]/full[key]-1)*100)
        comparisons.append(record)
    pd.DataFrame(comparisons).to_csv(OUT/'performance_changes.csv',index=False)
    # The user's rounded reference is retained as a separate comparison.
    rounded=[]
    for name,r in [('Simplified single CCI predictor',single),('CCI Top6 ensemble without PTDF',ensemble)]:
        for key,b in [('mae_s',25.89),('rmse_s',52.23),('r2',.9124)]:
            rounded.append(dict(model=name,metric=key,user_full_reference=b,simplified=float(r[key]),delta=float(r[key]-b),relative_change_pct=float((r[key]/b-1)*100)))
    pd.DataFrame(rounded).to_csv(OUT/'comparison_with_user_rounded_reference.csv',index=False)
    classification=[]
    for r in meta:
        cfg=r.get('args') or r.get('config') or {}
        cls='Other model / graph predictor / tensor cache'
        if r.get('class')=='B_candidate':
            enabled=cfg.get('supcr',bool(cfg.get('omega_align',0) or cfg.get('omega_fusion',0)))
            cls='B: CCI + RTCL + regression head' if enabled else 'B-near: cross-attention + head; no RTCL'
        if 'scaffold' in r['path'] or 'resume_test' in r['path']: scope='different split / scaffold / recovery test'
        elif cfg.get('test_data') and 'unified' in cfg['test_data']: scope='unified SMRT candidate'
        elif 'predret' in r['path'].lower(): scope='PredRet / external system / fold'
        else: scope='check source metadata'
        if 'metadata_error' in r: cls='metadata reader limitation; see error'
        classification.append(dict(path=r['path'],category=cls,split_scope=scope,epoch=r.get('epoch'),
          fusion=cfg.get('fusion'),omega_align=cfg.get('omega_align'),omega_fusion=cfg.get('omega_fusion'),
          metadata_error=r.get('metadata_error','')))
    pd.DataFrame(classification).to_csv(OUT/'checkpoint_classification.csv',index=False)
    val=historical[(historical.split=='val') & historical.path.str.contains('base_predictions/')].sort_values('mae_s')
    assert val.iloc[0]['column']=='pred_target_power_exact_repro_seed1234'
    def hrow(path,col,split='test'):
        return historical[(historical.path==path)&(historical.column==col)&(historical.split==split)].iloc[0]
    historical_summary=[]
    for family in ['A_chemberta_seed2026','B_chemberta_bilstm_seed2026','C_concat_no_supcr_seed2026','G_cross_attention_no_supcr_seed2026',
                   'D_residual_no_supcr_seed2026','D_residual_no_supcr_seed777','D_residual_no_supcr_seed3407',
                   'E_residual_supcr_seed2026','E_residual_supcr_seed777','E_residual_supcr_seed3407']:
        p=f'outputs/paper_ablation_20260825/runs/{family}/predictions_test.csv'
        r=hrow(p,'pred_rt');v=hrow(p.replace('_test.csv','_val.csv'),'pred_rt','val')
        historical_summary.append(dict(model=family,path=p,validation_mae_s=float(v.mae_s),**{k:float(r[k]) for k in ['mae_s','rmse_s','medae_s','mape_pct','r2','pearson']}))
    pd.DataFrame(historical_summary).to_csv(OUT/'historical_ablation_metrics.csv',index=False)
    d=pd.DataFrame(historical_summary)
    erows=d[d.model.str.startswith('E_')]
    drows=d[d.model.str.startswith('D_')]
    mae_delta=float(single.mae_s-full.mae_s);ens_delta=float(ensemble.mae_s-full.mae_s)
    text=['## Simplified MCRL performance estimation','',
      '审计日期：2026-10-11（Asia/Shanghai）。本次没有训练、微调或重新选择集成权重；所有结果写入新的审计目录。',
      '',f'**核心结论：已找到完全匹配的 B 类 checkpoint。单个 CCI 预测器在当前 test 上实测 MAE = {single.mae_s:.4f} s；保留原有 Top6 CCI 集成而仅去除 PTDF 时，MAE = {ensemble.mae_s:.4f} s。两者均不是根据验证集外推的估计。**','',
      '### Model','',
      'Model: Simplified MCRL / CCI-based SMILES–Morgan RT predictor（历史代码命名为 CCRL / HSCL / SupCR）。',
      '',
      'SMILES encoder: ChemBERTa-zinc-base-v1 / 6-layer RoBERTa；拼接第 2、4、6 层 hidden states，经单层 BiLSTM（每方向 128 hidden units），输出 256 维表征；first-token pooling，长度 256。',
      '',
      'Morgan encoder: radius 2、2048-bit、useChirality=True；每 16 bit 构成一个 token，加 CLS 和位置嵌入；2 层 Transformer，hidden dimension 256、8 heads、FFN 512。',
      '',
      'CCI: `BimodalCrossAttentionFusion`，SMILES 为 query，Morgan 为 key/value；self-attention → residual AddNorm → cross-attention → residual AddNorm → FFN；256 dimensions / 8 heads。',
      '',
      'Contrastive learning: retention-time-guided SupCR / RTCL，跨视图投影对齐和融合表征 SupCR；按连续 RT 距离排序，τ=0.07，similarity_weight=0.05，ω_align=0.01，主单模型 ω_fusion=0.005，原始 ratio objective（supcr_use_log=False）。训练同时使用 MAE + Huber 回归目标，huber_weight=0.2。',
      '',
      'Prediction head: `h_res = concat(h_SMILES, h_CCI, h_Morgan)`，768 → 256 → 128 → 1 的 MLP（GELU、dropout）；预测经训练集拟合的 Yeo–Johnson 变换反变换为秒。对比投影头属于历史联合训练的一部分，测试不使用 RT 标签作为输入。',
      '',
      '### Removed modules','',
      '- PTDF：不再融合 CCI 分支与图分支的 RT 标量。',
      '- RDKit descriptors：去除 PureGIN 内部的 10 维理化/结构描述符及其 `desc_proj` 路径；Morgan 指纹仍通过 RDKit 生成。',
      '- GNN ensemble：去除 PureGIN seeds 2026 / 3407 / 777 的全部预测。',
      '',
      '注意：`cross_attention_residual` 中的 residual 指深度表征拼接，不是 RDKit descriptor residual correction，必须保留才能匹配 B 类架构。当前 Scheme A 的最终公式没有独立的 OOF/RF/GBDT residual correction；历史 descriptor 校正是辅助实验，不能再额外归入本次删除造成的损失。',
      '',
      '### A / B 模型区分与主要证据','',
      '| 类别 | 实际模型 | 来源 / 处理 |','|---|---|---|',
      f'| A 完整模型 | CCI Top6 + 三个带描述符的 PureGIN + PTDF | {link("outputs/final_fusion_ccrl_gnn/metrics_final_ccrl_gnn_ptdf.json","最终定义/指标")}；本次从保存预测重算指标 |',
      f'| B 单模型（主要简化结果） | SMILES + Morgan + CCI/RTCL + RT head；seed1234，epoch90 | {link("checkpoints/supcr_tuning_aliases/exact1234_best.pt","exact1234 checkpoint")}；本次完整 test 推理 |',
      f'| B 集成（仅删除 PTDF） | 原 Top6 候选池、5 个非零成员的 CCI 预测器加权 | {link("outputs/top6_pool_target_power_ensemble/calibrated_ensemble_metrics.json","原有 Top6 权重")}；本次从六个 checkpoint 的新预测重建 |',
      f'| B 历史独立消融 | E_residual_supcr seeds2026/777/3407 | {link("outputs/paper_ablation_20260825/02_supcr_strict_ablation/supcr_seed_results.csv","三 seed 实验")}；使用保存的 test 预测重新计算 |',
      '| 接近 B、缺少 RTCL | D_residual_no_supcr / G_cross_attention_no_supcr | 作为辅助对照；不等同于保留完整 CCI/RTCL 的 B |',
      '| 其它 split | scaffold / PredRet / 多系统多 fold | 已登记，但不混入当前 SMRT test 比较 |',
      '',
      '主单模型 exact1234 在六个原始 Top6 候选中验证集 MAE 最低（33.0599 s），且是最终消融表已有的单模型 baseline；没有根据本次 test 重新挑选模型。',
      '',
      '### Estimated performance（本次 checkpoint 实测）','',
      f'MAE: {single.mae_s:.4f} s  \nRMSE: {single.rmse_s:.4f} s  \nR²: {single.r2:.6f}',
      '',
      '| 模型 | MAE (s) | RMSE (s) | MedAE (s) | MAPE (%) | R² | Pearson r |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in [('Full MCRL（保存的最终预测）',full),('Simplified MCRL（单个 CCI，复评）',single),('CCI Top6，无 PTDF（复评）',ensemble)]:
        text.append(f'| {label} | {r.mae_s:.4f} | {r.rmse_s:.4f} | {r.medae_s:.4f} | {r.mape_pct:.4f} | {r.r2:.6f} | {r.pearson:.6f} |')
    text += ['', '### Comparison','',
      'Full MCRL: MAE 25.89 s（用户给定值；保存预测重算为 25.8893 s）。','',
      f'Simplified MCRL: MAE {single.mae_s:.2f} s（一个 CCI 预测器）；保留 CCI Top6 时为 {ensemble.mae_s:.2f} s。','',
      '| 简化方案 | MAE 增加 (s) | MAE 相对增加 | RMSE 增加 (s) | RMSE 相对增加 | R² 绝对下降 | R² 相对下降 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in [('单个 CCI',single),('CCI Top6、只去 PTDF',ensemble)]:
        text.append(f'| {name} | {r.mae_s-full.mae_s:.4f} | {(r.mae_s/full.mae_s-1)*100:.2f}% | {r.rmse_s-full.rmse_s:.4f} | {(r.rmse_s/full.rmse_s-1)*100:.2f}% | {full.r2-r.r2:.6f} | {(1-r.r2/full.r2)*100:.2f}% |')
    text += ['',
      '计算定义：ΔMAE = MAE_simplified − MAE_full；相对 MAE 变化 = ΔMAE / MAE_full × 100%。R² 绝对下降是数值差，不是百分比；相对下降另按 R²_full 为分母计算。表格使用完整精度结果；另存相对于用户 25.89 / 52.23 / 0.9124 的比较，保留两位小数时结论相同。不能把“MAE 增加百分比”泛化为所有指标的统一性能下降百分比。',
      '',
      '系统级消融解释：保留 CCI Top6 的方案只去掉整个 PTDF/图/描述符分支，是回答“删除 PTDF”最直接的对照；单个 CCI 的方案还取消了前半部分的 Top6 集成。因此单模型相对于完整模型的全部损失不能仅归因于 PTDF。',
      '',
      '删除 PTDF 后返回完整的 `pred_ensemble` / 单模型输出，不保留最终公式中的 0.30 缩放，否则 RT 会被错误缩小。',
      '',
      '### 六个 B 类 checkpoint 的 test 复评','',
      '| 成员 | epoch | 原 Top6 权重 | 历史 val MAE (s) | 本次 test MAE (s) | RMSE (s) | R² |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for m in protocol['members']:
        r=current.loc[m['name']];v=val[val.column=='pred_'+m['name']].iloc[0]
        text.append(f'| {m["name"]} | {m["epoch"]} | {m["ensemble_weight"]:.2f} | {v.mae_s:.4f} | {r.mae_s:.4f} | {r.rmse_s:.4f} | {r.r2:.6f} |')
    text += ['',
      'Top6 六个候选的固定权重分别为 0.30、0.25、0.10、0.20、0.15、0.00（表中顺序），合计为 1。最后一个成员原本权重为零；本次仍评估以完善历史记录，重建集成时保持零权重。',
      '',
      '### 其它历史消融结果','',
      '| 实验 | val MAE (s) | test MAE (s) | test RMSE (s) | test R² |','|---|---:|---:|---:|---:|']
    for r in historical_summary:
        text.append(f'| {r["model"]} | {r["validation_mae_s"]:.4f} | {r["mae_s"]:.4f} | {r["rmse_s"]:.4f} | {r["r2"]:.6f} |')
    text += ['',
      f'E 系列带 SupCR 的三 seed test MAE 为 {erows.mae_s.min():.2f}–{erows.mae_s.max():.2f} s，均值 {erows.mae_s.mean():.4f} ± {erows.mae_s.std(ddof=1):.4f} s（样本标准差），与“单模型约 31–33 s”的判断一致。不是把各 seed 的预测平均后的 ensemble MAE。',
      '',
      f'严格 D/E 三 seed 对照中，不使用 SupCR 的平均 MAE 为 {drows.mae_s.mean():.4f} s，使用 SupCR 为 {erows.mae_s.mean():.4f} s；已有数据没有显示稳定的平均 MAE 改善。不能仅凭当前完整模型性能推断 CCI/RTCL 具有显著独立贡献。',
      '',
      '### 审计范围、对齐与数值复现','',
      f'- 递归登记 {summary["file_counts"]["checkpoint_or_tensor_cache"]:,} 个 .pt/.pth/.ckpt 模型或缓存、{summary["file_counts"]["log"]:,} 个日志、{summary["file_counts"]["metrics_prediction_config_report_or_data"]:,} 个 CSV/JSON/TXT/MD 等记录或数据文件。',
      f'- 检查 {summary["file_counts"]["archive"]} 个 ZIP 的成员清单，登记 {summary["archived_records"]} 个可能相关的存档条目；没有需要替代本地匹配 checkpoint 的存档恢复。未发现磁盘 TensorBoard `tfevents` 文件。',
      f'- 29 个路径具备 SMILES–Morgan cross-attention 架构候选元数据，包括 best/latest、发布副本、无 SupCR 对照和 scaffold 模型，不能解释为 29 个独立且完全匹配的 B 实验。具体分类见 checkpoint_classification.csv。',
      f'- 从符合条件的保存预测中重算 {summary["current_split_prediction_columns"]} 个当前 val/test 预测列；{summary["prediction_files_not_usable_for_current_split"]} 个候选文件缺少逐样本身份信息或属于其它样本集，排除于本次同划分比较。',
      '- 四个恢复/烟雾测试 checkpoint 含 sklearn / numpy 序列化对象，受限元数据读取器无法解析，已记录具体路径及错误；六个用于本结论的 checkpoint 均已成功加载并严格匹配全部权重。',
      '- 排除 Git、依赖库、缓存的 Python bytecode 等非实验目录；未修改历史结果。逐文件清单同时保留不相关文件，便于追查检索范围。',
      '',
      '当前划分：train 63,163；val 7,019；test 7,798，RT 单位为秒，test RT 范围 365–1461 s。三个文件 SHA-256 与 `data/unified_split_sha256.txt` 一致。历史预测必须与当前 split 的完整 sample_id 集合、SMILES、真实 RT 一致后才能纳入；不以文件名或样本数单独确认划分。',
      '',
      '六个模型均以 `eval()` + `torch.inference_mode()`、float32、batch32、单一原始 SMILES 视图推理；不计算训练 loss、不反向传播、不更新参数。Yeo–Johnson 仅拟合 train.csv；测试集不参与归一化拟合、模型挑选或权重搜索。',
      '',
      '指标统一用当前 CSV 的 float64 RT 计算：MedAE 是 NumPy 标准中位数（偶数样本取中间两数平均）；MAPE = 100 × mean(|pred−true|/|true|)，本 test 没有零 RT；Pearson 为预测与标签的线性相关系数，不能替代 R²。历史 PyTorch lower median 与标准中位数、float32 标签会产生很小的差异。',
      '',
      '本次环境：Python '+protocol['versions']['python']+'，PyTorch '+protocol['versions']['torch']+'，RDKit '+protocol['versions']['rdkit']+'，RTX 4070 Laptop GPU。与服务器保存预测之间存在小幅数值偏差：','',
      '| 模型 | 平均绝对预测差 (s) | 最大绝对预测差 (s) | 所有样本 <0.05 s |','|---|---:|---:|---|']
    for c in checks:
        text.append(f'| {c["model"]} | {c["mean_abs_difference_s"]:.6f} | {c["max_abs_difference_s"]:.6f} | {c["within_0_05s"]} |')
    single_hist=hrow('outputs/top6_pool_target_power_ensemble/base_predictions/target_power_exact_repro_seed1234_test.csv','pred_target_power_exact_repro_seed1234')
    ens_hist=hrow('outputs/top6_pool_target_power_ensemble/predictions_test_calibrated.csv','pred_ensemble')
    text += ['',
      f'严格逐样本 0.05 s 阈值没有全部通过，不能宣称位级复现。主要单模型历史 MAE {single_hist.mae_s:.8f} → 本次 {single.mae_s:.8f} s；Top6 历史 MAE {ens_hist.mae_s:.8f} → 本次 {ensemble.mae_s:.8f} s。主要结论所用 MAE 差小于 0.0001 s，四位小数 R² 和两位小数 MAE/RMSE 结论保持一致；偏差可能来自推理硬件/算子与软件环境，未单独定位原因。',
      '',
      '### Conclusion / 论文竞争力','',
      f'删除整个 PTDF/图/描述符分支、保留 CCI Top6：MAE 从 25.89 增至 {ensemble.mae_s:.2f} s，增加 {ens_delta:.2f} s（误差增加 {(ensemble.mae_s/full.mae_s-1)*100:.2f}%）；R² 为 {ensemble.r2:.4f}。进一步改成一个 SMILES–Morgan CCI + RT head：MAE {single.mae_s:.2f} s、RMSE {single.rmse_s:.2f} s、R² {single.r2:.4f}；相对完整模型 MAE 增加 {mae_delta:.2f} s（{(single.mae_s/full.mae_s-1)*100:.2f}%）。',
      '',
      '项目已有同划分复现汇总中的 Transformer-LSTM MAE 为 26.2352 s、DeepGCN-RT 为 26.6228 s；本次未重新运行这些基线。单 CCI 和 CCI Top6 的 MAE 均高于这两个较强对照，因此不能保留完整 MCRL 的精度优势或宣称领先。仍可作为可用的跨视图 RT 预测模型、表示学习研究对象和消融基线，但单凭这些预测结果，独立论文的性能竞争力偏弱。该判断主要依据同划分对照，而不是仅比较文献表。',
      '',
      '外部文献作为背景：DeepGCN-RT 报告 SMRT MAE 26.55 s；RT-Transformer 报告去除未保留分子后 MAE 27.30 s（未过滤为 33.41 s）；ABCoRT 报告去除未保留分子后 MAE 25.75 s。这些不是本项目同一 test split 的重新评估，不能直接合并为严格排名。分别见 [DeepGCN-RT 原文](https://www.sciencedirect.com/science/article/pii/S0021967323006647)、[RT-Transformer 原文摘要](https://pubmed.ncbi.nlm.nih.gov/38402516/)、[ABCoRT 原文](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c02179)。',
      '',
      '若论文仅保留前半部分，应把主要结果改为“单 CCI 实测约 31.06 s”或明确标注“CCI ensemble 29.86 s”，保留 RTCL 贡献证据的限度，并通过现有严格消融、泛化或推理成本证据支持研究价值。当前结果不足以单独证明可发表性，也不支持无 PTDF 后仍维持 25.89 s。',
      '',
      '### 可复核产物','',
      '| 文件 | 内容 |','|---|---|']
    for file,desc in [('reevaluated_test_metrics.csv','六个 checkpoint、无 PTDF 集成、完整模型的六项指标'),
      ('predictions_test_comparison.csv','7798 条对齐的真实 RT、各单模型、无 PTDF 集成和完整模型预测'),
      ('performance_changes.csv','相对本次完整精度指标的绝对/相对变化'),
      ('comparison_with_user_rounded_reference.csv','相对于用户给定 25.89 / 52.23 / 0.9124 的比较'),
      ('evaluation_protocol.json','完整 checkpoint args、SHA-256、epoch、软件版本及评估配置'),
      ('historical_prediction_reproduction_checks.json','新预测与历史预测逐样本偏差'),
      ('historical_current_split_metrics.csv','当前 val/test 上可对齐的历史预测指标'),
      ('historical_ablation_metrics.csv','A/B/C/D/E/G 历史消融汇总'),
      ('file_inventory.csv','递归文件清单'),('archive_inventory.csv','ZIP 成员索引'),
      ('record_keyword_hits.csv','日志、配置和报告中的关键词线索'),
      ('checkpoint_metadata.json','所有模型或张量 checkpoint 的受限元数据读取结果'),
      ('checkpoint_classification.csv','架构与 split 分类'),('prediction_alignment_rejections.csv','其它 split 或身份无法核实的预测候选'),
      ('evaluation.log','本次实际推理日志')]:
        text.append(f'| {link("outputs/simplified_mcrl_audit_20261011/"+file,file)} | {desc} |')
    text += ['', '复现命令（项目根目录；无需训练）：','', '```powershell',
      'python scripts/audit_simplified_mcrl_20261011.py',
      'python scripts/evaluate_simplified_mcrl_20261011.py',
      'python scripts/report_simplified_mcrl_20261011.py','```','']
    report=OUT/'simplified_mcrl_performance_estimation.md'
    report.write_text('\n'.join(text),encoding='utf-8')
    print(report)
    print(pd.DataFrame(comparisons)[['model','delta_mae_s','relative_change_mae_s_pct','delta_rmse_s','relative_change_rmse_s_pct','delta_r2']].to_string(index=False))
if __name__=='__main__': main()
