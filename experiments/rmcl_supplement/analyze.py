"""Audit saved single-predictor data without neural training or invented results."""
import statistics as st
from pathlib import Path
from common import ROOT, SEEDS, aligned_predictions, metrics, read_csv, read_json, selected_assets, write_csv, write_json


def main(output='results/analysis'):
    out=ROOT/output
    if not out.resolve().is_relative_to((ROOT/'results').resolve()) and not out.resolve().is_relative_to((ROOT/'validation').resolve()):
        raise ValueError('Analysis output must stay under results/ or validation/')
    truth=ROOT/'data/unified/test.csv'
    records=[]
    for variant in ['base','cma','rscl','full']:
        for seed in SEEDS:
            a=selected_assets(variant,seed)
            if a is None:
                records.append(dict(variant=variant,seed=seed,status='DATA REQUIRED'))
                continue
            y,p=aligned_predictions(a['predictions'],truth)
            records.append(dict(variant=variant,seed=seed,status='available',origin=a['origin'],source=str(a['predictions'].relative_to(ROOT)),**metrics(y,p)))
    write_csv(out/'ablation_seed_metrics.csv',records)
    means=[]
    for variant in ['base','cma','rscl','full']:
        rs=[r for r in records if r['variant']==variant and r['status']=='available']
        vals=[r['MAE'] for r in rs]
        means.append(dict(variant=variant,n_seeds=len(vals),complete_three_seed=len(vals)==3,
                          MAE_mean=st.mean(vals) if vals else None,MAE_SD=st.stdev(vals) if len(vals)>1 else None))
    write_csv(out/'ablation_mean_sd.csv',means)
    effects=[]
    for seed in SEEDS:
        cells={r['variant']:r for r in records if r['seed']==seed and r['status']=='available'}
        for label,before,after in [('CMA_without_RSCL','base','cma'),('RSCL_with_CMA','cma','full'),('RSCL_without_CMA','base','rscl')]:
            if before in cells and after in cells:
                effects.append(dict(seed=seed,effect=label,MAE_reduction=cells[before]['MAE']-cells[after]['MAE']))
        if len(cells)==4:
            effects.append(dict(seed=seed,effect='interaction_difference_in_differences',
                MAE_reduction=(cells['cma']['MAE']-cells['full']['MAE'])-(cells['base']['MAE']-cells['rscl']['MAE'])))
    write_csv(out/'paired_configuration_effects.csv',effects)
    baseline=[]
    for name in ['gcn_kensert','one_d_cnn_original_style','gnn_rt','deepgcn_rt']:
        p=ROOT/f'reference/baselines/{name}.csv'
        try:
            y,pr=aligned_predictions(p,truth)
            baseline.append(dict(model=name,status='identity_aligned',**metrics(y,pr)))
        except Exception as e:
            baseline.append(dict(model=name,status='UNVERIFIED_NOT_RANKED',reason=str(e)))
    archived=read_csv(ROOT/'reference/baselines/transformer_lstm_archived.csv')
    baseline.append(dict(model='archived_transformer_lstm',status='NO_MOLECULAR_IDENTIFIERS_NOT_STRICTLY_RANKED',
        **metrics([float(r['rt']) for r in archived],[float(r['prediction']) for r in archived])))
    y,pr=aligned_predictions(ROOT/'reference/main_single/predictions_test.csv',truth)
    main_metric=metrics(y,pr)
    baseline.append(dict(model='single_RMCL_exact1234',status='identity_aligned',**main_metric))
    write_csv(out/'baseline_identity_audit.csv',baseline)
    raw=read_csv(ROOT/'reference/main_single/predictions_test.csv')
    ys=sorted(y)
    def quantile(q):
        z=(len(ys)-1)*q; i=int(z);frac=z-i
        return ys[i]*(1-frac)+ys[min(i+1,len(ys)-1)]*frac
    edges=[quantile(i/10) for i in range(11)]
    bins=[]
    for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
        rs=[r for r in raw if (float(r['true_rt'])>=lo if i==0 else float(r['true_rt'])>lo) and float(r['true_rt'])<=hi]
        bins.append(dict(lower=lo,upper=hi,lower_inclusive=i==0,**metrics([float(r['true_rt']) for r in rs],[float(r['pred_rt']) for r in rs])))
    write_csv(out/'main_single_RT_deciles.csv',bins)
    y,pr=aligned_predictions(ROOT/'reference/generalization/scaffold_single_predictions.csv',ROOT/'data/scaffold/test.csv')
    scaffold=metrics(y,pr)
    transfer=[]
    frozen=read_csv(ROOT/'reference/predret/frozen_single.csv')
    for system in sorted({r['system'] for r in frozen}):
        fr=[r for r in frozen if r['system']==system]
        values={key:[] for key in ['single','transformer_lstm']}
        for fold in range(10):
            for key in values:
                path=ROOT/f'reference/predret/{key}/{system}/fold_{fold:02d}.csv'
                target=ROOT/f'data/predret_strict/{system}/fold_{fold:02d}/test.csv'
                y,pr=aligned_predictions(path,target)
                values[key].append(metrics(y,pr)['MAE'])
        transfer.append(dict(system=system,n=len(fr),frozen_single_MAE=metrics([float(r['rt']) for r in fr],[float(r['pred_rt']) for r in fr])['MAE'],
            adapted_single_MAE_mean=st.mean(values['single']),adapted_single_MAE_SD=st.stdev(values['single']),
            transformer_lstm_MAE_mean=st.mean(values['transformer_lstm']),transformer_lstm_MAE_SD=st.stdev(values['transformer_lstm'])))
    write_csv(out/'predret_single_comparison.csv',transfer)
    summary=dict(main_single=main_metric,scaffold_single=scaffold,
        new_training_performed=False,new_checkpoint_inference_performed=False,
        missing_ablation_cells=[dict(variant=r['variant'],seed=r['seed']) for r in records if r['status']!='available'],
        baseline_identity_failures=[r['model'] for r in baseline if r['status']!='identity_aligned'],
        predret_single_winning_systems=[r['system'] for r in transfer if r['adapted_single_MAE_mean']<r['transformer_lstm_MAE_mean']],
        interpretation='fold SD is not a confidence interval; source overlap and historical runtime differences remain disclosed in protocol.json')
    write_json(out/'summary.json',summary)
    print(f'Analysis written to {out}; missing cells={len(summary["missing_ablation_cells"])}')
    return summary
