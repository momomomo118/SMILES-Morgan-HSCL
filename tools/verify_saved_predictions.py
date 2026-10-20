"""Recompute archived SI metrics and partition checks; no ML dependencies or fitting."""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]

def metrics(y, p):
    n = len(y)
    if not n or len(p) != n or any(not math.isfinite(v) for v in y + p):
        raise ValueError('Empty, nonfinite, or unequal metric inputs')
    e = [b-a for a,b in zip(y,p)]
    ae = list(map(abs,e))
    mu = math.fsum(y)/n
    sse = math.fsum(v*v for v in e)
    sst = math.fsum((v-mu)**2 for v in y)
    return {'n':n, 'mae_s':math.fsum(ae)/n, 'rmse_s':math.sqrt(sse/n),
            'medae_s':st.median(ae), 'mape_pct':100*math.fsum(a/abs(v) for a,v in zip(ae,y))/n,
            'r2':1-sse/sst}

def group(rows, *keys):
    result = collections.defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--si-dir', type=Path, default=ROOT/'supporting_information')
    ap.add_argument('--json-output', type=Path)
    args = ap.parse_args()
    cache = {}
    checks = []
    computed = {}
    differences = []
    max_truth_rounding = 0.0
    def load(name):
        if name not in cache:
            with (args.si_dir/'data'/name).open(encoding='utf-8',newline='') as f:
                cache[name] = list(csv.DictReader(f))
        return cache[name]
    def check(ok, name, detail=None):
        checks.append({'name':name, 'passed':bool(ok), 'detail':detail})
        if not ok:
            raise ValueError(name + ': ' + str(detail))
    def near(a,b,label,tol=5e-9):
        delta=abs(float(a)-float(b))
        differences.append(delta)
        check(delta<=tol,label,delta)
    def compare(m,row,prefix=''):
        for key,value in m.items():
            if key == 'n':
                continue
            near(value,row[prefix+key],prefix+key)
    partitions={}
    for name,label,counts in [('S1_smrt_random_assignments.csv','random',{'train':63163,'val':7019,'test':7798}),
                              ('S1_smrt_scaffold_assignments.csv','scaffold',{'train':63088,'val':7035,'test':7857})]:
        rows=load(name)
        check(len(rows)==77980 and len({r['sample_id'] for r in rows})==77980,label+' unique records')
        partitions[label]={r['sample_id']:r for r in rows}
        check(dict(collections.Counter(r['split'] for r in rows))==counts,label+' split counts')
        for col in ['sample_id','canonical_smiles','murcko_scaffold']:
            sets={k:{r[col] for r in rows if r['split']==k} for k in counts}
            for x,y in itertools.combinations(counts,2):
                actual=len(sets[x]&sets[y])
                saved=[r for r in load('S1_overlap_checks.csv') if r['dataset']=='SMRT_'+label and r['left']==x and r['right']==y and r['key']==col]
                if not saved:
                    saved=[r for r in load('S1_overlap_checks.csv') if r['dataset']=='SMRT_'+label and r['left']==y and r['right']==x and r['key']==col]
                check(len(saved)==1 and actual==int(saved[0]['shared_unique']),label+' '+col+' '+x+'/'+y,actual)
    check(set(partitions['random'])==set(partitions['scaffold']),'identical SMRT molecule IDs')
    for sid,r in partitions['random'].items():
        other=partitions['scaffold'][sid]
        check(r['smiles']==other['smiles'] and r['rt']==other['rt'],'SMRT record retained')
    # Pair saved predictions to explicit partition labels, not a positional row order.
    def bind(rows,partition,predcol):
        nonlocal max_truth_rounding
        yy=[];pp=[]
        for r in rows:
            target=partitions[partition][r['sample_id']]
            check(r['smiles']==target['smiles'],'SMRT SMILES binding')
            if 'split' in r:
                check(r['split']==target['split'],'SMRT split binding')
            truth=float(target['rt'])
            delta=abs(float(r['true_rt'])-truth)
            max_truth_rounding=max(max_truth_rounding,delta)
            check(delta<=1e-3,'stored true_rt rounding tolerance',delta)
            yy.append(truth);pp.append(float(r[predcol]))
        return metrics(yy,pp)
    ab={}
    for (run,),rows in group(load('S3_ablation_test_predictions.csv'),'run_id').items():
        check(len(rows)==7798 and {r['sample_id'] for r in rows}=={k for k,v in partitions['random'].items() if v['split']=='test'},run+' test IDs')
        ab[run]=bind(rows,'random','pred_rt')
        saved=next(r for r in load('S3_ablation_run_metrics.csv') if r['run_id']==run)
        compare(ab[run],saved)
    computed['ablation']=ab
    for (model,),rows in group(load('S3_baseline_test_predictions.csv'),'model').items():
        saved=next(r for r in load('S3_baseline_metrics_and_sources.csv') if r['model']==model)
        m=bind(rows,'random','pred_rt');compare(m,saved)
        computed['baseline/'+model]=m
    paired=[]
    for row in load('S3_rtcl_seed_pairs.csv'):
        seed=row['seed'];d=ab['D_residual_no_supcr_seed'+seed]['mae_s'];e=ab['E_residual_supcr_seed'+seed]['mae_s']
        near(d,row['without_rtcl_mae_s'],'RTCL without');near(e,row['with_rtcl_mae_s'],'RTCL with');near(d-e,row['mae_reduction_s'],'RTCL paired reduction')
        paired.append(d-e)
    computed['rtcl_paired_reduction_mean_s']=st.mean(paired)
    computed['rtcl_paired_reduction_sample_sd_s']=st.stdev(paired)
    finalrows=load('S2_smrt_final_predictions.csv')
    for (split,),rows in group(finalrows,'split').items():
        for r in rows:
            pred=math.fsum(float(r[c])*w for c,w in [('ccrl_ensemble',.30),('g2026',.24),('g3407',.18),('g777',.28)])
            near(pred,r['pred_final_ccrl_gnn_ptdf'],'SMRT fusion identity',1e-9)
        for col in ['ccrl_ensemble','pred_final_ccrl_gnn_ptdf']:
            m=bind(rows,'random',col)
            saved=next(r for r in load('S2_smrt_ensemble_metrics.csv') if r['split']==split and (('CCI' in r['model'])==(col=='ccrl_ensemble')))
            compare(m,saved);computed['SMRT/'+split+'/'+col]=m
        for seed in ['2026','3407','777']:
            m=bind(rows,'random','g'+seed)
            saved=next(r for r in load('S2_puregin_metrics.csv') if r['split']==split and r['model']=='PureGIN_'+seed)
            compare(m,saved)
    for (split,),rows in group(load('S4_scaffold_predictions.csv'),'split').items():
        for r in rows:
            inner=math.fsum(float(r[c])*w for c,w in [('pred_exact1234',.30),('pred_exact2025',.15),('pred_dropout015',.25),('pred_lowfusion',.10),('pred_aug050',.20)])
            near(inner,r['pred_cci_ensemble'],'scaffold inner fusion',1e-9)
            pred=.30*inner+.24*float(r['pred_gin2026'])+.18*float(r['pred_gin3407'])+.28*float(r['pred_gin777'])
            near(pred,r['pred_mcrl'],'scaffold outer fusion',1e-9)
        m=bind(rows,'scaffold','pred_mcrl')
        compare(m,next(r for r in load('S4_scaffold_metrics.csv') if r['split']==split))
        computed['scaffold/'+split]=m
    clean=load('S1_predret_cleaned.csv');cleanmap={r['sample_id']:r for r in clean}
    counts=collections.Counter(r['system'] for r in clean)
    expected={'Cao_HILIC':520,'Eawag_XBridgeC18':364,'FEM_Lipids':86,'FEM_long':412,'FEM_short':136,'IPB_Halle':76,'LIFE_new':173,'LIFE_old':183,'MTBLS87':146,'UniToyama_Atlantis':101}
    check(dict(counts)==expected,'PredRet ten-system counts')
    check(sum(int(r['n_merged']) for r in clean)==2391,'PredRet positive records before merging')
    assignment=group(load('S1_predret_fold_assignments.csv'),'system','outer_fold')
    test_seen=collections.Counter(); folds={}
    predictions=group(load('S4_predret_outer_test_predictions.csv'),'system','outer_fold')
    for key,rows in assignment.items():
        roles=group(rows,'role');sets={k[0]:{r['sample_id'] for r in v} for k,v in roles.items()}
        check(set(sets)=={'inner_train','inner_validation','outer_test'},'PredRet roles')
        expected_ids={r['sample_id'] for r in clean if r['system']==key[0]}
        check(set.union(*sets.values())==expected_ids,'PredRet fold coverage')
        for left,right in itertools.combinations(sets,2):
            check(not sets[left]&sets[right],'PredRet ID separation')
            lcan={cleanmap[s]['canonical_smiles'] for s in sets[left]};rcan={cleanmap[s]['canonical_smiles'] for s in sets[right]}
            check(not lcan&rcan,'PredRet canonical separation')
        pr=predictions[key]
        check({r['sample_id'] for r in pr}==sets['outer_test'],'outer prediction IDs')
        test_seen.update(sets['outer_test'])
        saved=next(r for r in load('S4_predret_fold_results.csv') if (r['system'],r['outer_fold'])==key)
        for role,countcol in [('inner_train','n_inner_train'),('inner_validation','n_inner_validation'),('outer_test','n_test')]:
            check(len(sets[role])==int(saved[countcol]),'fold role count')
        fm={}
        for method,col in [('proposed','proposed_pred_rt'),('baseline','baseline_pred_rt')]:
            yy=[];pp=[]
            for r in pr:
                target=cleanmap[r['sample_id']]
                check(r['smiles']==target['smiles'] and r['system']==target['system'],'PredRet molecule binding')
                near(r['true_rt'],target['rt'],'PredRet target label',1e-9)
                yy.append(float(target['rt']));pp.append(float(r[col]))
            fm[method]=metrics(yy,pp);compare(fm[method],saved,method+'_')
        folds[key]=fm
    check(len(assignment)==100 and len(predictions)==100,'100 outer folds')
    check(len(test_seen)==2197 and set(test_seen.values())=={1},'each target record evaluated once')
    weights=group(load('S4_predret_fold_weights.csv'),'system','outer_fold')
    for key,ww in weights.items():
        check(len(ww)==8 and len({r['predictor'] for r in ww})==8,'eight candidate weights')
        check(all(float(r['weight'])>=0 for r in ww),'nonnegative fold weights')
        near(math.fsum(float(r['weight']) for r in ww),1,'weight simplex')
    wins=[]
    for row in load('S4_predret_system_results.csv'):
        system=row['system']
        for method in ['proposed','baseline']:
            for metric in ['mae_s','rmse_s','medae_s','mape_pct','r2']:
                vv=[folds[(system,str(i))][method][metric] for i in range(10)]
                near(st.mean(vv),row[method+'_'+metric+'_mean'],'system fold mean')
                near(st.stdev(vv),row[method+'_'+metric+'_sd'],'system sample SD')
        if float(row['delta_fold_mean_mae_s'])<0:
            wins.append(system)
    check(len(wins)==6,'six lower-MAE target systems',wins)
    # Store compact evidence: no per-molecule values in the verification log.
    result={'status':'PASS','checks':len(checks),'metric_comparisons':len(differences),'max_numeric_comparison_difference':max(differences),'max_embedded_SMRT_label_rounding_s':max_truth_rounding,'computed':computed,'predret_winning_systems':wins,'structural_checks':'Set checks use archived canonical/Murcko keys; structures were not recomputed with RDKit.','verification_scope':['Fixed partitions and record binding','Archived structural keys','Saved prediction metrics and fusion coefficients','Assignment-table RT labels checked against embedded labels within 0.001 s']}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Saved predictions PASS:',len(checks),'checks;',len(differences),'numeric comparisons; six of ten target systems lower MAE.')
    print('No model fitting, neural inference, or prediction changes were performed.')

if __name__=='__main__':
    main()
