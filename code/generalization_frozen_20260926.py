"""Frozen SMRT -> PredRet evaluation; never fits to target-system labels.

Run from any directory with the project's rt-tr Python environment.
Original models, data and historical results are read-only.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys, json, hashlib, gc, argparse, platform, subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
# Isolated dependency: leaves the installed rt-tr environment unchanged.
RDKIT_RUNTIME = ROOT/'work/generalization_runtime_rdkit2025'
if RDKIT_RUNTIME.exists(): sys.path.insert(0,str(RDKIT_RUNTIME))
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from transformers import AutoTokenizer

sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code')]
from train_multimodal import build_model
from train_pure_gnn_for_ensemble import PureGIN, smiles_to_graph, collate

MEMBERS = [
 ('exact1234', 'exact1234_best.pt', 'target_power_exact_repro_seed1234', .30),
 ('dropout015', 'dropout015_seed777_best_cross_attention_residual.pt', 'target_power_div_dropout015_seed777', .25),
 ('lowfusion', 'lowfusion_seed777_best_cross_attention_residual.pt', 'target_power_div_lowfusion_seed777', .10),
 ('aug050', 'aug050_seed777_best_cross_attention_residual.pt', 'target_power_div_aug050_seed777', .20),
 ('exact2025', 'exact2025_best.pt', 'target_power_exact_repro_seed2025', .15),
]
GIN_WEIGHTS = {2026: .24, 3407: .18, 777: .28}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: raise ValueError('Invalid SMILES: '+str(s))
    return Chem.MolToSmiles(m, isomericSmiles=True)

@torch.inference_mode()
def predict_cci(model, smiles, tokenizer, transform, device, batch_size):
    # API accepts molecular strings only, no target labels or system identifiers.
    result=[]
    for start in range(0, len(smiles), batch_size):
        ss=smiles[start:start+batch_size]
        tokens=tokenizer(ss, truncation=True, padding='max_length', max_length=256,
                         add_special_tokens=True, return_tensors='pt')
        fps=torch.tensor([list(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s), 2, nBits=2048, useChirality=True)) for s in ss], dtype=torch.float32)
        p=model(input_ids=tokens['input_ids'].to(device), attention_mask=tokens['attention_mask'].to(device),
                morgan_fp=fps.to(device)).prediction.detach().cpu().numpy().reshape(-1,1)
        result.extend(transform.inverse_transform(p).reshape(-1).tolist())
    return np.asarray(result)

def graphs(smiles):
    items=[]
    for i,s in enumerate(smiles):
        x,a,d=smiles_to_graph(s)
        items.append(dict(sample_id=str(i),smiles=s,x=torch.tensor(x),adj=torch.tensor(a),
                          desc=torch.tensor(d),y=torch.tensor(0.)))
    return items

@torch.inference_mode()
def predict_gin(model, items, device, batch_size):
    result=[]
    for start in range(0,len(items),batch_size):
        b=collate(items[start:start+batch_size])
        p=model(*(b[k].to(device) for k in ['x','adj','mask','desc']))
        result.extend(p.cpu().numpy().tolist())
    return np.asarray(result)

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    assert len(y)==len(p) and np.isfinite(y).all() and np.isfinite(p).all()
    rho=float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None
    return dict(n=len(y), mae_s=float(np.mean(abs(y-p))), rmse_s=float(np.sqrt(np.mean((y-p)**2))),
                median_ae_s=float(np.median(abs(y-p))),r2=float(r2_score(y,p)) if len(y)>1 else None,
                spearman=rho)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'outputs/generalization_20260926/predret_frozen_verified'))
    parser.add_argument('--gin-feature-python',default=sys.executable)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4); torch.manual_seed(20260926)
    source_train=ROOT/'data/unified/train.csv'
    target_path=ROOT/'data/external_predret/transfer_10_systems_cleaned/all_cleaned.csv'
    paths=[source_train, ROOT/'data/unified/val.csv',ROOT/'data/unified/test.csv',target_path]
    paths += [ROOT/'FINAL_RELEASE_SCHEME_A/checkpoints'/m[1] for m in MEMBERS]
    paths += [ROOT/f'checkpoints/gnn/pure_gin_seed{s}/best_pure_gin.pt' for s in GIN_WEIGHTS]
    paths += sorted((ROOT/'pretrained/ChemBERTa-zinc-base-v1').glob('*.*'))
    protocol=dict(experiment='Frozen SMRT MCRL -> PredRet (zero-shot)',target_labels_used_for_fit=False,
        target_labels_used_for_selection=False,source_transform_fit='data/unified/train.csv only',
        cci_weights={m[0]:m[3] for m in MEMBERS},zero_weight_candidate='exact777 (0.0; skipped)',
        final_weights={'cci_ensemble':.30,**{f'gin{s}':w for s,w in GIN_WEIGHTS.items()}},
        residual_calibration=False,tta=1,rt_unit='seconds',reproduction_n=64,reproduction_max_abs_tolerance_s=.05,
        preprocessing='Original input SMILES for inference; canonical isomeric SMILES for overlap audit only',
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths if p.is_file()},
        software=dict(python=platform.python_version(),torch=torch.__version__,rdkit_cci=rdBase.rdkitVersion,rdkit_gin_features='2026.03.3'),
        gin_feature_script_sha256=digest(ROOT/'code/generalization_gin_features_20260926.py'),
        script_sha256=digest(Path(__file__)))
    protocol_path=out/'frozen_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8'))==protocol, 'Protocol changed; use a new output directory.'
    else: write_json(protocol_path,protocol)
    df=pd.read_csv(target_path)
    assert df.sample_id.is_unique and len(df)==2197 and df.system.nunique()==10
    assert np.allclose(df.rt,df.rt_min*60)
    df['canonical_smiles']=df.smiles.map(canonical)
    seen=set()
    for split in ['train','val','test']:
        source=pd.read_csv(ROOT/f'data/unified/{split}.csv')
        keys=set(source.smiles.map(canonical)); seen |= keys
        df[f'seen_smrt_{split}']=df.canonical_smiles.isin(keys)
        print(f'Overlap {split}: {df[f"seen_smrt_{split}"].sum()}',flush=True)
    df['unseen_smrt_all']=~df.canonical_smiles.isin(seen)
    df['unseen_smrt_development']=~(df.seen_smrt_train|df.seen_smrt_val)
    train_rt=pd.read_csv(source_train).rt.to_numpy(dtype=float).reshape(-1,1)
    transform=PowerTransformer(method='yeo-johnson',standardize=True).fit(train_rt)
    write_json(out/'source_transform.json',dict(lambdas=transform.lambdas_.tolist(),
               mean=transform._scaler.mean_.tolist(),scale=transform._scaler.scale_.tolist()))
    test=pd.read_csv(ROOT/'data/unified/test.csv').iloc[np.linspace(0,7797,64,dtype=int)].copy()
    smiles=test.smiles.tolist()+df.smiles.tolist()
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'pretrained/ChemBERTa-zinc-base-v1'),local_files_only=True)
    checks=[]; predictions={}
    def accept(name,p,reference):
        assert len(p)==len(smiles) and np.isfinite(p).all()
        ref=reference.set_index('sample_id').loc[test.sample_id]
        assert ref.smiles.tolist()==test.smiles.tolist()
        delta=abs(p[:len(test)]-ref.iloc[:,-1].to_numpy(float))
        checks.append(dict(member=name,n=len(test),max_abs_difference_s=float(delta.max()),
                           mean_abs_difference_s=float(delta.mean()),passed=bool(delta.max()<.05)))
        write_json(out/'source_reproduction_checks.json',checks)
        assert delta.max()<.05, f'{name}: source predictions not reproduced: {delta.max()}'
        predictions[name]=p[len(test):]
        print(name,checks[-1],flush=True)
    for name,filename,refname,w in MEMBERS:
        cache=out/f'raw_{name}.npy'
        if cache.exists(): p=np.load(cache)
        else:
            c=torch.load(ROOT/'FINAL_RELEASE_SCHEME_A/checkpoints'/filename,map_location='cpu',weights_only=False)
            a=SimpleNamespace(**c['args']); a.smiles_model=str(ROOT/'pretrained/ChemBERTa-zinc-base-v1')
            assert a.tta_smiles==1 and a.max_length==256 and a.target_transform=='power'
            model=build_model(a); model.load_state_dict(c['model_state_dict'],strict=True); del c
            model.to(device).eval()
            print('Predicting '+name,flush=True)
            p=predict_cci(model,smiles,tokenizer,transform,device,args.batch_size)
            np.save(cache,p); del model; gc.collect(); torch.cuda.empty_cache()
        ref=pd.read_csv(ROOT/f'outputs/top6_pool_target_power_ensemble/base_predictions/{refname}_test.csv')
        accept(name,p,ref[['sample_id','smiles',f'pred_{refname}']])
    feature_input=out/'gin_input_smiles.csv'
    pd.concat([test[['sample_id','smiles']],df[['sample_id','smiles']]],ignore_index=True).to_csv(feature_input,index=False)
    feature_path=out/'gin_features.pt'
    if feature_path.exists():
        meta=json.loads(feature_path.with_suffix('.json').read_text())
        assert meta['input_sha256']==digest(feature_input) and meta['features_sha256']==digest(feature_path)
    else:
        subprocess.run([args.gin_feature_python,str(ROOT/'code/generalization_gin_features_20260926.py'),str(feature_input),str(feature_path)],check=True)
    items=torch.load(feature_path,map_location='cpu',weights_only=False)
    assert [i['smiles'] for i in items]==smiles
    for seed,w in GIN_WEIGHTS.items():
        name=f'gin{seed}'; cache=out/f'raw_{name}.npy'
        if cache.exists(): p=np.load(cache)
        else:
            c=torch.load(ROOT/f'checkpoints/gnn/pure_gin_seed{seed}/best_pure_gin.pt',map_location='cpu',weights_only=False)
            model=PureGIN(in_dim=21); model.load_state_dict(c['model'],strict=True); del c
            model.to(device).eval(); print('Predicting '+name,flush=True)
            p=predict_gin(model,items,device,args.batch_size)
            np.save(cache,p); del model; gc.collect(); torch.cuda.empty_cache()
        ref=pd.read_csv(ROOT/f'checkpoints/gnn/pure_gin_seed{seed}/predictions_test_gnn.csv')
        accept(name,p,ref[['sample_id','smiles','pred_rt']])
    for name,p in predictions.items(): df['pred_'+name]=p
    df['pred_CCI_ensemble']=sum(m[3]*predictions[m[0]] for m in MEMBERS)
    df['pred_MCRL_frozen']=.30*df.pred_CCI_ensemble+sum(w*predictions[f'gin{s}'] for s,w in GIN_WEIGHTS.items())
    df['pred_SMRT_training_median']=float(np.median(train_rt))
    # Historical target-adapted OOF predictions: comparison only, never fed into the frozen model.
    adapted=[]
    for system in sorted(df.system.unique()):
        for pth in sorted((ROOT/'server_results/predret_compare_strict_20260909/proposed'/system).glob('fold_*/predictions_full_ensemble.csv')):
            d=pd.read_csv(pth); d['system']=system; d['fold']=pth.parent.name; adapted.append(d)
    ad=pd.concat(adapted,ignore_index=True)
    assert len(ad)==len(df) and not ad.duplicated(['system','sample_id']).any()
    df=df.merge(ad[['system','sample_id','smiles','true_rt','fold','pred_full_ensemble']],
        on=['system','sample_id'],how='left',validate='one_to_one',suffixes=('','_adapted'))
    assert df.pred_full_ensemble.notna().all()
    assert df.smiles.tolist()==df.smiles_adapted.tolist() and np.allclose(df.rt,df.true_rt,atol=.001)
    df=df.rename(columns={'pred_full_ensemble':'pred_MCRL_target_adapted_OOF'})
    df.to_csv(out/'predret_predictions.csv',index=False,encoding='utf-8-sig')
    rows=[]
    models=['MCRL_frozen','CCI_ensemble','gin2026','gin3407','gin777','SMRT_training_median','MCRL_target_adapted_OOF']
    for subset,mask in [('all',np.ones(len(df),bool)),('unseen_smrt_development',df.unseen_smrt_development),('unseen_smrt_all',df.unseen_smrt_all)]:
        sub=df.loc[mask]
        for system,d in list(sub.groupby('system'))+[('POOLED',sub)]:
            if not len(d): continue
            for name in models: rows.append(dict(subset=subset,system=system,model=name,**metrics(d.rt,d['pred_'+name])))
    scores=pd.DataFrame(rows); scores.to_csv(out/'predret_metrics.csv',index=False,encoding='utf-8-sig')
    macro=scores[scores.system!='POOLED'].groupby(['subset','model'])[['mae_s','rmse_s','r2','spearman']].mean().reset_index()
    macro.to_csv(out/'predret_system_macro_metrics.csv',index=False,encoding='utf-8-sig')
    # Paired uncertainty uses resampling within each system; no hyperparameter decisions.
    rng=np.random.default_rng(20260926); intervals=[]
    for system,d in df.groupby('system'):
        err=abs(d.pred_MCRL_frozen.to_numpy()-d.rt.to_numpy())
        boots=np.array([err[rng.integers(0,len(err),len(err))].mean() for _ in range(2000)])
        intervals.append(dict(system=system,n=len(err),mae_s=err.mean(),
                              bootstrap_mae_95_low_s=np.quantile(boots,.025),bootstrap_mae_95_high_s=np.quantile(boots,.975)))
    pd.DataFrame(intervals).to_csv(out/'frozen_mae_bootstrap_ci.csv',index=False,encoding='utf-8-sig')
    write_json(out/'completion.json',dict(status='complete',rows=len(df),systems=df.system.nunique(),
        source_reproduction_passed=all(c['passed'] for c in checks),target_fit=False,
        unseen_development=int(df.unseen_smrt_development.sum()),unseen_all=int(df.unseen_smrt_all.sum())))
    print(scores[(scores.subset=='all') & (scores.model=='MCRL_frozen')].to_string(index=False),flush=True)

if __name__=='__main__': main()
