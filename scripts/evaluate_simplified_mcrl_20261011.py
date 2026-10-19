"""Evaluate existing CCI checkpoints on current SMRT test; no training."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys, json, gc, time, platform
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/generalization_runtime_rdkit2025'),str(ROOT/'code/src'),str(ROOT/'scripts')]
import torch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from transformers import AutoTokenizer
from sklearn.preprocessing import PowerTransformer
from train_multimodal import build_model
from audit_simplified_mcrl_20261011 import metrics,digest

OUT=ROOT/'outputs/simplified_mcrl_audit_20261011'
MEMBERS=[
 ('target_power_exact_repro_seed1234','checkpoints/supcr_tuning_aliases/exact1234_best.pt',.30),
 ('target_power_div_dropout015_seed777','checkpoints/supcr_tuning/target_power_div_dropout015_seed777/best_cross_attention_residual.pt',.25),
 ('target_power_div_lowfusion_seed777','checkpoints/supcr_tuning/target_power_div_lowfusion_seed777/best_cross_attention_residual.pt',.10),
 ('target_power_div_aug050_seed777','checkpoints/supcr_tuning/target_power_div_aug050_seed777/best_cross_attention_residual.pt',.20),
 ('target_power_exact_repro_seed2025','checkpoints/supcr_tuning_aliases/exact2025_best.pt',.15),
 ('target_power_exact_repro_seed777','checkpoints/supcr_tuning_aliases/exact777_best.pt',0.),
]

def save_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    test=pd.read_csv(ROOT/'data/unified/test.csv')
    train=pd.read_csv(ROOT/'data/unified/train.csv')
    transform=PowerTransformer(method='yeo-johnson',standardize=True).fit(train.rt.to_numpy(float).reshape(-1,1))
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'pretrained/ChemBERTa-zinc-base-v1'),local_files_only=True)
    print(f'Preparing {len(test)} test inputs; device={device}; rdkit={rdBase.rdkitVersion}',flush=True)
    tokens=tokenizer(test.smiles.tolist(),truncation=True,padding='max_length',max_length=256,add_special_tokens=True,return_tensors='pt')
    fps=[]
    for s in test.smiles:
        mol=Chem.MolFromSmiles(s)
        assert mol is not None,'Invalid SMILES'
        fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=True)))
    fps=torch.tensor(np.asarray(fps,dtype=np.float32))
    combined=test[['sample_id','smiles']].copy();combined['true_rt']=test.rt
    rows=[]; checks=[]
    protocol=dict(training_performed=False,test_split='data/unified/test.csv',n_test=len(test),
      target_transform_fit='data/unified/train.csv only; float64; Yeo-Johnson + standardization',
      transform=dict(lambdas=transform.lambdas_.tolist(),mean=transform._scaler.mean_.tolist(),scale=transform._scaler.scale_.tolist()),
      tokenizer='pretrained/ChemBERTa-zinc-base-v1',max_length=256,morgan=dict(radius=2,n_bits=2048,use_chirality=True),
      original_smiles_preserved=True,test_augmentation=False,tta=1,batch_size=32,dtype='float32',device=device,
      versions=dict(python=platform.python_version(),torch=torch.__version__,rdkit=rdBase.rdkitVersion),
      split_sha256={s:digest(ROOT/f'data/unified/{s}.csv') for s in ['train','val','test']},
      script_sha256=digest(Path(__file__)),members=[])
    save_json('evaluation_protocol.json',protocol)
    for name,relative,w in MEMBERS:
        started=time.time();path=ROOT/relative
        print(f'Loading {name}, fixed ensemble weight={w}',flush=True)
        c=torch.load(path,map_location='cpu',weights_only=False)
        args=dict(c['args']);a=SimpleNamespace(**args)
        a.smiles_model=str(ROOT/'pretrained/ChemBERTa-zinc-base-v1')
        assert a.fusion=='cross_attention_residual' and a.target_transform=='power' and a.max_length==256
        assert a.tta_smiles==1 and a.morgan_mode=='normal'
        model=build_model(a);model.load_state_dict(c['model_state_dict'],strict=True)
        prefixes=sorted(set(k.split('.')[0] for k in c['model_state_dict']))
        assert not any('gnn' in k.lower() or 'descriptor' in k.lower() for k in c['model_state_dict'])
        protocol['members'].append(dict(name=name,checkpoint=relative,sha256=digest(path),epoch=c['epoch'],ensemble_weight=w,args=args,state_key_prefixes=prefixes))
        del c;gc.collect();model.to(device).eval()
        predictions=[]
        with torch.inference_mode():
            for start in range(0,len(test),32):
                end=start+32
                raw=model(input_ids=tokens['input_ids'][start:end].to(device),attention_mask=tokens['attention_mask'][start:end].to(device),morgan_fp=fps[start:end].to(device)).prediction.cpu().numpy().reshape(-1,1)
                predictions.extend(transform.inverse_transform(raw).reshape(-1).tolist())
                if start==0 or (start//32+1)%40==0:
                    print(f'{name}: {min(end,len(test))}/{len(test)}, elapsed={time.time()-started:.1f}s',flush=True)
        p=np.asarray(predictions,float)
        col='pred_'+name
        combined[col]=p
        one=combined[['sample_id','smiles','true_rt',col]]
        one.to_csv(OUT/f'predictions_test_reevaluated_{name}.csv',index=False)
        historical=pd.read_csv(ROOT/f'outputs/top6_pool_target_power_ensemble/base_predictions/{name}_test.csv').set_index('sample_id').loc[test.sample_id]
        assert historical.smiles.tolist()==test.smiles.tolist()
        delta=abs(p-historical[col].to_numpy(float))
        check=dict(model=name,max_abs_difference_s=float(delta.max()),mean_abs_difference_s=float(delta.mean()),within_0_05s=bool(delta.max()<.05))
        checks.append(check)
        row=dict(model=name,source='checkpoint reevaluation',checkpoint=relative,ensemble_weight=w,**metrics(test.rt,p));rows.append(row)
        print(json.dumps(row),flush=True);print(json.dumps(check),flush=True)
        save_json('historical_prediction_reproduction_checks.json',checks)
        save_json('evaluation_protocol.json',protocol)
        pd.DataFrame(rows).to_csv(OUT/'reevaluated_test_metrics.csv',index=False)
        del model;gc.collect();torch.cuda.empty_cache()
    combined['pred_cci_ensemble']=sum(w*combined['pred_'+name] for name,_,w in MEMBERS)
    rows.append(dict(model='CCI Top6 ensemble without PTDF',source='checkpoint reevaluation; original fixed weights',**metrics(test.rt,combined.pred_cci_ensemble)))
    final=pd.read_csv(ROOT/'outputs/final_fusion_ccrl_gnn/predictions_test_final_ccrl_gnn_ptdf.csv').set_index('sample_id').loc[test.sample_id]
    assert final.smiles.tolist()==test.smiles.tolist()
    assert np.allclose(final.true_rt,test.rt,atol=.00015,rtol=0)
    combined['pred_full_mcrl']=final.pred_final_ccrl_gnn_ptdf.to_numpy(float)
    rows.append(dict(model='Full MCRL Scheme A',source='saved final predictions recomputed',**metrics(test.rt,combined.pred_full_mcrl)))
    original=pd.read_csv(ROOT/'outputs/top6_pool_target_power_ensemble/predictions_test_calibrated.csv').set_index('sample_id').loc[test.sample_id]
    delta=abs(combined.pred_cci_ensemble.to_numpy()-original.pred_ensemble.to_numpy())
    checks.append(dict(model='CCI Top6 ensemble',max_abs_difference_s=float(delta.max()),mean_abs_difference_s=float(delta.mean()),within_0_05s=bool(delta.max()<.05)))
    combined.to_csv(OUT/'predictions_test_comparison.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'reevaluated_test_metrics.csv',index=False)
    save_json('historical_prediction_reproduction_checks.json',checks)
    print('EVALUATION COMPLETE',flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__': main()
