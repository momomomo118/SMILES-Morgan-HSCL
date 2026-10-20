"""Standalone SMRT MCRL point prediction with train-fitted inverse target transform.

This entry point is syntax-checked but has not been dynamically validated in the
preparation runtime. It does not train models, evaluate targets, or use RT labels
for input molecules. Exact recovery of archived predictions is not established.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]

def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--smiles')
    src.add_argument('--input-csv',type=Path,help='Columns: smiles, optional sample_id; RT labels are not read.')
    ap.add_argument('--model-root',type=Path,default=REPO)
    ap.add_argument('--train-csv',type=Path,default=REPO/'data/unified/train.csv')
    ap.add_argument('--pretrained-dir',type=Path)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace existing output: '+str(args.output))
    # Heavy imports occur after parsing, so --help remains available without ML packages.
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.preprocessing import PowerTransformer
    from transformers import AutoTokenizer
    sys.path[:0]=[str(REPO/'code/src'),str(REPO/'code')]
    from train_multimodal import build_model
    from train_pure_gnn_for_ensemble import PureGIN,collate,smiles_to_graph
    pre=args.pretrained_dir or args.model_root/'pretrained/ChemBERTa-zinc-base-v1'
    tokenizer=AutoTokenizer.from_pretrained(str(pre),local_files_only=True)
    y_train=pd.read_csv(args.train_csv)['rt'].astype(float).to_numpy().reshape(-1,1)
    transform=PowerTransformer(method='yeo-johnson',standardize=True).fit(y_train)
    manifest=json.loads((REPO/'configs/scheme_a/model_artifacts.json').read_text(encoding='utf-8'))
    models=[]
    for member in manifest['members']:
        if not member['active']:
            continue
        path=args.model_root/member['path']
        if file_sha256(path)!=member['sha256']:
            raise ValueError('Checkpoint hash mismatch: '+member['path'])
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if member['family']=='CCI':
            saved=dict(payload['args']);saved['smiles_model']=str(pre)
            model=build_model(SimpleNamespace(**saved))
            model.load_state_dict(payload['model_state_dict'],strict=True)
        else:
            model=PureGIN(in_dim=21,hidden=256,layers=5,desc_dim=10,dropout=.15)
            model.load_state_dict(payload['model'],strict=True)
        model.eval().to(args.device)
        models.append((member,model))
        del payload
    if args.smiles:
        inputs=[{'sample_id':'query_000001','smiles':args.smiles.strip()}]
    else:
        with args.input_csv.open(encoding='utf-8-sig',newline='') as f:
            inputs=[{'sample_id':r.get('sample_id') or 'query_'+str(i+1).zfill(6),'smiles':r['smiles'].strip()} for i,r in enumerate(csv.DictReader(f))]
    results=[]
    with torch.inference_mode():
        for row in inputs:
            smi=row['smiles'];mol=Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError('Invalid molecular input: '+row['sample_id'])
            enc=tokenizer(smi,truncation=True,padding='max_length',max_length=256,return_tensors='pt')
            fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=True)
            morgan=torch.tensor(list(fp),dtype=torch.float32).unsqueeze(0).to(args.device)
            x,adj,desc=smiles_to_graph(smi)
            batch=collate([{'sample_id':row['sample_id'],'smiles':smi,'x':torch.tensor(x),'adj':torch.tensor(adj),'desc':torch.tensor(desc),'y':torch.tensor(0.)}])
            cci=[];graph=[];components={}
            for member,model in models:
                if member['family']=='CCI':
                    pred=model(input_ids=enc['input_ids'].to(args.device),attention_mask=enc['attention_mask'].to(args.device),morgan_fp=morgan).prediction
                    value=float(transform.inverse_transform(pred.detach().cpu().numpy().reshape(-1,1))[0,0])
                    cci.append(member['inner_weight']*value)
                else:
                    value=float(model(batch['x'].to(args.device),batch['adj'].to(args.device),batch['mask'].to(args.device),batch['desc'].to(args.device)).item())
                    graph.append(member['outer_weight']*value)
                components[member['member']]=value
            cci_prediction=sum(cci)
            results.append(dict(row,cci_ensemble_s=cci_prediction,pred_mcrl_s=.30*cci_prediction+sum(graph),**components))
    if not results:
        raise ValueError('No molecular inputs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print('Wrote',len(results),'new point predictions to',args.output)
    print('This is SMRT-model inference, not target-system PredRet adaptation.')

if __name__=='__main__':
    main()
