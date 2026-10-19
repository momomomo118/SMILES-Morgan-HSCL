"""Read-only historical audit; writes new audit artifacts, never trains."""
import os, sys, json, csv, re, zipfile, pickle, io, hashlib
from collections import OrderedDict, Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/simplified_mcrl_audit_20261011'
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
SKIP = {'.git', 'node_modules', '__pycache__', '.venv', 'site-packages',
        'figure2_plot_deps', 'generalization_runtime_rdkit202409',
        'generalization_runtime_rdkit2025', 'generalization_runtime_rdkit202503',
        'generalization_runtime_rdkit202603_py310', 'simplified_mcrl_audit_20261011'}

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    assert len(y) == len(p) and np.isfinite(y).all() and np.isfinite(p).all()
    e = p-y
    return dict(n=len(y), mae_s=float(np.mean(abs(e))), rmse_s=float(np.sqrt(np.mean(e*e))),
                medae_s=float(np.median(abs(e))), mape_pct=float(np.mean(abs(e)/abs(y))*100),
                r2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
                pearson=float(np.corrcoef(y,p)[0,1]))

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inert(*args, **kwargs): return None
class MetadataReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict': return OrderedDict
        if module.startswith('torch'): return inert
        raise pickle.UnpicklingError(f'Unsupported global {module}.{name}')
    def persistent_load(self,pid): return None

def main():
    files=[]; checkpoints=[]; archives=[]; hits=[]
    pattern=re.compile(r'CCRL|MCRL|CCI|PTDF|supcr|contrastive|cross.attention|fusion|retention',re.I)
    for parent,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in SKIP and not d.endswith('.dist-info')]
        for name in names:
            p=Path(parent)/name; rel=p.relative_to(ROOT).as_posix(); ext=p.suffix.lower()
            category=None
            if ext in {'.pt','.pth','.ckpt'}: category='checkpoint_or_tensor_cache'
            elif 'tfevents' in name.lower(): category='tensorboard'
            elif ext in {'.log','.out','.err'}: category='log'
            elif ext in {'.csv','.json','.txt','.md','.yaml','.yml'}: category='metrics_prediction_config_report_or_data'
            elif ext=='.zip': category='archive'
            if category is None: continue
            files.append(dict(path=rel,category=category,size_bytes=p.stat().st_size))
            if category=='checkpoint_or_tensor_cache':
                row=dict(path=rel,size_bytes=p.stat().st_size)
                try:
                    with zipfile.ZipFile(p) as z:
                        pk=next(n for n in z.namelist() if n.endswith('/data.pkl'))
                        m=MetadataReader(io.BytesIO(z.read(pk))).load()
                    if isinstance(m,dict):
                        row.update(keys=list(m),epoch=m.get('epoch'),args=m.get('args'),config=m.get('config'),metrics=m.get('metrics'))
                        state=m.get('model_state_dict',m.get('model',m.get('state_dict')))
                        if isinstance(state,dict): row['state_key_prefixes']=dict(Counter(k.split('.')[0] for k in state))
                        cfg=row.get('args') or row.get('config') or {}
                        if isinstance(cfg,dict) and cfg.get('fusion') in ('cross_attention','cross_attention_residual'):
                            row['class']='B_candidate' if not any(k.startswith(('gnn','graph','descriptor')) for k in (state or {})) else 'mixed_candidate'
                        elif isinstance(state,dict) and any(k.startswith('convs') for k in state): row['class']='graph_branch'
                        else: row['class']='other_model_or_cache'
                    else: row['class']='tensor_cache_or_other'
                except Exception as e: row['metadata_error']=str(e)
                checkpoints.append(row)
            elif category=='archive':
                try:
                    with zipfile.ZipFile(p) as z:
                        for member in z.infolist():
                            if re.search(r'\.(pt|pth|ckpt|json|csv|log|out|txt|md)$|tfevents',member.filename,re.I):
                                archives.append(dict(archive=rel,member=member.filename,size_bytes=member.file_size))
                except Exception as e: archives.append(dict(archive=rel,error=str(e)))
            elif ext in {'.json','.txt','.md','.log','.out','.err','.yaml','.yml'}:
                # Stream each textual experiment record. Save one bounded hit per file.
                try:
                    with open(p,encoding='utf-8',errors='replace') as f:
                        for number,line in enumerate(f,1):
                            match=pattern.search(line)
                            if match:
                                hits.append(dict(path=rel,line=number,keyword=match.group(0),excerpt=line.strip()[:350]));break
                except OSError: pass
    pd.DataFrame(files).to_csv(OUT/'file_inventory.csv',index=False)
    pd.DataFrame(archives).to_csv(OUT/'archive_inventory.csv',index=False)
    pd.DataFrame(hits).to_csv(OUT/'record_keyword_hits.csv',index=False)
    (OUT/'checkpoint_metadata.json').write_text(json.dumps(checkpoints,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    refs={s:pd.read_csv(ROOT/f'data/unified/{s}.csv').set_index('sample_id') for s in ['test','val']}
    scores=[]; rejected=[]
    for row in files:
        rel=row['path']
        if not rel.endswith('.csv') or not re.search('pred|result',rel,re.I): continue
        p=ROOT/rel
        try:
            header=pd.read_csv(p,nrows=0).columns
            if not {'sample_id','smiles'}.issubset(header):
                rejected.append(dict(path=rel,reason='No sample_id/smiles: cannot establish exact current split'));continue
            predcols=[c for c in header if c.lower().startswith('pred') and c not in ['prediction_source','prediction_path']]
            if not predcols: continue
            df=pd.read_csv(p)
            if not df.sample_id.is_unique: raise ValueError('duplicate sample_id')
            split=next((s for s,r in refs.items() if len(df)==len(r) and set(df.sample_id)==set(r.index)),None)
            if split is None: raise ValueError('not exactly current unified validation/test sample set')
            r=refs[split].loc[df.sample_id]
            if df.smiles.tolist()!=r.smiles.tolist(): raise ValueError('SMILES mismatch')
            truth=next((c for c in ['true_rt','rt','y_true'] if c in df),None)
            if truth and not np.allclose(df[truth],r.rt,atol=.00015,rtol=0): raise ValueError('RT label mismatch')
            for col in predcols:
                if not pd.api.types.is_numeric_dtype(df[col]): continue
                scores.append(dict(path=rel,column=col,split=split,**metrics(r.rt,df[col])))
        except Exception as e: rejected.append(dict(path=rel,reason=str(e)))
    pd.DataFrame(scores).to_csv(OUT/'historical_current_split_metrics.csv',index=False)
    pd.DataFrame(rejected).to_csv(OUT/'prediction_alignment_rejections.csv',index=False)
    summary=dict(file_counts=dict(Counter(r['category'] for r in files)),checkpoint_classes=dict(Counter(r.get('class','metadata_error') for r in checkpoints)),
                 archived_records=len(archives),keyword_record_hits=len(hits),current_split_prediction_columns=len(scores),
                 prediction_files_not_usable_for_current_split=len(rejected),
                 split_sha256={s:digest(ROOT/f'data/unified/{s}.csv') for s in ['train','val','test']},
                 excluded_nonexperiment_dependency_directories=sorted(SKIP))
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    table=pd.DataFrame(scores)
    print(table[table.path.str.contains('base_predictions|runs/E_residual') & (table.split=='val')][['path','column','mae_s','rmse_s','r2']].to_string(index=False))
if __name__=='__main__': main()
