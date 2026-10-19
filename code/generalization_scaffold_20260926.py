"""Complete the six-member scaffold experiment with epoch-level restart support.

--prepare audits inputs and freezes configurations; --run executes sequentially.
--worker NAME is launched by the supervisor. No SMRT-finetuned weights are loaded
into scaffold training. Existing scaffold-only GINs are verified and reused.
"""
import sys, os, json, time, random, gc, argparse, subprocess, functools, itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from generalization_frozen_20260926 import (MEMBERS, GIN_WEIGHTS, digest, write_json,
    canonical, graphs, predict_gin, metrics, np, pd, torch, Chem, PureGIN, AutoTokenizer, SimpleNamespace)
from rdkit import rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
import train_multimodal as tm
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

OUT=ROOT/'outputs/generalization_20260926/scaffold_top6'
OLD=ROOT/'outputs/paper_ablation_20260825/04_scaffold_generalization/scaffold_results'
ALL_MEMBERS=MEMBERS+[('exact777','exact777_best.pt','target_power_exact_repro_seed777',0.)]

def atomic_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');write_json(temp,obj);os.replace(temp,path)

def save_torch(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,path)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    partitions={}; all_frames=[]; audit={}; input_hashes={}
    for split in ['train','val','test']:
        p=OLD/f'data/scaffold_{split}.csv';d=pd.read_csv(p)
        assert d.sample_id.is_unique and (d.split==split).all()
        assert np.isfinite(d.rt).all()
        molecules=d.smiles.map(canonical)
        scaffold=d.smiles.map(lambda s: Chem.MolToSmiles(
            MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)),isomericSmiles=False))
        partitions[split]=(set(d.sample_id),set(molecules),set(scaffold))
        audit[split]=dict(n=len(d),n_scaffolds=scaffold.nunique(),acyclic=int((scaffold=='').sum()))
        dest=OUT/f'data/{split}.csv'
        if dest.exists(): assert digest(dest)==digest(p)
        else: dest.write_bytes(p.read_bytes())
        input_hashes[str(p.relative_to(ROOT))]=digest(p)
        all_frames.append(d)
    for a,b in itertools.combinations(partitions,2):
        overlaps=[len(x&y) for x,y in zip(partitions[a],partitions[b])]
        audit[a+'_'+b]=dict(zip(['sample_id_overlap','molecule_overlap','achiral_scaffold_overlap'],overlaps))
        assert not any(overlaps),f'Overlap {a}/{b}: {overlaps}'
    union=pd.concat(all_frames).set_index('sample_id').sort_index()
    source=pd.concat([pd.read_csv(ROOT/f'data/unified/{s}.csv') for s in ['train','val','test']]).set_index('sample_id').sort_index()
    assert union.index.is_unique and union.index.equals(source.index)
    assert union.smiles.equals(source.smiles) and np.allclose(union.rt,source.rt)
    write_json(OUT/'split_audit.json',audit)
    configs={}
    for name,filename,ref,w in ALL_MEMBERS:
        c=torch.load(ROOT/'FINAL_RELEASE_SCHEME_A/checkpoints'/filename,map_location='cpu',weights_only=False)
        a=dict(c['args']); del c
        # Metadata only: no source-fitted model or optimizer state enters training.
        a.update(train_data=str(OUT/'data/train.csv'),val_data=str(OUT/'data/val.csv'),test_data=str(OUT/'data/test.csv'),
                 smiles_model=str(ROOT/'pretrained/ChemBERTa-zinc-base-v1'),smiles_backbone='custom',
                 output_dir=str(OUT/name),checkpoint_dir=str(OUT/name),prediction_output=None,
                 resume_checkpoint=None,eval_only_checkpoint=None,device='cuda',num_workers=0)
        configs[name]=a
        (OUT/name).mkdir(exist_ok=True)
        cp=OUT/name/'config.json'
        if cp.exists(): assert json.loads(cp.read_text())==a
        else:write_json(cp,a)
    gin_provenance={}
    for seed in GIN_WEIGHTS:
        p=OLD/f'gnn/seed{seed}/best_pure_gin.pt';c=torch.load(p,map_location='cpu',weights_only=False)
        a=c['args']
        for split in ['train','val','test']:
            assert (ROOT/a[split+'_data']).resolve()==(OLD/f'data/scaffold_{split}.csv').resolve()
        assert a['seed']==seed and a['epochs']==180 and a['patience']==25 and a['layers']==5 and a['hidden']==256
        gin_provenance[str(seed)]=dict(path=str(p.relative_to(ROOT)),sha256=digest(p),args=a,best_epoch=c['epoch'])
    protocol=dict(experiment='Scaffold-disjoint complete MCRL candidate architecture',
        candidate_count=dict(cci_prediction_models=6,gin_models=3),
        initialization='ChemBERTa unsupervised pretrained backbone; remaining modules freshly initialized',
        checkpoint_selection='Scaffold validation MAE only; max 200 epochs, patience 20',
        source_checkpoint_usage='Read hyperparameters only; source-fitted weights and optimizer states NEVER reused',
        target_transform='PowerTransformer fitted exclusively on scaffold train RT',
        primary_fusion='Two-stage nonnegative MAE linear program, sum weights=1, validation only; same rule re-evaluated on random-split saved predictions',
        secondary_fusion='Historical fixed source coefficients, reported separately from validation-fitted primary comparison',
        test_used_for_selection=False,rdkit=rdBase.rdkitVersion,input_hashes=input_hashes,
        gin_provenance=gin_provenance,configs=configs,
        resume='Last completed epoch incl optimizer, scheduler, torch/numpy/python/dataset RNG; RDKit augmentation seeded by epoch. Bitwise equivalence across interrupted runs is not guaranteed.',
        limitations='Single predefined scaffold partition; hyperparameters and candidate identities inherited from earlier SMRT development. This is not a prospective external holdout.')
    p=OUT/'protocol.json'
    if p.exists(): assert json.loads(p.read_text(encoding='utf-8'))==protocol
    else:write_json(p,protocol)
    print('PREPARED',json.dumps(audit),flush=True)

def verify_gin():
    checks=[]
    for seed in GIN_WEIGHTS:
        c=torch.load(OLD/f'gnn/seed{seed}/best_pure_gin.pt',map_location='cpu',weights_only=False)
        model=PureGIN(in_dim=21);model.load_state_dict(c['model']);del c
        model.cuda().eval()
        for split in ['val','test']:
            d=pd.read_csv(OUT/f'data/{split}.csv')
            ref=pd.read_csv(OLD/f'gnn/seed{seed}/predictions_{split}_gnn.csv').set_index('sample_id')
            assert ref.index.is_unique and set(ref.index)==set(d.sample_id)
            r=ref.loc[d.sample_id];assert r.smiles.tolist()==d.smiles.tolist()
            assert np.allclose(r.true_rt,d.rt,atol=.001)
            idx=np.linspace(0,len(d)-1,64,dtype=int)
            pred=predict_gin(model,graphs(d.iloc[idx].smiles.tolist()),'cuda',16)
            delta=abs(pred-r.iloc[idx].pred_rt.to_numpy())
            checks.append(dict(seed=seed,split=split,max_abs_difference_s=float(delta.max()),passed=bool(delta.max()<.05)))
            write_json(OUT/'gin_reproduction_checks.json',checks)
            assert delta.max()<.05, checks[-1]
        del model;gc.collect();torch.cuda.empty_cache()
    print('GIN verification passed',flush=True)

def worker(name):
    target=OUT/name
    a=SimpleNamespace(**json.loads((target/'config.json').read_text()))
    torch.set_num_threads(4);tm.set_seed(a.seed)
    # Limit log volume while preserving periodic batch throughput diagnostics.
    tm.tqdm=functools.partial(tm.tqdm,mininterval=30,maxinterval=60)
    transform=tm.build_target_transformer(a);a.tail_rt_threshold=tm.resolve_tail_rt_threshold(a)
    tokenizer=AutoTokenizer.from_pretrained(a.smiles_model,local_files_only=True)
    tr,va,te=tm.build_dataloaders(a,tokenizer,transform)
    model=tm.build_model(a).cuda()
    optimizer=tm.build_optimizer(a,model)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3)
    align,fusion=tm.build_loss_fns(a)
    start=1;best=float('inf');bad=0;history=[]
    last=target/'last.pt';best_path=target/'best.pt'
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
        start=c['epoch']+1;best=c['best'];bad=c['bad'];history=c['history']
        random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'])
        torch.cuda.set_rng_state_all(c['cuda_rng']);tr.dataset._smiles_rng.setstate(c['augmentation_rng']);del c
        print(f'RESUME {name} epoch {start}',flush=True)
    else: print(f'FRESH INITIALIZATION {name}; no SMRT-fitted state loaded',flush=True)
    for epoch in range(start,a.epochs+1):
        if bad>=a.early_stopping_patience:break
        rdBase.SeedRandomNumberGenerator(a.seed+epoch)
        began=time.time();atomic_json(target/'status.json',dict(status='training',member=name,epoch=epoch,pid=os.getpid()))
        train=tm.train_one_epoch(model,tr,optimizer,align,fusion,a,epoch)
        val=tm.evaluate(model,va,align,fusion,a,desc=f'Validation {epoch}',target_transformer=transform)
        assert np.isfinite(val['mae'])
        scheduler.step(val['mae'])
        if val['mae']<best:
            best=val['mae'];bad=0
            save_torch(best_path,dict(model_state_dict=model.state_dict(),args=vars(a),epoch=epoch,metrics=val))
        else:bad+=1
        history.append(dict(epoch=epoch,seconds=time.time()-began,train=train,val=val,best_val_mae=best,bad_epochs=bad))
        save_torch(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,best=best,bad=bad,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),augmentation_rng=tr.dataset._smiles_rng.getstate()))
        atomic_json(target/'history.json',history)
        print(f'{name} epoch={epoch} val_mae={val["mae"]:.6f} best={best:.6f} elapsed={time.time()-began:.1f}s',flush=True)
    c=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(c['model_state_dict'])
    for split,loader in [('val',va),('test',te)]:
        tm.save_test_predictions(model,loader,a,target/f'predictions_{split}.csv',tokenizer,use_tta=False,target_transformer=transform)
    atomic_json(target/'complete.json',dict(status='complete',best_epoch=c['epoch'],best_val_mae=best,epochs_completed=len(history)))
    atomic_json(target/'status.json',dict(status='complete',member=name,pid=os.getpid()))

def simplex_mae(y,p):
    n,k=p.shape;ident=eye(n,format='csr');sp=csr_matrix(p)
    res=linprog(np.r_[np.zeros(k),np.ones(n)/n],A_ub=vstack([hstack([sp,-ident]),hstack([-sp,-ident])]),
        b_ub=np.r_[y,-y],A_eq=csr_matrix(np.r_[np.ones(k),np.zeros(n)].reshape(1,-1)),b_eq=[1.],
        bounds=[(0,None)]*(n+k),method='highs')
    if not res.success:raise RuntimeError(res.message)
    w=np.clip(res.x[:k],0,1);return w/w.sum()

def prediction_matrix(partition,split):
    data=pd.read_csv(OUT/f'data/{split}.csv' if partition=='scaffold' else ROOT/f'data/unified/{split}.csv')
    values=[]
    for name,filename,ref,w in ALL_MEMBERS:
        p=(OUT/name/f'predictions_{split}.csv') if partition=='scaffold' else ROOT/f'outputs/top6_pool_target_power_ensemble/base_predictions/{ref}_{split}.csv'
        r=pd.read_csv(p).set_index('sample_id').loc[data.sample_id]
        assert r.smiles.tolist()==data.smiles.tolist() and np.allclose(r.true_rt,data.rt,atol=.001)
        values.append(r['pred_rt' if partition=='scaffold' else f'pred_{ref}'].to_numpy())
    for seed in GIN_WEIGHTS:
        p=OLD/f'gnn/seed{seed}/predictions_{split}_gnn.csv' if partition=='scaffold' else ROOT/f'checkpoints/gnn/pure_gin_seed{seed}/predictions_{split}_gnn.csv'
        r=pd.read_csv(p).set_index('sample_id').loc[data.sample_id]
        assert r.smiles.tolist()==data.smiles.tolist() and np.allclose(r.true_rt,data.rt,atol=.001)
        values.append(r.pred_rt.to_numpy())
    mat=np.stack(values,axis=1);assert np.isfinite(mat).all()
    return data,mat

def finalize():
    rows=[];weights={}
    for partition in ['random','scaffold']:
        val,pv=prediction_matrix(partition,'val')
        w1=simplex_mae(val.rt.to_numpy(),pv[:,:6]);fval=np.column_stack([pv[:,:6]@w1,pv[:,6:]])
        w2=simplex_mae(val.rt.to_numpy(),fval)
        weights[partition]=dict(cci=dict(zip([m[0] for m in ALL_MEMBERS],w1.tolist())),
            ptdf=dict(zip(['CCI_ensemble']+[f'GIN{s}' for s in GIN_WEIGHTS],w2.tolist())),selection_split='val')
        # Freeze weights on disk BEFORE accessing test predictions.
        write_json(OUT/'validation_fusion_weights.json',weights)
        for split in ['val','test']:
            d,p=prediction_matrix(partition,split)
            pred=np.column_stack([p[:,:6]@w1,p[:,6:]])@w2
            fixed=.30*(p[:,:6]@np.array([m[3] for m in ALL_MEMBERS]))+p[:,6:]@np.array(list(GIN_WEIGHTS.values()))
            d['pred_validation_fitted']=pred;d['pred_historical_fixed_coefficients']=fixed
            d.to_csv(OUT/f'{partition}_{split}_complete_mcrl.csv',index=False)
            for label,pp in [('matched_validation_fusion',pred),('historical_fixed_coefficients',fixed)]:
                rows.append(dict(partition=partition,split=split,protocol=label,**metrics(d.rt,pp)))
    pd.DataFrame(rows).to_csv(OUT/'random_vs_scaffold_metrics.csv',index=False)
    atomic_json(OUT/'queue_status.json',dict(status='complete',completed=[m[0] for m in ALL_MEMBERS],pid=os.getpid()))

def run():
    assert (OUT/'protocol.json').exists(),'Run --prepare first'
    # Avoid concurrent supervisors writing the same checkpoints.
    import msvcrt
    lock=open(OUT/'queue.lock','a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for p,h in protocol['input_hashes'].items(): assert digest(ROOT/p)==h
    for item in protocol['gin_provenance'].values():assert digest(ROOT/item['path'])==item['sha256']
    torch.set_num_threads(4);verify_gin()
    completed=[]
    for name,*_ in ALL_MEMBERS:
        if not (OUT/name/'complete.json').exists():
            atomic_json(OUT/'queue_status.json',dict(status='running',current=name,completed=completed,pid=os.getpid(),
                remaining=[m[0] for m in ALL_MEMBERS if m[0] not in completed]))
            with open(OUT/name/'train.log','ab',buffering=0) as log:
                rc=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if rc:
                atomic_json(OUT/'queue_status.json',dict(status='failed',current=name,returncode=rc,completed=completed,pid=os.getpid()))
                raise RuntimeError(f'{name} failed with exit code {rc}; inspect train.log and restart --run')
        completed.append(name)
    finalize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
    elif a.run:run()
    else:p.error('Choose --prepare, --run or --worker NAME')
