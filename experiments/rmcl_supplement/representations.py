"""Matched checkpoint inference, quantitative geometry, and explicitly exploratory plots."""
import gc
from pathlib import Path
import numpy as np
from common import ROOT, SEEDS, read_csv, selected_assets, write_csv, write_json, digest


def geometry(features, y, pairs):
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    x=features.astype(float);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    dist=1-(x[pairs[:,0]]*x[pairs[:,1]]).sum(1)
    rho=float(spearmanr(dist,abs(y[pairs[:,0]]-y[pairs[:,1]])).statistic)
    neighbors=NearestNeighbors(n_neighbors=min(11,len(x)),metric='cosine').fit(x).kneighbors(x,return_distance=False)
    errors=[]
    for i,ns in enumerate(neighbors):
        ns=ns[ns!=i][:10]
        errors.append(float(np.mean(abs(y[ns]-y[i]))))
    return dict(pairwise_cosine_RT_spearman=rho if np.isfinite(rho) else None,
                neighbor_RT_absolute_distance=float(np.mean(errors)),
                mean_feature_std=float(features.std(axis=0).mean()))


def plot_panels(arrays, y, titles, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    fig,axes=plt.subplots(1,len(arrays),figsize=(5*len(arrays),4),squeeze=False)
    for ax,x,title in zip(axes[0],arrays,titles):
        # Each panel has an independent fit. Coordinates/cluster gaps cannot be compared across panels.
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        x=PCA(n_components=min(50,len(x)-1,x.shape[1]),random_state=20261015).fit_transform(x)
        z=TSNE(n_components=2,perplexity=min(30.,(len(x)-1)/3),init='pca',learning_rate='auto',random_state=20261015).fit_transform(x)
        points=ax.scatter(z[:,0],z[:,1],c=y,s=5,cmap='viridis',vmin=y.min(),vmax=y.max(),alpha=.7)
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(points,ax=axes.ravel().tolist(),label='Experimental RT (s)',shrink=.8)
    fig.suptitle('Exploratory t-SNE; independent panel fits; geometry metrics required',fontsize=10)
    fig.savefig(destination,dpi=200,bbox_inches='tight');plt.close(fig)


def main(args):
    from verify_package import preflight
    preflight(args.device)
    import runtime as rt
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from sklearn.preprocessing import PowerTransformer
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to((ROOT/'results').resolve()):raise ValueError('Output must stay under results/')
    truth=read_csv(ROOT/'data/unified/test.csv')
    if not 30<=args.samples<=len(truth):raise ValueError('samples must be between 30 and test size')
    rng=np.random.default_rng(20261015)
    idx=np.sort(rng.choice(len(truth),size=args.samples,replace=False))
    ids=[truth[i]['sample_id'] for i in idx]
    y=np.asarray([float(truth[i]['rt']) for i in idx])
    pairs=rng.integers(0,len(idx),size=(20000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'selected_samples.csv',[dict(sample_id=truth[i]['sample_id'],smiles=truth[i]['smiles'],rt=truth[i]['rt']) for i in idx])
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'pretrained/ChemBERTa-zinc-base-v1'),local_files_only=True)
    train=read_csv(ROOT/'data/unified/train.csv')
    transform=PowerTransformer().fit(np.asarray([float(r['rt']) for r in train]).reshape(-1,1))
    ds=rt.legacy.MultimodalRTDataset(ROOT/'data/unified/test.csv',tokenizer=tokenizer,max_length=256,target_transformer=transform)
    loader=DataLoader(torch.utils.data.Subset(ds,idx.tolist()),batch_size=args.batch_size,shuffle=False,num_workers=0)
    quantitative=[]; missing=[]
    for seed in args.seeds:
        panels={}
        for variant in ['base','cma','rscl','full']:
            a=selected_assets(variant,seed)
            if a is None:
                missing.append(dict(variant=variant,seed=seed,reason='DATA REQUIRED'));continue
            if not a['checkpoint'].exists():raise FileNotFoundError(f'{a["checkpoint"]}: unzip the checkpoint pack')
            cfg=rt.config_for(variant,seed,args.device,a['config'])
            rt.legacy.set_seed(seed)
            model=rt.build_model(cfg)
            checkpoint=rt.torch_load(a['checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            del checkpoint
            model.to(args.device).eval()
            collected={k:[] for k in ['smiles_pooled','morgan_pooled','fusion_pooled','z_smiles','z_morgan','z_fusion']}
            with torch.inference_mode():
                for batch in loader:
                    inputs={k:batch[k].to(args.device) for k in ['input_ids','attention_mask','morgan_fp']}
                    pred=model(**inputs)
                    for k in collected:
                        v=pred[k]
                        if k=='fusion_pooled' and v is None:v=torch.cat([pred['smiles_pooled'],pred['morgan_pooled']],dim=-1)
                        if v is not None:collected[k].append(v.cpu().numpy())
            features={k:np.concatenate(v) for k,v in collected.items() if v}
            np.savez_compressed(out/f'{a["run_id"]}_features.npz',sample_id=np.asarray(ids),rt=y,**features)
            for name in ['smiles_pooled','morgan_pooled','fusion_pooled']:
                quantitative.append(dict(variant=variant,seed=seed,feature=name,origin=a['origin'],**geometry(features[name],y,pairs)))
            zs=features['z_smiles'];zm=features['z_morgan']
            zs=zs/np.maximum(np.linalg.norm(zs,axis=1,keepdims=True),1e-12)
            zm=zm/np.maximum(np.linalg.norm(zm,axis=1,keepdims=True),1e-12)
            sim=zs@zm.T;matched=np.diag(sim);unmatched=(sim.sum()-matched.sum())/(len(y)*(len(y)-1))
            quantitative.append(dict(variant=variant,seed=seed,feature='cross_view_projection',origin=a['origin'],
                matched_cosine=float(matched.mean()),unmatched_cosine=float(unmatched),
                retrieval_top1=float(np.mean(np.argmax(sim,axis=1)==np.arange(len(y))))))
            if variant=='cma':plot_panels([features['smiles_pooled'],features['fusion_pooled']],y,
                ['Before CMA: SMILES pooled','After CMA: cross-view pooled'],out/f'figure1_CMA_seed{seed}.png')
            if variant in ('cma','full'):panels[variant]=features['fusion_pooled']
            write_json(out/f'{a["run_id"]}_provenance.json',dict(checkpoint=str(a['checkpoint'].relative_to(ROOT)),sha256=digest(a['checkpoint']),
                origin=a['origin'],no_training=True,selection_seed=20261015,n=len(idx),sample_selection_uses_rt=False,environment=rt.environment()))
            del model,features,collected
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if set(panels)=={'cma','full'}:plot_panels([panels['cma'],panels['full']],y,
            ['CMA without RSCL','CMA with RSCL'],out/f'figure2_RSCL_seed{seed}.png')
    write_csv(out/'geometry_metrics.csv',quantitative)
    write_json(out/'missing_models.json',missing)
    print('Representation analysis complete. Interpret quantitative metrics across seeds; plots alone do not establish causality.')
