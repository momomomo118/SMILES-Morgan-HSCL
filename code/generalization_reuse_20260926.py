"""Retrospective architecture-matched scaffold comparison; zero neural training."""
from generalization_scaffold_20260926 import *

DEST=ROOT/'outputs/generalization_20260926/scaffold_reuse'

def load(partition,split):
    path=OLD/f'data/scaffold_{split}.csv' if partition=='scaffold' else ROOT/f'data/unified/{split}.csv'
    d=pd.read_csv(path);values=[]
    paths=[OLD/'ccrl_resume_epoch11'/('prediction_val.csv' if split=='val' else 'prediction.csv')] if partition=='scaffold' else [ROOT/f'outputs/paper_ablation_20260825/runs/E_residual_supcr_seed2026/predictions_{split}.csv']
    paths += [OLD/f'gnn/seed{s}/predictions_{split}_gnn.csv' if partition=='scaffold' else ROOT/f'checkpoints/gnn/pure_gin_seed{s}/predictions_{split}_gnn.csv' for s in GIN_WEIGHTS]
    for p in paths:
        ref=pd.read_csv(p).set_index('sample_id')
        assert ref.index.is_unique and set(ref.index)==set(d.sample_id)
        ref=ref.loc[d.sample_id]
        assert ref.smiles.tolist()==d.smiles.tolist() and np.allclose(ref.true_rt,d.rt,atol=.001)
        values.append(ref.pred_rt.to_numpy())
    mat=np.stack(values,axis=1);assert np.isfinite(mat).all()
    return d,mat,[path]+paths

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    random_c=torch.load(ROOT/'checkpoints/paper_ablation_20260825/E_residual_supcr_seed2026/best.pt',map_location='cpu',weights_only=False)
    scaffold_c=torch.load(ROOT/'checkpoints/paper_ablation_20260825/scaffold_generalization/ccrl_resume_epoch11/best_cross_attention_residual.pt',map_location='cpu',weights_only=False)
    fields=['seed','target_transform','epochs','early_stopping_patience','batch_size','eval_batch_size','lr','encoder_lr','weight_decay','dropout','grad_clip','max_length','smiles_layers','morgan_hidden_dim','morgan_layers','morgan_heads','fusion_dim','fusion_heads','task_loss','huber_weight','huber_beta','temperature','similarity_weight','omega_align','omega_fusion','smiles_augment_prob','train_smiles_augmentation']
    config_comparison={k:dict(random=random_c['args'][k],scaffold=scaffold_c['args'][k]) for k in fields}
    assert all(v['random']==v['scaffold'] for v in config_comparison.values())
    protocol=dict(status='retrospective_existing_results',new_neural_training_epochs=0,
        architecture='one CCI-based RT predictor (seed 2026) + PureGIN seeds 2026/3407/777',
        full_top6_final_model=False,shared_recorded_hyperparameters=config_comparison,
        selection='nonnegative MAE linear program with sum(weights)=1 on each validation split, frozen before test evaluation',
        differences=['Random training scheduler patience=2; scaffold scheduler patience=3.',
         'Scaffold run resumed from its own epoch-11 best checkpoint with optimizer reset; training was interrupted after resumed epoch 24. Not a fully converged equal-budget comparison.',
         'Historical feature-generation environments are not fully documented. Stored predictions are reused unchanged.',
         'No conclusion of causal CCI efficacy or strictly isolated scaffold-split effect is supported by this retrospective comparison.'],
        scaffold_random_checkpoint_transfer=False,checkpoint_best_epochs=dict(random=random_c['epoch'],scaffold_resume=scaffold_c['epoch']),
        inputs={})
    # Document tensor shape equivalence, allowing wrapper prefix on RegressionHead.
    def shapes(c):return {k.replace('regression_head.net.','regression_head.'):list(v.shape) for k,v in c['model_state_dict'].items()}
    assert shapes(random_c)==shapes(scaffold_c)
    protocol['model_tensor_shapes_match']=True
    del random_c,scaffold_c
    for part in ['random','scaffold']:
        for split in ['val','test']:
            _,_,paths=load(part,split)
            for path in paths:protocol['inputs'][str(path.relative_to(ROOT))]=digest(path)
    write_json(DEST/'protocol.json',protocol)
    names=['CCI_predictor_seed2026']+[f'GIN{s}' for s in GIN_WEIGHTS]
    rows=[];weights={};cis=[];rng=np.random.default_rng(20260926)
    for part in ['random','scaffold']:
        val,pv,_=load(part,'val');w=simplex_mae(val.rt.to_numpy(),pv)
        weights[part]=dict(zip(names,w.tolist()));write_json(DEST/'validation_weights.json',weights)
        for split in ['val','test']:
            d,p,_=load(part,split);d['pred_single_CCI_configuration']=p@w
            for j,name in enumerate(names):d['pred_'+name]=p[:,j]
            d.to_csv(DEST/f'{part}_{split}_predictions.csv',index=False,encoding='utf-8-sig')
            for name in ['single_CCI_configuration']+names:
                rows.append(dict(partition=part,split=split,model=name,**metrics(d.rt,d['pred_'+name])))
            if split=='test':
                scaffold=d.smiles.map(lambda s:Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)),isomericSmiles=False))
                for name in ['single_CCI_configuration']+names:
                    g=pd.DataFrame(dict(scaffold=scaffold,err=abs(d.rt-d['pred_'+name]))).groupby('scaffold').err.agg(['sum','count'])
                    err=g['sum'].to_numpy();count=g['count'].to_numpy();boots=[]
                    for _ in range(2000):
                        idx=rng.integers(0,len(g),len(g));boots.append(err[idx].sum()/count[idx].sum())
                    cis.append(dict(partition=part,model=name,n_scaffolds=len(g),
                        mae_ci_low_s=float(np.quantile(boots,.025)),mae_ci_high_s=float(np.quantile(boots,.975)),method='scaffold cluster bootstrap; fixed trained models'))
    result=pd.DataFrame(rows);result.to_csv(DEST/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cis).to_csv(DEST/'scaffold_cluster_bootstrap.csv',index=False,encoding='utf-8-sig')
    print(result[result.split=='test'].to_string(index=False),flush=True)
    write_json(DEST/'completion.json',dict(status='complete',new_training_epochs=0,original_predictions_modified=False,fully_controlled_comparison=False))
    # Explicitly supersede earlier preparation; there is no active training queue.
    write_json(OUT/'queue_status.json',dict(status='not_started',reason='User requested reuse and minimum training; retrospective comparison used instead',new_training_epochs=0))

if __name__=='__main__':main()
