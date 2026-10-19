"""Training is only entered by the explicit run.py train subcommand."""
import contextlib
import gc
import json
import math
import os
import time
from dataclasses import asdict

from common import ROOT, SEEDS, run_id, read_json, write_json, digest, selected_assets


def jobs(args):
    variants = args.variants or ['base','cma','rscl','full']
    seeds = args.seeds or list(SEEDS)
    for variant in variants:
        for seed in seeds:
            asset = selected_assets(variant, seed)
            if args.plan == 'missing' and asset:
                continue
            yield variant, seed


def train_one(variant, seed, args):
    import torch
    import runtime as rt
    cfg = rt.config_for(variant, seed, args.device)
    if args.epochs is not None:
        if args.epochs < 1: raise ValueError('epochs must be positive')
        cfg.epochs = args.epochs
        cfg.run_id += f'_diagnostic{args.epochs}'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run preflight before training')
    root = ROOT/'results/runs'/cfg.run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'worker.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f'pid={os.getpid()}\n'.encode()); os.close(fd)
    try:
        conf = asdict(cfg)
        conf['protocol_version'] = 'rmcl_supplement_v1'
        conf['fusion_supcr_target'] = 'concat(smiles,morgan)' if variant == 'rscl' else 'CMA pooled'
        conf['epochs_override'] = args.epochs is not None
        conf['rdkit_augmentation_epoch_seeded'] = True
        conf['interpretation'] = 'CMA configuration including its regression input; not an isolated attention-capacity control'
        signature_data = {k:v for k,v in conf.items() if k not in ('device',)}
        signature_data['train_sha256'] = digest(cfg.train_data)
        signature_data['val_sha256'] = digest(cfg.val_data)
        signature_data['runtime'] = rt.environment()
        signature_data['code_sha256'] = {p: digest(ROOT/p) for p in ('runtime.py','train_jobs.py','scripts/paper_ablation_runner_20260825.py','code/src/utils/multimodal_dataset.py')}
        signature = __import__('hashlib').sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest()
        if (root/'complete.json').exists():
            if read_json(root/'complete.json')['signature'] != signature:
                raise RuntimeError('Completed run signature differs; use a separately extracted package for a new protocol')
            print(f'SKIP completed {cfg.run_id}', flush=True)
            return
        latest = rt.torch_load(root/'latest.pt') if (root/'latest.pt').exists() else None
        if latest and latest['signature'] != signature:
            raise RuntimeError('Resume signature differs; refusing to mix configurations')
        write_json(root/'config.json', conf)
        write_json(root/'environment.json', rt.environment())
        rt.legacy.set_seed(seed)
        transformer, (train_loader,val_loader,test_loader) = rt.resources(cfg)
        model = rt.build_model(cfg).to(cfg.device)
        optimizer = rt.legacy.build_optimizer(cfg, model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2)
        align = rt.legacy.CrossModalSupCRLoss(temperature=cfg.temperature,similarity_weight=cfg.similarity_weight,use_log=False)
        fusion = rt.legacy.FusionSupCRLoss(temperature=cfg.temperature,use_log=False)
        history=[]; best=math.inf; best_epoch=-1; stale=0; start=1; elapsed=0.
        if latest:
            model.load_state_dict(latest['model_state_dict'],strict=True)
            optimizer.load_state_dict(latest['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v) and k!='step': state[k]=v.to(cfg.device)
            scheduler.load_state_dict(latest['scheduler_state_dict'])
            history=latest['history']; best=latest['best_val'];best_epoch=latest['best_epoch'];stale=latest['stale']
            start=latest['epoch']+1;elapsed=latest['elapsed_seconds']
            rt.restore_rng(latest['rng'],train_loader.dataset)
            del latest
        began=time.monotonic()
        with (root/'training.log').open('a',encoding='utf-8') as log:
            for epoch in range(start,cfg.epochs+1):
                if stale >= cfg.early_stopping_patience: break
                # RDKit owns a separate enumeration RNG. Seed by epoch for restartability.
                from rdkit import rdBase
                rdBase.SeedRandomNumberGenerator(seed*1009+epoch)
                model.train(); loss_sum=0.; count=0
                for batch in train_loader:
                    batch={k:v.to(cfg.device) if torch.is_tensor(v) else v for k,v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    out=model(batch.get('input_ids'),batch.get('attention_mask'),batch.get('morgan_fp'))
                    loss,_=rt.legacy.compute_loss(out,batch['labels'],batch['labels_original'],cfg,align,fusion)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                    optimizer.step();loss_sum+=float(loss.detach());count+=1
                val=rt.legacy.evaluate(model,val_loader,cfg,transformer,'val')
                if not math.isfinite(val['mae']): raise RuntimeError('Nonfinite validation MAE')
                scheduler.step(val['mae'])
                if val['mae']<best:
                    best=val['mae'];best_epoch=epoch;stale=0
                    rt.save_checkpoint(root/'best.pt',dict(model_state_dict=model.state_dict(),epoch=epoch,args=asdict(cfg),signature=signature))
                else: stale+=1
                row=dict(epoch=epoch,train_loss=loss_sum/max(1,count),val_mae=val['mae'],best_val_mae=best,stale=stale)
                history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(cfg.run_id,json.dumps(row),flush=True)
                rt.save_checkpoint(root/'latest.pt',dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),epoch=epoch,history=history,best_val=best,best_epoch=best_epoch,
                    stale=stale,rng=rt.capture_rng(train_loader.dataset),elapsed_seconds=elapsed+time.monotonic()-began,signature=signature))
                write_json(root/'history.json',history)
        checkpoint=rt.torch_load(root/'best.pt')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        val=rt.legacy.evaluate(model,val_loader,cfg,transformer,'val',root/'predictions_val.csv')
        # Test is evaluated only after validation selection; never fed to scheduler or selection.
        test=rt.legacy.evaluate(model,test_loader,cfg,transformer,'test',root/'predictions_test.csv')
        write_json(root/'metrics.json',dict(run_id=cfg.run_id,variant=variant,seed=seed,best_epoch=best_epoch,val=val,test=test,
            elapsed_seconds=elapsed+time.monotonic()-began,signature=signature,config=conf))
        write_json(root/'complete.json',dict(status='complete',signature=signature,best_epoch=best_epoch,
            best_sha256=digest(root/'best.pt'),epochs_completed=len(history),test_used_for_selection=False))
        del model,optimizer,scheduler,checkpoint
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:
        lock.unlink(missing_ok=True)


def main(args):
    from verify_package import preflight
    preflight(args.device)
    queue=list(jobs(args))
    print('Queued jobs:',[run_id(v,s) for v,s in queue],flush=True)
    for variant,seed in queue:
        train_one(variant,seed,args)
