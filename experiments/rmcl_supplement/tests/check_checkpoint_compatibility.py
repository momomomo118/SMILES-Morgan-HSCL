"""Load representative historical weights strictly; no inference or training."""
import gc
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, selected_assets, write_json
import runtime as rt


def main():
    rt.torch.set_num_threads(2)
    records=[]
    for variant in ['base','cma','full']:
        asset=selected_assets(variant,2026)
        cfg=rt.config_for(variant,2026,'cpu',asset['config'])
        model=rt.build_model(cfg)
        checkpoint=rt.torch_load(asset['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        records.append(dict(variant=variant,checkpoint=str(asset['checkpoint'].relative_to(ROOT)),
                            strict_state_load=True,epoch=checkpoint.get('epoch'),parameters=sum(p.numel() for p in model.parameters())))
        print(records[-1],flush=True)
        del checkpoint,model
        gc.collect()
    write_json(ROOT/'validation/checkpoint_compatibility.json',dict(status='PASS',records=records,
        inference_performed=False,training_performed=False,environment=rt.environment()))


if __name__=='__main__':main()
