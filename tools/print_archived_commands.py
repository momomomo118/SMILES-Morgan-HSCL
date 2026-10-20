"""Print recorded CCI training/evaluation commands without executing them."""
import argparse
import ast
import json
from pathlib import Path
import shlex

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['train','eval'],default='train')
    ap.add_argument('--member',default='all')
    ap.add_argument('--split',choices=['val','test'],default='test')
    ap.add_argument('--output-root',default='../reproduction_runs')
    ap.add_argument('--model-root',type=Path,default=Path('.'),help='Root containing pretrained/ and checkpoints/.')
    args=ap.parse_args()
    source=ROOT/'code/src/train_multimodal.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    supported=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            for item in node.args:
                if isinstance(item,ast.Constant) and isinstance(item.value,str) and item.value.startswith('--'):
                    supported.add(item.value)
    configs=json.loads((ROOT/'supporting_information/configs/cci_checkpoint_arguments.json').read_text(encoding='utf-8'))
    if args.member!='all' and args.member not in configs:
        raise ValueError('Unknown member: '+args.member)
    for name,obj in configs.items():
        if args.member not in ['all',name]:
            continue
        cfg=dict(obj['args'])
        folder=Path(args.output_root)/args.mode/name/(args.split if args.mode=='eval' else 'fit')
        cfg.update(train_data='data/unified/train.csv',val_data='data/unified/val.csv',test_data='data/unified/'+(args.split if args.mode=='eval' else 'test')+'.csv',smiles_backbone='custom',smiles_model='pretrained/ChemBERTa-zinc-base-v1',output_dir=folder.as_posix(),checkpoint_dir=(folder/'checkpoints').as_posix(),prediction_output=(folder/'predictions.csv').as_posix(),eval_only_checkpoint=obj['checkpoint'] if args.mode=='eval' else None,smoke_test=False)
        cfg['smiles_model']=(args.model_root/'pretrained/ChemBERTa-zinc-base-v1').as_posix()
        if args.mode=='eval':
            cfg['eval_only_checkpoint']=(args.model_root/obj['checkpoint']).as_posix()
        cmd=['python','-B','code/src/train_multimodal.py']
        for key,value in cfg.items():
            flag='--'+key.replace('_','-')
            if flag not in supported or value is None:
                continue
            if key=='supcr_use_log':
                cmd.append('--supcr-use-log' if value else '--no-supcr-use-log')
            elif isinstance(value,bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag,str(value)])
        print('# '+name+'; recorded parameters, redirected outputs; NOT EXECUTED')
        print(shlex.join(cmd))
    print('# These commands are instructions only. This helper never starts training or inference.')

if __name__=='__main__':
    main()
