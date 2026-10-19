"""Explicit commands for an offline, isolated RMCL supplement package."""
import argparse
from common import ROOT, SEEDS, PREFIX


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='Show remaining jobs; no ML imports or training')
    v=sub.add_parser('verify',help='Verify manifest hashes; no training')
    v.add_argument('--weights',action='store_true',help='Also require and hash historical checkpoint pack')
    pre=sub.add_parser('preflight',help='Read-only dependency/data/GPU checks; no model training')
    pre.add_argument('--device',default='cuda')
    tr=sub.add_parser('train',help='Explicitly start training')
    tr.add_argument('--plan',choices=['missing','all'],default='missing')
    tr.add_argument('--variants',nargs='+',choices=list(PREFIX))
    tr.add_argument('--seeds',nargs='+',type=int,choices=SEEDS)
    tr.add_argument('--device',default='cuda')
    tr.add_argument('--epochs',type=int,help='Diagnostic override; such runs are marked non-final')
    an=sub.add_parser('analyze',help='Recompute metrics from saved predictions; no neural training')
    an.add_argument('--output',default='results/analysis')
    re=sub.add_parser('representations',help='Explicit checkpoint inference and representation analysis; no training')
    re.add_argument('--device',default='cuda')
    re.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    re.add_argument('--samples',type=int,default=1500)
    re.add_argument('--batch-size',type=int,default=16)
    re.add_argument('--output',default='results/representations')
    a=p.parse_args()
    if a.command=='list':
        from train_jobs import jobs
        a.plan='missing';a.variants=None;a.seeds=None
        print('\n'.join(f'{v} seed={s}' for v,s in jobs(a)))
    elif a.command=='verify':
        from verify_package import verify
        verify(a.weights)
    elif a.command=='preflight':
        from verify_package import preflight
        preflight(a.device)
    elif a.command=='train':
        from train_jobs import main as train
        train(a)
    elif a.command=='analyze':
        from analyze import main as analyze
        analyze(a.output)
    else:
        from representations import main as representations
        representations(a)


if __name__=='__main__':main()
