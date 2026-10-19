"""Integrity checks use only the Python standard library."""
import math
from pathlib import Path
from common import ROOT, digest, read_json, read_csv


def verify(weights=False):
    manifest=read_json(ROOT/'manifest.json')
    checked=0
    for row in manifest['files']:
        if row['archive']=='checkpoints' and not weights: continue
        p=ROOT/row['path']
        if not p.is_file():raise FileNotFoundError(f'Missing {p}; unzip both packs into the same parent directory if needed')
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise ValueError(f'Hash/size mismatch: {p}')
        checked+=1
    print(f'Integrity PASS: {checked} files; weights_required={weights}',flush=True)
    return checked


def preflight(device='cuda'):
    verify(False)
    import runtime as rt
    env=rt.environment()
    print(env,flush=True)
    expected=read_json(ROOT/'protocol.json')['recommended_versions']
    mismatches={k:(env.get(k),v) for k,v in expected.items() if str(env.get(k))!=v}
    if mismatches:
        raise RuntimeError(f'Dependency versions differ: {mismatches}. Install requirements.txt before scientific runs.')
    if device.startswith('cuda') and not rt.torch.cuda.is_available():raise RuntimeError('No available CUDA device')
    for split,n in [('train',63163),('val',7019),('test',7798)]:
        rows=read_csv(ROOT/f'data/unified/{split}.csv')
        if len(rows)!=n or len({x['sample_id'] for x in rows})!=n:raise ValueError(f'Bad {split} count/IDs')
        if not all(math.isfinite(float(x['rt'])) for x in rows):raise ValueError('Nonfinite RT')
    print('Preflight PASS; no training or checkpoint inference was performed')


if __name__=='__main__':verify()
