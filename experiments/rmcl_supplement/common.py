"""Portable helpers; importing this module never starts training."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics

ROOT = Path(__file__).resolve().parent
SEEDS = (2026, 777, 3407)
PREFIX = {'base': 'C_concat_no_supcr', 'cma': 'D_residual_no_supcr',
          'rscl': 'F_concat_supcr', 'full': 'E_residual_supcr',
          'smiles': 'B_chemberta_bilstm'}


def run_id(variant, seed):
    return f'{PREFIX[variant]}_seed{seed}'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def metrics(y, p):
    if not y or len(y) != len(p) or not all(math.isfinite(x) for x in y + p):
        raise ValueError('Empty, inconsistent, or nonfinite predictions')
    e = [b-a for a, b in zip(y, p)]
    ae = [abs(v) for v in e]
    mean = statistics.mean(y)
    sst = sum((v-mean)**2 for v in y)
    return dict(n=len(y), MAE=statistics.mean(ae), RMSE=math.sqrt(statistics.mean(v*v for v in e)),
                R2=1-sum(v*v for v in e)/sst if sst else None,
                MedAE=statistics.median(ae), mean_signed_error=statistics.mean(e),
                within30_pct=100*sum(v <= 30 for v in ae)/len(ae),
                within60_pct=100*sum(v <= 60 for v in ae)/len(ae),
                within120_pct=100*sum(v <= 120 for v in ae)/len(ae))


def aligned_predictions(pred_path, truth_path, pred_col='pred_rt', tolerance=.001):
    truth = read_csv(truth_path)
    rows = read_csv(pred_path)
    by_id = {x['sample_id']: x for x in rows}
    if len(by_id) != len(rows) or len({x['sample_id'] for x in truth}) != len(truth):
        raise ValueError(f'Duplicate IDs: {pred_path}')
    if set(by_id) != {x['sample_id'] for x in truth}:
        raise ValueError(f'ID set mismatch: {pred_path}')
    y, p = [], []
    for x in truth:
        r = by_id[x['sample_id']]
        if r.get('smiles') != x['smiles'] or not math.isfinite(float(r['true_rt'])) or abs(float(r['true_rt'])-float(x['rt'])) > tolerance:
            raise ValueError(f'Molecule/label mismatch: {pred_path}, {x["sample_id"]}')
        y.append(float(x['rt']))
        p.append(float(r[pred_col]))
    return y, p


def selected_assets(variant, seed):
    """Prefer a completed new run; otherwise use explicitly indexed historical evidence."""
    rid = run_id(variant, seed)
    new = ROOT / 'results/runs' / rid
    if (new / 'complete.json').exists():
        return dict(run_id=rid, origin='new', directory=new, checkpoint=new/'best.pt',
                    config=new/'config.json', predictions=new/'predictions_test.csv')
    ref = ROOT / 'reference/ablation' / rid
    index = read_json(ROOT/'reference_index.json')
    if rid in index:
        return dict(run_id=rid, origin='historical', directory=ref,
                    checkpoint=ROOT/index[rid]['checkpoint'], config=ref/'config.json',
                    predictions=ref/'predictions_test.csv')
    return None
