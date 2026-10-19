"""Materialize the published scaffold and PredRet assignments without re-splitting."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / 'supporting_information'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def encode_csv(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def prepare(destination):
    subprocess.run([sys.executable, '-B', str(SI / 'verify_package.py')], check=True)
    exports = {}
    scaffold = read_csv(SI / 'data/S1_smrt_scaffold_assignments.csv')
    fields = ['sample_id', 'smiles', 'rt', 'split', 'canonical_smiles', 'murcko_scaffold', 'stored_scaffold_group']
    for split in ('train', 'val', 'test'):
        exports[Path('smrt_scaffold') / (split + '.csv')] = encode_csv(
            [row for row in scaffold if row['split'] == split], fields)

    cleaned = read_csv(SI / 'data/S1_predret_cleaned.csv')
    records = {(row['system'], row['sample_id']): row for row in cleaned}
    if len(records) != len(cleaned):
        raise ValueError('Duplicate PredRet system/sample identifiers')
    fields = ['sample_id', 'system', 'smiles', 'canonical_smiles', 'rt', 'split']
    exports[Path('predret/all_cleaned.csv')] = encode_csv(
        [dict(row, split='all') for row in cleaned], fields)
    groups = defaultdict(lambda: defaultdict(list))
    role_to_split = {'inner_train': 'train', 'inner_validation': 'val', 'outer_test': 'test'}
    for assignment in read_csv(SI / 'data/S1_predret_fold_assignments.csv'):
        system, sample = assignment['system'], assignment['sample_id']
        fold = int(assignment['outer_fold'])
        role = role_to_split[assignment['role']]
        groups[(system, fold)][role].append(dict(records[(system, sample)], split=role))
    for (system, fold), splits in sorted(groups.items()):
        if set(splits) != {'train', 'val', 'test'}:
            raise ValueError(f'Incomplete fold: {system}/{fold}')
        ids = {split: {row['sample_id'] for row in rows} for split, rows in splits.items()}
        canonical = {split: {row['canonical_smiles'] for row in rows} for split, rows in splits.items()}
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if ids[left] & ids[right] or canonical[left] & canonical[right]:
                raise ValueError(f'Overlapping partitions: {system}/{fold}')
        folder = Path('predret/by_system_10fold') / system / f'fold_{fold:02d}'
        for split, rows in splits.items():
            exports[folder / (split + '.csv')] = encode_csv(rows, fields)
        exports[folder / 'outer_train.csv'] = encode_csv(
            [dict(row, split='outer_train') for split in ('train', 'val') for row in splits[split]], fields)
        exports[folder / 'outer_test.csv'] = encode_csv(
            [dict(row, split='outer_test') for row in splits['test']], fields)
    # Check existing targets before writing any files. Never silently replace other data.
    for relative, data in exports.items():
        path = destination / relative
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f'Different file already exists; choose another --output-dir: {path}')
    for relative, data in exports.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    print(f'Prepared {len(exports)} files: 77,980 scaffold records and {len(cleaned):,} PredRet records across {len(groups)} folds.')
    print(destination.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/reproduction')
    prepare(parser.parse_args().output_dir)
