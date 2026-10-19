"""Verify the distributed SI package without model or chemistry dependencies."""
from pathlib import Path
import csv, hashlib, json

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        p=ROOT/r['path'];b=p.read_bytes()
        assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
    catalog=json.loads((ROOT/'metadata/data_dictionary.json').read_text(encoding='utf-8'))['tables']
    for r in catalog:
        with (ROOT/r['path']).open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==r['columns'],r['path']
            assert sum(1 for _ in reader)==r['rows'],r['path']
    def load(name):
        with (ROOT/'data'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    assert len(load('S4_predret_fold_results.csv'))==100
    assert len(load('S4_predret_outer_test_predictions.csv'))==2197
    assert len(load('S4_predret_fold_weights.csv'))==800
    assert len(load('S1_smrt_random_assignments.csv'))==77980
    assert len(load('S1_smrt_scaffold_assignments.csv'))==77980
    try:
        from pypdf import PdfReader
        assert len(PdfReader(ROOT/'MCRL_Supporting_Information.pdf').pages)==manifest['pdf_pages']
    except ImportError:pass
    print('SI package PASS:',len(manifest['files']),'files;',len(catalog),'CSV tables; hashes and counts verified.')

if __name__=='__main__':main()
