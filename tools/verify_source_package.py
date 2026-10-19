"""Verify the source snapshot using only the Python standard library."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise SystemExit('Integrity failure: ' + item['path'])
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=item['path'])
    expected = {x['path'] for x in manifest['files']} | {'SOURCE_MANIFEST.json'}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and not any(x in {'.git', '__pycache__'} for x in p.relative_to(ROOT).parts)}
    unexpected = actual - expected
    missing = expected - actual
    if missing or unexpected:
        raise SystemExit('File set differs: missing=' + str(sorted(missing)) + ', extra=' + str(sorted(unexpected)))
    print('Source package PASS:', len(manifest['files']), 'files; hashes and Python syntax checked.')


if __name__ == '__main__':
    main()
