"""Check MCRL model files by recorded hashes without importing torch or fitting models."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-root',type=Path,default=REPO)
    ap.add_argument('--json-output',type=Path)
    args=ap.parse_args()
    items=json.loads((REPO/'configs/scheme_a/model_artifacts.json').read_text(encoding='utf-8'))['members']
    pre=json.loads((REPO/'configs/scheme_a/pretrained_model.json').read_text(encoding='utf-8'))
    expected=[{'path':m['path'],'sha256':m['sha256'],'required':m['active']} for m in items]
    expected.extend({'path':'pretrained/ChemBERTa-zinc-base-v1/'+m['path'],'sha256':m['local_sha256'],'required':True} for m in pre['required_files'])
    results=[]
    for item in expected:
        path=args.model_root/item['path']
        exists=path.is_file()
        actual=sha(path) if exists else None
        results.append({'path':item['path'],'required':item['required'],'exists':exists,'matches_sha256':actual==item['sha256'],'expected_sha256':item['sha256'],'actual_sha256':actual})
    passed=all(r['matches_sha256'] for r in results if r['required'])
    result={'status':'PASS' if passed else 'MISSING_OR_MISMATCHED_ASSETS','model_root':str(args.model_root),'neural_load_or_inference_tested':False,'checks':results}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Model artifact hashes:',result['status'],';',sum(r['matches_sha256'] for r in results),'/',len(results),'available and matched.')
    print('This verifies file identities, not dynamic checkpoint loading or inference.')
    raise SystemExit(0 if passed else 2)

if __name__=='__main__':
    main()
