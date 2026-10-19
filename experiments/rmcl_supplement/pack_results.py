"""Archive only newly generated results; never package or modify historical inputs."""
import argparse
import datetime
import zipfile
from common import ROOT, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-latest',action='store_true',help='Include optimizer checkpoints for moving an interrupted run')
    a=p.parse_args()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest=ROOT/f'RMCL_results_{stamp}.zip'
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for f in sorted((ROOT/'results').rglob('*')):
            if not f.is_file() or f.name.endswith(('.tmp','.lock')):continue
            if f.name=='latest.pt' and not a.include_latest:continue
            z.write(f,f.relative_to(ROOT).as_posix())
        for f in ['protocol.json','source_inventory.csv','reference_index.json','manifest.json']:
            z.write(ROOT/f,f)
    dest.with_suffix('.zip.sha256').write_text(f'{digest(dest)}  {dest.name}\n',encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
