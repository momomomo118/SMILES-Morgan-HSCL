"""Download original SMRT checkpoint release assets and verify their SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--include-zero-weight', action='store_true')
    parser.add_argument('--member', action='append', help='Download only a named member; may be repeated.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads((repo / 'configs/scheme_a/model_artifacts.json').read_text(encoding='utf-8'))
    if args.member and set(args.member) - {item['member'] for item in inventory['members']}:
        parser.error('Unknown member name.')
    for item in inventory['members']:
        if args.member and item['member'] not in args.member:
            continue
        if not item['active'] and not args.include_zero_weight:
            continue
        target = args.model_root / item['path']
        if target.exists():
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if target.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise SystemExit('Existing file differs; refusing overwrite: ' + str(target))
            print('Verified existing:', item['member'])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.download')
        digest = hashlib.sha256()
        size = 0
        # Exclusive creation prevents overwriting either a checkpoint or a prior partial download.
        with partial.open('xb') as output:
            request = urllib.request.Request(item['public_download_url'], headers={'User-Agent': 'MCRL-artifact-downloader'})
            with urllib.request.urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise SystemExit('Downloaded file failed integrity check: ' + str(partial))
        # Path.rename on Windows refuses to replace an existing target.
        partial.rename(target)
        print('Downloaded and verified:', item['member'])


if __name__ == '__main__':
    main()
