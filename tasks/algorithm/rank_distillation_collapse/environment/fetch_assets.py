#!/usr/bin/env python3
"""Fetch immutable public assets with mandatory SHA-256 and size checks.

The cache is not a delivery path: each validated file is copied to workspace or
prepared. A corrupt cache or download is a hard failure, never a fallback.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


def verify(path, asset):
    if not path.is_file() or path.stat().st_size != asset['size']:
        raise ValueError(f"size mismatch: {asset['name']}")
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != asset['sha256']:
        raise ValueError(f"SHA-256 mismatch: {asset['name']}")


def deliver(asset, cache_dir, roots, offline=False):
    relative = Path(asset['dest'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('asset destination must be relative')
    destination = roots[asset['destination']] / relative
    if destination.exists():
        verify(destination, asset)
        return
    cached = cache_dir / asset['sha256']
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        verify(cached, asset)
    elif offline:
        raise FileNotFoundError(f"asset absent from offline cache: {asset['name']}")
    else:
        if not asset['url'].startswith('https://'):
            raise ValueError('downloads require HTTPS')
        if asset['revision'] in ('main', 'master', 'latest', '') or asset['revision'] not in asset['url']:
            raise ValueError('asset URL must contain its immutable revision or release identity')
        fd, temp = tempfile.mkstemp(prefix='.download-', dir=cache_dir)
        try:
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(asset['url'], timeout=120) as response:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            verify(Path(temp), asset)
            os.replace(temp, cached)
        finally:
            Path(temp).unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic delivery prevents an interrupted copy from becoming a valid-looking artifact.
    fd, temp = tempfile.mkstemp(prefix='.deliver-', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(cached, temp)
        verify(Path(temp), asset)
        os.replace(temp, destination)
    finally:
        Path(temp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', default=['model', 'dataset', 'build'])
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    environment = Path(os.environ['PTDBENCH_ENVIRONMENT'])
    roots = {'workspace': Path(os.environ['PTDBENCH_WORKSPACE']), 'prepared': Path(os.environ['PTDBENCH_PREPARED'])}
    manifest = json.loads((environment / 'assets.json').read_text())
    for asset in manifest['assets']:
        if asset['group'] not in args.groups:
            continue
        if asset['group'] == 'model':
            cache = Path(os.environ.get('PTDBENCH_MODELS', roots['prepared'] / 'download-cache/models'))
        else:
            cache = Path(os.environ.get('PTDBENCH_DATASETS', roots['prepared'] / 'download-cache/datasets'))
        print(f"Verify/fetch {asset['name']}", flush=True)
        deliver(asset, cache / 'autorlbench-sha256', roots, args.offline)


if __name__ == '__main__':
    main()
