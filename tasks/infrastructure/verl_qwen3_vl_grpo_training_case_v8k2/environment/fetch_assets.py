"""Fetch pinned assets with streaming size/SHA checks, or copy matching caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

workspace=Path(os.environ.get('PTDBENCH_WORKSPACE','/workspace')).resolve()
here=Path(__file__).parent
resources=workspace/'resources';resources.mkdir(exist_ok=True)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
# Mirrored HF endpoint; huggingface.co itself is not on the cluster egress
# whitelist, but https_proxy is exported by the prepare shell and urllib
# honors it, so the mirror URL goes out through the whitelist proxy.
endpoint=os.environ.get('HF_ENDPOINT','https://hf-mirror.com').rstrip('/')
def fetch(url,path,bytes_):
    # Streams into <path>.partial and renames it into place only once the full
    # size is on disk, so callers never observe a partial file. Range-resumes
    # across retries; a short read (the mirror serves 200-status throttle
    # pages) is discarded rather than resumed, since its bytes are garbage.
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(8):
        done=partial.stat().st_size if partial.exists() else 0
        headers={'Range':'bytes=%d-'%done} if done else {}
        exc=None
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=600) as src,partial.open('ab' if done else 'wb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
        except (urllib.error.URLError,TimeoutError,OSError) as caught:
            exc=caught
        size=partial.stat().st_size if partial.exists() else 0
        if size>=bytes_:partial.replace(path);return
        if size!=done:partial.unlink()
        wait=min(10*2**attempt,120)
        print(f'[fetch] retry {attempt+1} after {exc or "short read"}; sleeping {wait}s',flush=True)
        time.sleep(wait)
    raise ValueError('download failed '+url)
for asset in json.loads((here/'assets.json').read_text()):
    destination=resources/('model' if asset['kind']=='model' else 'raw')
    root=Path(os.environ.get('PTDBENCH_MODELS' if asset['kind']=='model' else 'PTDBENCH_DATASETS','/ptdbench/models' if asset['kind']=='model' else '/ptdbench/datasets'))
    # Warm cache: ptdbench_fetch.py's flat owner__name layout (whole-repo
    # snapshot, unpinned revision) plus this task's revision-pinned dir. Both
    # are only trusted after the per-file size+SHA check below.
    warm=root/asset['repository'].replace('/','__')
    pinned=root/(asset['repository'].replace('/','__')+'__'+asset['revision'][:12])
    for row in asset['files']:
        if row['path']=='.gitattributes':continue
        path=destination/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']:continue
        if path.exists():raise ValueError('existing asset mismatch '+str(path))
        temp=path.with_suffix(path.suffix+'.partial')
        candidates=[cached for base in (warm,pinned) if (cached:=base/row['path']).is_file() and cached.stat().st_size==row['bytes'] and sha(cached)==row['sha256']]
        if candidates:
            with candidates[0].open('rb') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
        else:
            # Dataset repos live under /datasets/ on the mirror; model repos do not.
            prefix='datasets/' if asset['kind']=='dataset' else ''
            url=endpoint+'/'+prefix+asset['repository']+'/resolve/'+asset['revision']+'/'+urllib.parse.quote(row['path'],safe='/')
            pinned.mkdir(parents=True,exist_ok=True)
            staged=pinned/row['path']
            if not (staged.is_file() and staged.stat().st_size==row['bytes'] and sha(staged)==row['sha256']):
                fetch(url,staged,row['bytes'])
                if staged.stat().st_size!=row['bytes'] or sha(staged)!=row['sha256']:raise ValueError('download checksum '+row['path'])
            with staged.open('rb') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('download checksum '+row['path'])
        temp.replace(path)
subprocess.run([sys.executable,str(here/'prepare_data.py'),'--raw-root',str(resources/'raw'),'--output',str(resources)],check=True)
