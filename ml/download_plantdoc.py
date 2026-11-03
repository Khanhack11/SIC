"""Download original PlantDoc files at a pinned revision, verifying Git blob hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import quote
from ml.download_dataset import fetch, blob_hash

REPO='pratikkayal/PlantDoc-Dataset'
REVISION='5467f6012d78d1c446145d5f582da6096f852ae8'


def local_path(entry):
    path=Path(entry['path'])
    if any(c in path.name for c in '<>:"|?*') or path.name.endswith((' ','.')):
        return path.with_name(entry['sha']+'.jpg')
    return path


def download(output):
    output.mkdir(parents=True,exist_ok=True)
    provenance=output/'source.json'
    if provenance.exists():
        source=json.loads(provenance.read_text())
    else:
        revision=REVISION
        tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1'))
        if tree.get('truncated'): raise ValueError('Incomplete repository tree')
        entries=[v for v in tree['tree'] if v['type']=='blob' and (v['path'].startswith(('train/','test/')) or v['path'] in ['README.md','LICENSE.txt'])]
        source={'repository':f'https://github.com/{REPO}','revision':revision,'files':entries}
        provenance.write_text(json.dumps(source,indent=2),encoding='utf-8')
    def transfer(entry):
        path=output/local_path(entry)
        if not path.resolve().is_relative_to(output.resolve()): raise ValueError('Unsafe source path')
        if path.exists() and blob_hash(path.read_bytes())==entry['sha']: return
        content=fetch(f"https://raw.githubusercontent.com/{REPO}/{source['revision']}/"+quote(entry['path'],safe='/'))
        if blob_hash(content)!=entry['sha']: raise ValueError(f'Hash mismatch: {entry["path"]}')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(content);temp.replace(path)
    failures=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending={pool.submit(transfer,entry):entry for entry in source['files']}
        for index,future in enumerate(as_completed(pending),1):
            try: future.result()
            except Exception as error: failures.append({'path':pending[future]['path'],'error':str(error)})
            if index%200==0: print(f'PlantDoc verified {index}/{len(pending)}, failures={len(failures)}',flush=True)
    (output/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise RuntimeError(f'{len(failures)} files failed; rerun to resume')
    print('PlantDoc complete:',source['revision'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/raw/plantdoc_snapshot'))
    download(parser.parse_args().output)
