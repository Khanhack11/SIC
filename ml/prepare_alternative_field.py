"""Add publisher-labelled field sources to a new manifest, preserving parent holdouts."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from ml.analyze_recommended_datasets import inspect_image, record
from ml.download_recommended_datasets import checksum, save_json, transfer
from ml.near_duplicate_audit import HashIndex
from ml.runtime import check_pause

CCMT_SEPTORIA_FOLDER = '558d9ee5-c2f2-4dd1-9240-12d45a0532d7'
CCMT_RAW_CLASSES = {
    CCMT_SEPTORIA_FOLDER: ('septoria leaf spot', 'Tomato_Septoria_leaf_spot'),
    '7b44bb20-452c-468d-9caf-9149ea7b21e7': ('fall armyworm', 'Unsupported'),
}
BARI_CLASSES = {'Healthy': 'Potato_healthy', 'Fungal Late Blight': 'Potato_Late_blight',
                'Bacterial Soft Rot': 'Unsupported', 'Viral Leaf Roll': 'Unsupported',
                'Viral PVX': 'Unsupported', 'Viral PVY': 'Unsupported'}


def select_ccmt(metadata):
    if metadata['id'] != 'bwh3zbpkpv' or metadata['version'] != 1:
        raise ValueError('Expected CCMT publisher version 1')
    raw_groups = defaultdict(set)
    for row in metadata['files']:
        if '_train_' not in row['filename'].lower() and '_valid_' not in row['filename'].lower():
            raw_groups[row['content_details']['sha256_hash']].add(row.get('folder_id'))
    candidates = [r for r in metadata['files'] if r.get('folder_id') in CCMT_RAW_CLASSES]
    if any(Path(r['filename']).name != r['filename'] for r in candidates):
        raise ValueError('Unsafe publisher image filename')
    files = [r for r in candidates
             if len(raw_groups[r['content_details']['sha256_hash']]) == 1
             and r['filename'].lower().startswith(CCMT_RAW_CLASSES[r['folder_id']][0])]
    if not files:
        raise ValueError('Unexpected raw CCMT inventory')
    return sorted(files, key=lambda r: r['id'])


def bari_target(member):
    parts = Path(member).parts
    if len(parts) != 3 or parts[0] != 'Potato Leaf Disease Dataset':
        return None
    if parts[1] not in BARI_CLASSES or not parts[2].lower().startswith('orig_') or Path(member).suffix.lower() != '.jpg':
        return None
    return BARI_CLASSES[parts[1]]


def acquire(hub, ccmt_metadata, bari_metadata):
    ccmt = json.loads(ccmt_metadata.read_text(encoding='utf-8'))
    files = select_ccmt(ccmt)
    folder = hub/'ccmt_ghana_v1'
    save_json(folder/'source_files.json', files)
    selected_ids = {r['id'] for r in files}
    save_json(folder/'excluded_label_conflicts.json', [
        {'id': r['id'], 'filename': r['filename'], 'sha256': r['content_details']['sha256_hash'],
         'reason': 'filename_folder_label_mismatch' if not r['filename'].lower().startswith(CCMT_RAW_CLASSES[r['folder_id']][0])
                   else 'cross_folder_sha_conflict'}
        for r in ccmt['files'] if r.get('folder_id') in CCMT_RAW_CLASSES and r['id'] not in selected_ids])
    save_json(folder/'source.json', {'homepage': 'https://data.mendeley.com/datasets/bwh3zbpkpv/1',
        'doi': '10.17632/bwh3zbpkpv.1', 'license': 'CC-BY-4.0',
        'subset': 'Raw tomato Septoria and maize fall armyworm (outside SIC classes)',
        'publisher_folder_classes': CCMT_RAW_CLASSES, 'files': len(files),
        'publisher_metadata_sha256': checksum(ccmt_metadata),
        'review': 'Publisher reports expert validation; no project expert adjudication.'})
    def download(item):
        check_pause()
        path = folder/'original'/(item['id']+'_'+item['filename'])
        info = item['content_details']
        transfer(info['download_url'], path, info['size'], info['sha256_hash'])
        original, label = CCMT_RAW_CLASSES[item['folder_id']]
        return {'path': str(path.resolve()), 'label': label,
                'original_label': original, 'source_dataset': 'ccmt_ghana_v1',
                'publisher_sha256': info['sha256_hash'], 'publisher_file_id': item['id']}
    additions = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, row in enumerate(pool.map(download, files), 1):
            additions.append(row)
            if index % 200 == 0 or index == len(files):
                print('CCMT publisher hashes verified:', index, '/', len(files), flush=True)
    bari = json.loads(bari_metadata.read_text(encoding='utf-8'))
    if bari['id'] != 'd5b3fzpw3g' or bari['version'] != 1 or len(bari['files']) != 1:
        raise ValueError('Expected BARI publisher version 1 archive')
    folder = hub/'potato_bari_v1';info = bari['files'][0]['content_details']
    archive = folder/'original.zip'
    transfer(info['download_url'], archive, info['size'], info['sha256_hash'])
    excluded = 0
    with ZipFile(archive) as zipped:
        for member in zipped.namelist():
            label = bari_target(member)
            if label is None:
                excluded += 1;continue
            path = folder/'original'/Path(member).parts[1]/Path(member).name
            content = zipped.read(member);path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != content:
                raise ValueError('Existing extracted BARI image changed')
            if not path.exists():path.write_bytes(content)
            additions.append({'path': str(path.resolve()), 'label': label,
                'original_label': Path(member).parts[1], 'source_dataset': 'potato_bari_v1',
                'publisher_archive_sha256': info['sha256_hash'], 'archive_member': member})
    save_json(folder/'source.json', {'homepage': 'https://data.mendeley.com/datasets/d5b3fzpw3g/1',
        'doi': '10.17632/d5b3fzpw3g.1', 'license': 'CC-BY-4.0', 'archive_sha256': checksum(archive),
        'selected_orig_named_files': sum(r['source_dataset']=='potato_bari_v1' for r in additions),
        'excluded_augmented_or_other_entries': excluded,
        'review': 'Publisher source labels, orig_ filenames; not project expert adjudication.'})
    folder = hub/'corn_ecuador_healthy_v1'
    for item in json.loads((folder/'source_files.json').read_text(encoding='utf-8')):
        path = folder/'original/Healthy'/item['filename']
        expected = item['content_details']['sha256_hash']
        if checksum(path) != expected:raise ValueError('Ecuador publisher checksum mismatch')
        additions.append({'path': str(path.resolve()), 'label': 'Corn_healthy',
            'original_label': 'Healthy', 'source_dataset': 'corn_ecuador_healthy_v1',
            'publisher_sha256': expected})
    return additions


def exclusion_reason(row):
    if row['split'] != 'train':return None
    if row.get('source_dataset') == 'plantdoc_detection_crop':return 'crop_lineage_pending_review'
    name = ' '.join(row.get('original_label', '').lower().replace('_', ' ').split())
    if name in {'corn rust', 'potato phytophthora'}:return 'taxonomic_mapping_pending_review'
    if row['label'].endswith('_healthy') and name in {'corn', 'potato', 'tomato', 'corn leaf', 'potato leaf', 'tomato leaf'}:
        return 'generic_healthy_pending_review'
    return None


def prepare(hub, parent, output, additions):
    if output.exists():raise ValueError('Output exists; preserve immutable manifests')
    manifest = parent/'records.json';raw = manifest.read_bytes();digest = hashlib.sha256(raw).hexdigest()
    if json.loads((parent/'audit.json').read_text())['records_sha256'] != digest:
        raise ValueError('Parent manifest changed')
    old = json.loads(raw);excluded = [];rows = []
    for row in old:
        reason = exclusion_reason(row)
        if reason:excluded.append({'path': row['path'], 'reason': reason})
        else:rows.append(dict(row, audit_group=row['group']))
    cache = {}
    for path in sorted((hub/'analysis').glob('*_inventory.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            value = json.loads(line)
            if value.get('status') == 'valid':cache[value['sha256']] = value
    index = HashIndex();pixels = set();shas = set()
    for row in rows:
        check_pause();info = cache.get(row['sha256'])
        if info is None:
            path = Path(row['path']);info = inspect_image((record('parent',path,row['label']),path.read_bytes()))
        if info['sha256'] != row['sha256'] or info['status'] != 'valid':raise ValueError('Parent image changed/invalid')
        pixels.add(info['pixel_sha256']);shas.add(info['sha256'])
        index.add(int(info['phash'],16), int(info['dhash'],16))
    new_audit = []
    for number, row in enumerate(additions, 1):
        check_pause();path = Path(row['path']);info = inspect_image((record(row['source_dataset'],path,row['label']),path.read_bytes()))
        new_audit.append(info)
        if info['status'] != 'valid' or min(info['width'],info['height']) < 64:
            excluded.append({'path': str(path), 'reason': 'invalid_or_small'});continue
        ph,dh = int(info['phash'],16),int(info['dhash'],16)
        if info['sha256'] in shas or info['pixel_sha256'] in pixels or any((dh^other).bit_count()<=4 for _,other in index.neighbors(ph,4)):
            excluded.append({'path': str(path), 'reason': 'duplicate_or_near_match_parent_or_new'});continue
        group = 'alternative:'+info['pixel_sha256']
        rows.append(dict(row,sha256=info['sha256'],pixel=info['pixel_sha256'],group=group,audit_group=group,
                         domain='MultiSourceField',new_source=True,split='train',
                         label_review='publisher source label; no project expert adjudication'))
        shas.add(info['sha256']);pixels.add(info['pixel_sha256']);index.add(ph,dh)
        if number%200==0:print('Audited new source images:',number,'/',len(additions),flush=True)
    held = {(r['path'],r['sha256'],r['label'],r['split']) for r in old if r['split']!='train'}
    assert held == {(r['path'],r['sha256'],r['label'],r['split']) for r in rows if r['split']!='train'}
    held_sha = {r['sha256'] for r in rows if r['split']!='train'}
    held_groups = {r['audit_group'] for r in rows if r['split']!='train'}
    assert not any(r['sha256'] in held_sha or r['audit_group'] in held_groups for r in rows if r['split']=='train')
    if not any(r.get('new_source') for r in rows):raise ValueError('No new source images survived audit')
    output.mkdir(parents=True)
    save_json(output/'records.json',rows);save_json(output/'new_image_audit.json',new_audit)
    audit={'parent_records_path':str(manifest.resolve()),'parent_records_sha256':digest,
        'records_sha256':checksum(output/'records.json'),'old_holdouts_preserved':True,
        'preparation_script_sha256':checksum(Path(__file__)),
        'new_sources_train_only':True,'counts':dict(Counter(f"{r['domain']}/{r['split']}" for r in rows)),
        'new_sources':dict(Counter(r['source_dataset'] for r in rows if r.get('new_source'))),
        'excluded':excluded,'test_used_for_selection':False,
        'limitations':['New source capture-session IDs unavailable; all new sources train-only.',
          'Perceptual hash filtering is heuristic, not proof of physical-leaf independence.',
          'Published labels are not independently adjudicated by this project.',
          'Validation is historical and has been used by parent selection; no fresh external test claim.']}
    save_json(output/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='excluded'},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub',type=Path,default=Path('data/raw/dataset_hub_20261103'))
    parser.add_argument('--parent',type=Path,default=Path('data/processed/field_mobile_v2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ccmt-metadata',type=Path,required=True)
    parser.add_argument('--bari-metadata',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.hub,args.parent,args.output,acquire(args.hub,args.ccmt_metadata,args.bari_metadata))
