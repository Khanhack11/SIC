"""Combine audited source-label supplements; keep existing holdouts and a new source external."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from ml.analyze_recommended_datasets import inspect_image, record, write_csv
from ml.download_recommended_datasets import ROOT, save_json
from ml.near_duplicate_audit import HashIndex


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def assign_new_splits(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['label']].append(row)
    for label, items in sorted(grouped.items()):
        random.Random('expanded42:' + label).shuffle(items)
        # Capture-session/physical-leaf IDs are unavailable for Ecuador; keep this source train-only.
        train_only = all(row.get('source_name') == 'CornEcuadorHealthy' for row in items)
        count = max(1, round(.2 * len(items))) if len(items) >= 5 and not train_only else 0
        for index, row in enumerate(items):
            row['split'] = 'validation' if index < count else 'train'


def prepare(hub, parent, output, triage=None, inventory_path=None):
    if output.exists():
        raise ValueError('Preserve prior data manifests; choose a new output')
    parent_path = parent / 'records.json'
    content = parent_path.read_bytes()
    parent_digest = hashlib.sha256(content).hexdigest()
    if json.loads((parent/'audit.json').read_text())['records_sha256'] != parent_digest:
        raise ValueError('Parent audit mismatch')
    original = json.loads(content)
    inventory_path = inventory_path or hub/'analysis/image_inventory.csv'
    original_sources = {'plantvillage','plantdoc_classification','plantdoc_detection','plantwild_v2','plantseg'}
    inventory = [r for r in read_csv(inventory_path) if r['dataset'] in original_sources]
    if len(inventory) != 35947:
        raise ValueError('Expected complete original five-source inventory')
    by_sha = {r['sha256']: r for r in inventory if r['status'] == 'valid'}
    index = HashIndex()
    pixels = set()
    for row in inventory:
        if row['status'] != 'valid':
            continue
        pixels.add(row['pixel_sha256'])
        index.add(int(row['phash'],16), (int(row['dhash'],16), row['duplicate_group']))
    records, rejected = [], []
    for row in original:
        row = dict(row)
        row['audit_group'] = by_sha[row['sha256']]['duplicate_group']
        records.append(row)
    protected = {r['audit_group'] for r in records if r['split'] != 'train'}
    clean = []
    for row in records:
        if row['split'] == 'train' and row['audit_group'] in protected:
            rejected.append({'path':row['path'],'reason':'old training near-match with protected holdout'})
        else:
            clean.append(row)
    records = clean
    excluded = json.loads(triage.read_text()) if triage else {}
    supplements = []
    for row in read_csv(hub/'analysis/review/classification_639.csv'):
        if row['id'] in excluded:
            rejected.append({'path':row['path'],'reason':excluded[row['id']]})
            continue
        audited_group = by_sha[row['sha256']]['duplicate_group']
        if audited_group in protected:
            raise ValueError('Candidate crosses old holdout boundary')
        supplements.append({'path':row['path'],'sha256':row['sha256'],'label':row['target_label'],
            'group':audited_group,'audit_group':audited_group,'domain':'FieldSupplement',
            'source_name':'PlantWildV2','original_label':row['source_label'],
            'label_review':'published source label; automatic QC and layout triage; no project expert adjudication'})
    new_audit = []
    sources = [
        ('corn_ecuador_healthy_v1','CornEcuadorHealthy','Corn_healthy','supplement'),
        ('potato_holeta_v2','HoletaExternal',None,'external'),
    ]
    for folder, source, label, use in sources:
        paths = sorted((hub/folder/'original').rglob('*.jpg'))
        if not paths:
            raise ValueError('Downloaded source missing: '+folder)
        for number, path in enumerate(paths,1):
            if label:
                target = label
            elif path.parent.name.lower() == 'healthy':
                target = 'Potato_healthy'
            elif 'late' in path.parent.name.lower():
                target = 'Potato_Late_blight'
            else:
                raise ValueError('Unrecognized Holeta label: '+str(path))
            source_id = source+':'+path.name
            if source_id in excluded:
                rejected.append({'path':str(path),'reason':excluded[source_id]})
                continue
            item = inspect_image((record(folder,path,target),path.read_bytes()))
            new_audit.append(item)
            if item['status'] != 'valid' or min(item['width'],item['height']) < 64:
                rejected.append({'path':str(path),'reason':'image decoding or resolution'})
                continue
            ph,dh=int(item['phash'],16),int(item['dhash'],16)
            neighbors = [group for _,(old_dh,group) in index.neighbors(ph,3) if (dh^old_dh).bit_count()<=4]
            if item['pixel_sha256'] in pixels or neighbors:
                rejected.append({'path':str(path),'reason':'exact/perceptual overlap with existing source or prior new image','matched_groups':neighbors[:5]})
                continue
            group=source+':'+item['pixel_sha256'][:24]
            index.add(ph,(dh,group));pixels.add(item['pixel_sha256'])
            row={'path':str(path.resolve()),'sha256':item['sha256'],'label':target,'group':group,'audit_group':group,
                 'domain':'FieldSupplement' if use=='supplement' else source,'source_name':source,
                 'original_label':path.parent.name,'label_review':'published source folder label; no project expert adjudication'}
            if use=='supplement':
                supplements.append(row)
            else:
                row['split']='external_test';records.append(row)
            if number%100==0:
                print(source,'audited',number,'/',len(paths),flush=True)
    assign_new_splits(supplements)
    records.extend(supplements)
    held_groups={r['audit_group'] for r in records if r['split'] != 'train'}
    held_sha={r['sha256'] for r in records if r['split'] != 'train'}
    if any(r['audit_group'] in held_groups or r['sha256'] in held_sha for r in records if r['split']=='train'):
        raise ValueError('Training and holdout overlap')
    output.mkdir(parents=True)
    save_json(output/'records.json',records)
    save_json(output/'new_source_image_audit.json',new_audit)
    audit={'parent_records_path':str(parent_path.resolve()),'parent_records_sha256':parent_digest,
           'records_sha256':hashlib.sha256((output/'records.json').read_bytes()).hexdigest(),
           'seed':42,'new_source_validation_fraction':.2,'counts':dict(Counter(f"{r['domain']}/{r['split']}/{r['label']}" for r in records)),
           'inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
           'new_sources':dict(Counter(f"{r.get('source_name','previous')}/{r['split']}" for r in records)),
           'rejected':rejected,'old_holdouts_preserved':True,'test_used_for_selection':False,
           'limitations':['Public source labels and layout triage are not project expert validation.',
             'pHash cannot guarantee different physical leaves; capture-session IDs are unavailable.',
             'Old validation/test remain historical benchmarks; parent weights have already been selected on old validation.',
             'HoletaExternal is reserved in entirety for one external evaluation after configuration is fixed.']}
    save_json(output/'audit.json',audit)
    print(json.dumps({'sources':audit['new_sources'],'rejected':len(rejected),'records':len(records)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub',type=Path,default=ROOT)
    parser.add_argument('--parent',type=Path,default=Path('data/processed/field_mobile_v1'))
    parser.add_argument('--output',type=Path,default=Path('data/processed/field_expanded_v2_20261103'))
    parser.add_argument('--triage',type=Path)
    parser.add_argument('--inventory',type=Path)
    args=parser.parse_args()
    prepare(args.hub,args.parent,args.output,args.triage,args.inventory)
