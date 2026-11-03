"""Audit and group PlantDoc before combining with the immutable PlantVillage split."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from PIL import Image
from backend.config import CLASS_NAMES
from backend.mobile_model import FIELD_CLASSES
from backend.preprocessing import orient_rgb
from ml.near_duplicate_audit import HashIndex, perceptual_hashes
from ml.download_plantdoc import local_path

MAPPING={
    'corn gray leaf spot':'Corn_Gray_leaf_spot', 'corn rust leaf':'Corn_Common_rust',
    'potato leaf':'Potato_healthy','potato leaf early blight':'Potato_Early_blight',
    'potato leaf late blight':'Potato_Late_blight','tomato leaf':'Tomato_healthy',
    'tomato early blight leaf':'Tomato_Early_blight','tomato leaf late blight':'Tomato_Late_blight',
    'tomato septoria leaf spot':'Tomato_Septoria_leaf_spot'}


def prepare(source, village, output):
    if output.exists(): raise ValueError('Choose a new output directory; splits are immutable')
    failures=json.loads((source/'failures.json').read_text())
    if failures: raise ValueError('Dataset download incomplete; resume first')
    source_meta=json.loads((source/'source.json').read_text())
    pvmeta=json.loads((village/'audit.json').read_text())
    records=[]; pv_pixels=set()
    for split in ('train','validation','test'):
        content=(village/f'{split}.csv').read_bytes()
        if hashlib.sha256(content).hexdigest()!=pvmeta['manifest_sha256'][split]: raise ValueError('PlantVillage manifest changed')
        for row in csv.DictReader(content.decode('utf-8').splitlines()):
            records.append({'path':str(Path(pvmeta['source_root'])/row['path']),'label':row['label'],
                'sha256':row['sha256'],'group':'pv:'+row['group'],'domain':'PlantVillage','split':split})
            pv_pixels.add(row['pixel_sha256'])
    village_index=HashIndex()
    for number,row in enumerate(records,1):
        with Image.open(row['path']) as original:
            ph,dh=perceptual_hashes(orient_rgb(original))
        village_index.add(ph,dh)
        if number%2000==0:print(f'Indexed {number} PlantVillage perceptual hashes',flush=True)
    field=[]; rejected=[]; index=HashIndex(); parent=[]
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    for entry in source_meta['files']:
        relative=Path(entry['path'])
        if len(relative.parts)!=3 or relative.parts[0] not in ('train','test'): continue
        path=source/local_path(entry)
        try:
            raw=path.read_bytes()
            with Image.open(path) as original:
                image=orient_rgb(original)
                if min(image.size)<32 or image.width*image.height>20_000_000: raise ValueError('Invalid dimensions')
                pixel=hashlib.sha256(image.tobytes()).hexdigest()
                if pixel in pv_pixels: raise ValueError('Exact pixels overlap PlantVillage')
                ph,dh=perceptual_hashes(image)
                if any((dh^other).bit_count()<=4 for _,other in village_index.neighbors(ph,4)):
                    raise ValueError('Conservative perceptual overlap with PlantVillage')
            label=MAPPING.get(relative.parts[1].lower(),'Unsupported')
            # Generic "Corn leaf blight" is not a verified Northern Leaf Blight label.
            if relative.parts[1].lower()=='corn leaf blight':
                raise ValueError('Ambiguous corn blight label excluded')
            item={'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'pixel':pixel,
                  'label':label,'original_label':relative.parts[1],'official_split':relative.parts[0],
                  'domain':'PlantDoc','dhash':dh}
            current=len(field);parent.append(current)
            for _,other in index.neighbors(ph,4):
                if (dh^field[other]['dhash']).bit_count()<=4: parent[find(current)]=find(other)
            field.append(item);index.add(ph,current)
        except Exception as error: rejected.append({'path':str(relative),'reason':str(error)})
    groups=defaultdict(list)
    for i,item in enumerate(field): groups[find(i)].append(item)
    candidates=defaultdict(list); removed=0
    for gid,items in sorted(groups.items()):
        if len({r['original_label'] for r in items})>1:
            rejected.extend({'path':r['path'],'reason':'Conflicting labels in perceptual group'} for r in items);continue
        # One representative per conservative duplicate group; official test always wins.
        item=min(items,key=lambda r:(r['official_split']!='test',r['path']))
        removed+=len(items)-1
        item={k:v for k,v in item.items() if k!='dhash'}
        item['group']=f'pd:{gid}'
        if any(r['official_split']=='test' for r in items):
            item['split']='test';records.append(item)
        else: candidates[item['label']].append(item)
    for label,items in sorted(candidates.items()):
        random.Random('42:'+label).shuffle(items)
        count=max(1,round(.2*len(items))) if len(items)>1 else 0
        for i,item in enumerate(items): item['split']='validation' if i<count else 'train';records.append(item)
    groups_by_split={split:{r['group'] for r in records if r['split']==split} for split in ('train','validation','test')}
    assert not any(groups_by_split[a]&groups_by_split[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    output.mkdir(parents=True)
    (output/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    report={'plantdoc_source':source_meta['repository'],'revision':source_meta['revision'],
        'labels':FIELD_CLASSES,'mapping':MAPPING,'seed':42,'records_sha256':hashlib.sha256((output/'records.json').read_bytes()).hexdigest(),
        'counts':dict(Counter(f"{r['domain']}/{r['split']}/{r['label']}" for r in records)),
        'duplicate_representatives_removed':removed,'rejected':rejected,
        'limitations':['Internet images and proxy labels, not locally expert-verified','Corn rust label mapped to project common-rust class, species ambiguity remains',
            'Generic corn leaf blight excluded; no PlantDoc corn-healthy class','Conservative perceptual grouping is not proof of physical-leaf disjointness',
            'Other PlantDoc labels mapped to Unsupported; does not cover all possible OOD inputs']}
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:sum(1 for r in records if r['domain']+'/'+r['split']==k) for k in ['PlantVillage/train','PlantDoc/train','PlantDoc/validation','PlantDoc/test']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('data/raw/plantdoc_snapshot'))
    parser.add_argument('--village',type=Path,default=Path('data/processed/plantvillage'))
    parser.add_argument('--output',type=Path,default=Path('data/processed/field_mobile_v1'))
    args=parser.parse_args();prepare(args.source,args.village,args.output)
