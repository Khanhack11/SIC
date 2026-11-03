"""Export the actual immutable training labels and their numeric targets for review."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from backend.mobile_model import FIELD_CLASSES


def export(data, output):
    if output.exists():raise ValueError('Preserve existing review CSV; choose a new output')
    manifest=data/'records.json';content=manifest.read_bytes();sha=hashlib.sha256(content).hexdigest()
    if sha!=json.loads((data/'audit.json').read_text())['records_sha256']:raise ValueError('Manifest differs from audit')
    rows=json.loads(content)
    if any(row.get('label') not in FIELD_CLASSES for row in rows):raise ValueError('Manifest has an unknown target label')
    fields=['path','label','class_index','split','domain','source_dataset','original_label','sha256',
            'label_status','proposed_label','reviewer','review_notes']
    output.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter();preview={}
    with output.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            label=row['label']
            if label not in FIELD_CLASSES:raise ValueError('Unknown target label: '+label)
            item={key:row.get(key,'') for key in fields}
            item.update(class_index=FIELD_CLASSES.index(label),source_dataset=row.get('source_dataset',row.get('source_name','')),
                        original_label=row.get('original_label',row.get('source_label','')),
                        label_status=row.get('label_review','source label; not project-expert adjudicated'),
                        proposed_label='',reviewer='',review_notes='')
            writer.writerow(item);counts[(row['split'],label)]+=1;preview.setdefault(label,item)
    summary={'manifest':str(manifest),'manifest_sha256':sha,'rows':len(rows),
             'class_map':{str(i):name for i,name in enumerate(FIELD_CLASSES)},
             'counts':[{'split':split,'label':label,'count':n} for (split,label),n in sorted(counts.items())],
             'scope':'Read-only export. Proposed labels are not applied to the locked training manifest. Blank original_label means original text is absent from this record.'}
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with output.with_name(output.stem+'.preview.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(preview.values())
    print(json.dumps({'rows':len(rows),'labels':len(preview),'review_csv':str(output),'input_unchanged':hashlib.sha256(manifest.read_bytes()).hexdigest()==sha}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/field_mobile_v2'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.data,args.output)
