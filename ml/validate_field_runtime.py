"""Compare V3 eager/ONNX on fixed validation images and benchmark identical image pipelines."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.field_inference import FieldInferenceEngine
from backend.mobile_model import FIELD_CLASSES
from ml.runtime import check_pause
from ml.train_field import FieldDataset, domain_metrics


def compare(args):
    if args.output.exists(): raise ValueError('Preserve prior evidence; choose a new output')
    torch.set_num_threads(2)
    manifest=args.data/'records.json'
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest!=json.loads((args.data/'audit.json').read_text())['records_sha256']:raise ValueError('Manifest changed')
    rows=[r for r in json.loads(manifest.read_text()) if r['split']=='validation']
    fixtures=sorted(Path('sample_images/demo').glob('*.jpg'))
    raw=[p.read_bytes() for p in fixtures]
    engines={};cold={};logits={};targets=None
    for name,path in [('eager',None),('onnx',args.onnx)]:
        started=time.perf_counter();engine=FieldInferenceEngine(args.weights,onnx_path=path)
        if not engine.weights_loaded:raise ValueError(engine.load_error)
        cold[name]={'load_ms':(time.perf_counter()-started)*1000,
                    'first_predict_ms':engine.predict(raw[0])['latency_ms']}
        engines[name]=engine
    values={k:[] for k in engines};labels=[]
    with torch.inference_mode():
        for index,(images,target) in enumerate(DataLoader(FieldDataset(rows),batch_size=16)):
            check_pause();labels.append(target)
            for name,engine in engines.items(): values[name].append(engine.logits(images))
            if index%40==0:print('Validation parity batch',index+1,flush=True)
    targets=torch.cat(labels);logits={k:torch.cat(v) for k,v in values.items()}
    scores={k:(v/engines[k].temperature).softmax(1) for k,v in logits.items()}
    labels={k:v.argmax(1) for k,v in scores.items()}
    gates={k:(v.max(1).values>=.75)&(labels[k]!=FIELD_CLASSES.index('Unsupported')) for k,v in scores.items()}
    disagreements=int((labels['eager']!=labels['onnx']).sum())
    gate_disagreements=int((gates['eager']!=gates['onnx']).sum())
    difference=float((scores['eager']-scores['onnx']).abs().max())
    tolerance=.0001
    metrics={k:domain_metrics(v,targets,rows,engines[k].temperature) for k,v in logits.items()}
    timings={}
    for name,engine in engines.items():
        for i in range(8):engine.predict(raw[i%len(raw)])
        samples=[engine.predict(raw[i%len(raw)])['latency_ms'] for i in range(40)]
        timings[name]={'requests':len(samples),'mean_ms':float(np.mean(samples)),
                       'p50_ms':float(np.percentile(samples,50)),'p95_ms':float(np.percentile(samples,95)),
                       'max_ms':max(samples)}
    quality=disagreements==0 and gate_disagreements==0 and difference<=tolerance
    report={'validation_images':len(rows),'records_sha256':digest,
            'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
            'test_used':False,'label_disagreements':disagreements,'score_gate_disagreements':gate_disagreements,
            'max_probability_difference':difference,'probability_tolerance':tolerance,
            'max_logit_difference':float((logits['eager']-logits['onnx']).abs().max()),
            'quality_parity_passed':quality,'domains':metrics,'fresh_engine_in_shared_process':cold,
            'warm_cpu_timings':timings,'warmup_per_runtime':8,'threads_per_runtime':2,
            'benchmark_images':{p.name:hashlib.sha256(b).hexdigest() for p,b in zip(fixtures,raw)},
            'timing_scope':'Decode + shared letterbox/normalization + model + softmax; 4 demo JPEGs, 40 sequential calls. No upload validation, HTTP, advisory, mobile network or concurrency. First call is per engine in one shared process, not an isolated cold-system benchmark.',
            'runtime_eligible':quality and timings['onnx']['p95_ms']<=timings['eager']['p95_ms'],
            'deployment_note':'Runtime eligibility does not change the failed field-model release gate or make a disease prediction valid.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:report[k] for k in ['quality_parity_passed','runtime_eligible','label_disagreements','max_probability_difference']},flush=True)
    print(timings,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/field_mobile_v2'))
    parser.add_argument('--weights',type=Path,default=Path('models/candidate/mobilenetv3_field.pt'))
    parser.add_argument('--onnx',type=Path,default=Path('models/candidate/mobilenetv3_field.onnx'))
    parser.add_argument('--output',type=Path,default=Path('docs/final_release/villite_adaptation/runtime_validation.json'))
    compare(parser.parse_args())
