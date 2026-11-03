"""Compare models on the same fixed domain splits; CPU preprocessing follows each model's contract."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from backend.field_inference import FieldInferenceEngine
from backend.inference_engine import CropDiseaseInferenceEngine
from backend.mobile_model import FIELD_CLASSES
from backend.preprocessing import image_transform
from ml.train_field import FieldDataset, domain_metrics


def evaluate(args):
    if args.output.exists():raise ValueError('Preserve existing evidence; choose a new output')
    torch.set_num_threads(2)
    data_file=args.data/'records.json'
    if hashlib.sha256(data_file.read_bytes()).hexdigest()!=json.loads((args.data/'audit.json').read_text())['records_sha256']:
        raise ValueError('Split manifest changed')
    rows=[r for r in json.loads(data_file.read_text()) if r['split']==args.split]
    if args.baseline:
        engine=CropDiseaseInferenceEngine(device='cpu',runtime='onnx')
    else:engine=FieldInferenceEngine(args.weights)
    if not engine.weights_loaded:raise RuntimeError(engine.load_error)
    dataset=FieldDataset(rows,jpeg_draft=not args.baseline)
    if args.baseline:dataset.transform=image_transform()
    logits=[];targets=[];started=time.perf_counter()
    with torch.inference_mode():
        for index,(images,labels) in enumerate(DataLoader(dataset,batch_size=16)):
            values=engine.logits(images)
            if args.baseline:values=torch.cat([values,torch.full((len(values),1),-1e9)],1)
            logits.append(values);targets.append(labels)
            if index%40==0:print('Evaluation batch',index+1,flush=True)
    logits=torch.cat(logits);targets=torch.cat(targets)
    probabilities=(logits/engine.temperature).softmax(1);scores,predicted=probabilities.max(1)
    domains=domain_metrics(logits,targets,rows,engine.temperature)
    for domain,metrics in domains.items():
        mask=torch.tensor([r['domain']==domain for r in rows])
        diseased=torch.tensor([r['label']!='Unsupported' and not r['label'].endswith('_healthy') for r in rows])&mask
        healthy_pred=torch.tensor([FIELD_CLASSES[i].endswith('_healthy') for i in predicted.tolist()])
        accepted=(scores>=.75)&(predicted!=FIELD_CLASSES.index('Unsupported'))
        metrics['score_gate_coverage']=float(accepted[mask].float().mean())
        metrics['score_gate_accuracy']=float((predicted[accepted&mask]==targets[accepted&mask]).float().mean()) if (accepted&mask).any() else None
        metrics['known_diseased_count']=int(diseased.sum())
        metrics['known_diseased_high_confidence_healthy_count']=int((diseased&healthy_pred&accepted).sum())
        metrics['known_diseased_high_confidence_healthy_rate']=float((diseased&healthy_pred&accepted).sum()/diseased.sum()) if diseased.any() else None
        # Baseline has no Unsupported logit; its padded-logit NLL is not a meaningful OOD calibration measurement.
        if args.baseline:metrics['calibration']['nll']=None
    report={'split':args.split,'model':'baseline_v2_onnx' if args.baseline else 'candidate_v3_eager',
            'domains':domains,'temperature':engine.temperature,'runtime':engine.runtime,
            'records_sha256':hashlib.sha256(data_file.read_bytes()).hexdigest(),
            'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'seconds':time.perf_counter()-started,'count':len(rows),
            'score_gate_definition':'max probability >=0.75 and predicted != Unsupported, before field-healthy review; not final API acceptance',
            'limitations':'Proxy/internet labels; no local expert adjudication. Present-class metrics exclude absent ground-truth classes and retain confusion into all 12 labels.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v['known_macro_f1_present'] for k,v in domains.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/field_mobile_v1'))
    parser.add_argument('--weights',type=Path,default=Path('models/mobilenetv3_field_20261103.pt'))
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.baseline:args.weights=Path('models/mobilenetv2_crop_disease.pt')
    evaluate(args)
