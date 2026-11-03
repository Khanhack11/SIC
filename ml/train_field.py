"""Train a new V3 field candidate, selecting only on separate domain validation losses."""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import time
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from backend.mobile_model import FIELD_CLASSES, FIELD_ARCHITECTURE, FIELD_PREPROCESSING, build_field_model, field_transform, decode_field_image
from backend.preprocessing import orient_rgb
from ml.evaluate import classification_metrics, calibration_metrics
from ml.train import calibrate
from ml.train_gpu import atomic_save, cpu_state, train_mode
from ml.runtime import check_pause


class FieldDataset(Dataset):
    def __init__(self,records,training=False,jpeg_draft=True):
        self.records=records;self.transform=field_transform(training);self.jpeg_draft=jpeg_draft
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records[index];raw=Path(row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']: raise ValueError('Source image changed: '+row['path'])
        if self.jpeg_draft:tensor=self.transform(decode_field_image(raw))
        else:
            with Image.open(io.BytesIO(raw)) as image:tensor=self.transform(orient_rgb(image))
        return tensor,FIELD_CLASSES.index(row['label'])


def domain_metrics(logits,targets,rows,temperature=1.):
    result={}
    for domain in sorted({row['domain'] for row in rows}):
        mask=torch.tensor([r['domain']==domain for r in rows])
        if not mask.any(): continue
        values=classification_metrics(targets[mask].numpy(),logits[mask].argmax(1).numpy(),FIELD_CLASSES)
        present=[v['f1'] for v in values['per_class'].values() if v['support']]
        known=[v['f1'] for k,v in values['per_class'].items() if v['support'] and k!='Unsupported']
        values['macro_f1_present']=sum(present)/len(present)
        values['known_macro_f1_present']=sum(known)/len(known)
        values['calibration']=calibration_metrics(logits[mask],targets[mask],temperature)
        result[domain]=values
    return result


def weights_for(rows):
    counts=Counter((r['domain'],r['label']) for r in rows)
    classes=Counter(domain for domain,_ in counts)
    return torch.tensor([1/(counts[r['domain'],r['label']]*classes[r['domain']]) for r in rows],dtype=torch.double)


def logits_for(model,rows,device,batch_size):
    model.eval();values=[];targets=[]
    with torch.inference_mode():
        for images,labels in DataLoader(FieldDataset(rows),batch_size=batch_size):
            check_pause();values.append(model(images.to(device)).cpu());targets.append(labels)
    return torch.cat(values),torch.cat(targets)


def validation_loss(logits,labels,rows):
    if not rows or len(rows) != len(labels):
        raise ValueError('Validation rows and targets must be nonempty and aligned')
    losses=nn.functional.cross_entropy(logits,labels,reduction='none')
    domains=sorted({row['domain'] for row in rows})
    return sum(float(losses[torch.tensor([r['domain']==domain for r in rows])].mean()) for domain in domains)/len(domains)


def train(args):
    if args.output.exists(): raise ValueError('Choose a new output; existing candidate preserved')
    if not torch.cuda.is_available(): raise RuntimeError('GPU environment with CUDA required')
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);torch.cuda.manual_seed_all(42)
    data_file=args.data/'records.json';records=json.loads(data_file.read_text())
    config={'records_sha256':hashlib.sha256(data_file.read_bytes()).hexdigest(),'seed':42,
            'head_epochs':args.head_epochs,'finetune_epochs':args.finetune_epochs,'batch_size':args.batch_size,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'transform_sha256':hashlib.sha256(Path('backend/mobile_model.py').read_bytes()).hexdigest()}
    audit=json.loads((args.data/'audit.json').read_text())
    if audit['records_sha256']!=config['records_sha256']: raise ValueError('Manifest changed after audit')
    train_rows=[r for r in records if r['split']=='train']
    val_rows=[r for r in records if r['split']=='validation']
    # Test paths never enter loaders until training and calibration have finished.
    device='cuda';model=build_field_model(pretrained=True).to(device)
    for p in model.features.parameters():p.requires_grad_(False)
    started=time.perf_counter();history=[]
    args.cache.mkdir(parents=True,exist_ok=True)
    def cache(rows,name,augment):
        path=args.cache/(name+'.pt')
        identity=hashlib.sha256(json.dumps({'rows':rows,'augment':augment,'config':config},sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved=torch.load(path,weights_only=True)
            if saved['identity']!=identity: raise ValueError('Feature cache differs; choose new cache')
            return saved['features'],saved['labels']
        features=[];labels=[];model.eval()
        with torch.inference_mode():
            for index,(images,targets) in enumerate(DataLoader(FieldDataset(rows,augment),batch_size=args.batch_size)):
                check_pause();values=model.avgpool(model.features(images.to(device))).flatten(1)
                features.append(values.cpu());labels.append(targets)
                if index%50==0:print(f'{name} feature batch {index+1}',flush=True)
        saved={'identity':identity,'features':torch.cat(features),'labels':torch.cat(labels)}
        atomic_save(saved,path);return saved['features'],saved['labels']
    a,ta=cache(train_rows,'train_base',False);b,tb=cache(train_rows,'train_aug',True)
    vf,vt=cache(val_rows,'validation',False)
    features=torch.cat([a,b]).to(device);targets=torch.cat([ta,tb]).to(device);vf=vf.to(device)
    weights=weights_for(train_rows*2)
    optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.head_epochs)
    best_loss=float('inf');best_state=None;best_stage='';stale=0
    for epoch in range(args.head_epochs):
        check_pause()
        model.classifier.train();indices=torch.multinomial(weights,len(weights),replacement=True)
        for batch in indices.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model.classifier(features[batch]),targets[batch],label_smoothing=.05)
            loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():logits=model.classifier(vf).cpu()
        value=validation_loss(logits,vt,val_rows)
        metrics=domain_metrics(logits,vt,val_rows)
        history.append({'stage':'head','epoch':epoch+1,'validation_loss':value,
                        'f1':{k:v['macro_f1_present'] for k,v in metrics.items()}})
        if value<best_loss-1e-5:
            best_loss=value;best_state=cpu_state(model);best_stage=f'head/{epoch+1}';stale=0
            atomic_save({'state_dict':best_state,'config':config,'history':history,'stage':best_stage},args.output.with_suffix('.resume.pt'))
        else:stale+=1
        scheduler.step()
        if epoch%5==0:print(history[-1],flush=True)
        if stale>=18:break
    model.load_state_dict(best_state)
    del features,targets,vf,a,b
    torch.cuda.empty_cache()
    for module in list(model.features.children())[-3:]:
        for p in module.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.features.parameters() if p.requires_grad],'lr':1e-5},
                                {'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda')
    for epoch in range(args.finetune_epochs):
        train_mode(model)
        sampler=WeightedRandomSampler(weights_for(train_rows),len(train_rows),replacement=True)
        for index,(images,labels) in enumerate(DataLoader(FieldDataset(train_rows,True),batch_size=args.batch_size,sampler=sampler)):
            check_pause();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=nn.functional.cross_entropy(model(images.to(device)),labels.to(device),label_smoothing=.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            scaler.step(optimizer);scaler.update()
            if index%100==0:print(f'Fine-tune {epoch+1}, batch {index+1}, loss {float(loss.detach()):.4f}',flush=True)
        logits,vt=logits_for(model,val_rows,device,args.batch_size)
        value=validation_loss(logits,vt,val_rows)
        history.append({'stage':'finetune','epoch':epoch+1,'validation_loss':value,
                        'f1':{k:v['macro_f1_present'] for k,v in domain_metrics(logits,vt,val_rows).items()}})
        if value<best_loss:
            best_loss=value;best_state=cpu_state(model);best_stage=f'finetune/{epoch+1}'
        atomic_save({'state_dict':best_state,'config':config,'history':history,'stage':best_stage},args.output.with_suffix('.resume.pt'))
        print(history[-1],flush=True)
    model.load_state_dict(best_state)
    logits,vt=logits_for(model,val_rows,device,args.batch_size)
    temperature,calibration=calibrate(logits,vt)
    report={'config':config,'architecture':FIELD_ARCHITECTURE,'selected_stage':best_stage,'history':history,
            'validation':domain_metrics(logits,vt,val_rows,temperature),'temperature':temperature,
            'training_seconds':time.perf_counter()-started,'test_used_for_selection':False,
            'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
            'pretraining':'Torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2',
            'head':'960 -> 256 Hardswish Dropout(0.2) -> 12',
            'checkpoint_recovery':'resume.pt preserves best model/history; optimizer and batch position are not resumable'}
    atomic_save({'architecture':FIELD_ARCHITECTURE,'class_names':FIELD_CLASSES,'state_dict':best_state,
                 'preprocessing':FIELD_PREPROCESSING,'temperature':temperature,'calibration':calibration,'training':report},args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Candidate saved:',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/field_mobile_v1'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/training_runs/field/model.pt'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/field_features_v2'))
    parser.add_argument('--head-epochs',type=int,default=100)
    parser.add_argument('--finetune-epochs',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=32)
    train(parser.parse_args())
