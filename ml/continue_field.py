"""Warm-start V3 on train only, with exact batch-boundary recovery for this new run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.mobile_model import FIELD_ARCHITECTURE, FIELD_CLASSES, FIELD_PREPROCESSING, build_field_model
from ml.runtime import PAUSE_FILE, check_pause
from ml.train import calibrate
from ml.train_field import FieldDataset, domain_metrics, logits_for, validation_loss, weights_for
from ml.train_gpu import atomic_save, cpu_state, train_mode


def save_progress(path, model, optimizer, scaler, progress, config):
    atomic_save({'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'progress': progress, 'config': config,
                 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def restore_progress(path, model, optimizer, scaler, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise ValueError('Recovery configuration changed; preserve checkpoint and choose a new output')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['torch_rng'])
    random.setstate(saved['python_rng'])
    if saved['cuda_rng']:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA recovery requires the original GPU environment')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['progress']


def validate_parent_manifest(source, digest, audit, records, expanded=False):
    parent_digest = source['training']['config']['records_sha256']
    if parent_digest == digest:
        return
    if not expanded:
        raise ValueError('Parent used a different data manifest; use an audited expansion explicitly')
    if audit.get('parent_records_sha256') != parent_digest:
        raise ValueError('Expanded data does not identify the parent training manifest')
    parent_path = Path(audit['parent_records_path'])
    raw = parent_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent_digest:
        raise ValueError('Parent manifest changed')
    parent = json.loads(raw)
    original_holdout = {(r['path'], r['sha256'], r['label'], r['split']) for r in parent if r['split'] != 'train'}
    current = {(r['path'], r['sha256'], r['label'], r['split']) for r in records}
    if not original_holdout.issubset(current):
        raise ValueError('Existing holdout records were changed or removed')
    held_hashes = {r['sha256'] for r in records if r['split'] != 'train'}
    held_groups = {r['audit_group'] for r in records if r['split'] != 'train'}
    if any(r['sha256'] in held_hashes or r['audit_group'] in held_groups for r in records if r['split'] == 'train'):
        raise ValueError('Expanded training overlaps a protected holdout')


def finetuning_groups(model, blocks, backbone_lr, head_lr, top_lr=None):
    features = list(model.features.children())
    if not 1 <= blocks <= len(features):
        raise ValueError('Unfreeze blocks must fit the backbone')
    rates = [backbone_lr, head_lr] + ([] if top_lr is None else [top_lr])
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Learning rates must be finite and positive')
    for parameter in model.features.parameters(): parameter.requires_grad_(False)
    selected = features[-blocks:]
    for block in selected:
        for parameter in block.parameters(): parameter.requires_grad_(True)
    sections = [(selected, backbone_lr)] if top_lr is None else [(selected[:-4], backbone_lr), (selected[-4:], top_lr)]
    groups = []
    for section, rate in sections:
        parameters = [p for block in section for p in block.parameters() if p.requires_grad]
        if parameters: groups.append({'params': parameters, 'lr': rate})
    groups.append({'params': list(model.classifier.parameters()), 'lr': head_lr})
    return groups


def calibrate_balanced(logits, targets, rows):
    if len(rows) != len(targets) or not rows:
        raise ValueError('Calibration rows must align with targets')
    domains = sorted({r['domain'] for r in rows})
    masks = [torch.tensor([r['domain'] == domain for r in rows]) for domain in domains]
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50)
    def loss_at(temperature):
        losses = nn.functional.cross_entropy(logits / temperature, targets, reduction='none')
        return torch.stack([losses[mask].mean() for mask in masks]).mean()
    def closure():
        optimizer.zero_grad()
        loss = loss_at(log_temperature.exp().clamp(.05, 10)); loss.backward()
        return loss
    before = float(loss_at(1.).detach())
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(.05, 10))
    after = float(loss_at(temperature).detach())
    if not math.isfinite(after) or after > before: temperature, after = 1., before
    return temperature, {'method': 'equal_domain_temperature_nll_validation_only', 'domains': domains,
                         'temperature': temperature, 'validation_nll_before': before, 'validation_nll_after': after}


def train(args):
    if args.output.exists():
        raise ValueError('Existing final candidate preserved; choose a new output')
    if not torch.cuda.is_available():
        raise RuntimeError('Use the CUDA environment')
    torch.set_num_threads(2)
    torch.manual_seed(42); random.seed(42); torch.cuda.manual_seed_all(42)
    manifest = args.data / 'records.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit = json.loads((args.data / 'audit.json').read_text())
    if audit['records_sha256'] != digest:
        raise ValueError('Audited manifest changed')
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    for key, expected in {'architecture': FIELD_ARCHITECTURE, 'class_names': FIELD_CLASSES,
                          'preprocessing': FIELD_PREPROCESSING}.items():
        if source.get(key) != expected:
            raise ValueError('Parent checkpoint contract mismatch: ' + key)
    records = json.loads(manifest.read_text())
    expanded = getattr(args, 'expanded_data', False)
    validate_parent_manifest(source, digest, audit, records, expanded)
    train_rows = [r for r in records if r['split'] == 'train']
    val_rows = [r for r in records if r['split'] == 'validation']
    samples = args.samples_per_epoch if args.samples_per_epoch is not None else len(train_rows)
    if min(args.epochs, args.batch_size, samples) < 1 or not 0 <= args.min_village_f1 <= 1:
        raise ValueError('Positive training counts and a valid minimum village F1 are required')
    config = {'parent_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'records_sha256': digest, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'seed': 42, 'backbone_lr': args.backbone_lr, 'top_backbone_lr': args.top_backbone_lr,
              'head_lr': args.head_lr, 'expanded_data': expanded, 'unfreeze_blocks': args.unfreeze_blocks,
              'samples_per_epoch': samples, 'min_village_f1': args.min_village_f1,
              'balanced_calibration': args.balanced_calibration,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dataset_sampler_sha256': hashlib.sha256(Path('ml/train_field.py').read_bytes()).hexdigest(),
              'transform_sha256': hashlib.sha256(Path('backend/mobile_model.py').read_bytes()).hexdigest()}
    model = build_field_model().to('cuda')
    model.load_state_dict(source['state_dict'])
    optimizer = torch.optim.AdamW(finetuning_groups(model, args.unfreeze_blocks, args.backbone_lr,
                                 args.head_lr, args.top_backbone_lr), weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    recovery = args.output.with_suffix('.progress.pt')
    if recovery.exists():
        progress = restore_progress(recovery, model, optimizer, scaler, config)
        print('Recovered batch boundary:', progress['epoch'], progress['next_sample'], flush=True)
    else:
        logits, targets = logits_for(model, val_rows, 'cuda', args.batch_size)
        initial_metrics = domain_metrics(logits, targets, val_rows)
        village = initial_metrics.get('PlantVillage', {}).get('known_macro_f1_present', 0.)
        if village < args.min_village_f1:
            raise ValueError('Parent does not satisfy the requested village validation floor')
        print('Initial known F1:', {d:m['known_macro_f1_present'] for d,m in initial_metrics.items()}, flush=True)
        progress = {'epoch': 0, 'next_sample': 0, 'order': [], 'history': [],
                    'best_loss': validation_loss(logits, targets, val_rows),
                    'best_state': cpu_state(model), 'selected_stage': 'parent', 'active_seconds': 0.}
        save_progress(recovery, model, optimizer, scaler, progress, config)
    dataset = FieldDataset(train_rows, training=True)
    weights = weights_for(train_rows)
    started = time.perf_counter()
    while progress['epoch'] < args.epochs:
        check_pause()
        if not progress['order']:
            progress['order'] = torch.multinomial(weights, samples, replacement=True).tolist()
        # A private iterator seed prevents DataLoader startup from consuming augmentation RNG on resume.
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=progress['order'][progress['next_sample']:],
                            generator=torch.Generator().manual_seed(42 + progress['epoch']))
        train_mode(model)
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(images.to('cuda')), labels.to('cuda'), label_smoothing=.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            scaler.step(optimizer); scaler.update()
            progress['next_sample'] += len(labels)
            batch = (progress['next_sample'] + args.batch_size - 1) // args.batch_size
            if batch % 50 == 0 or progress['next_sample'] == samples or PAUSE_FILE.exists():
                progress['active_seconds'] += time.perf_counter() - started; started = time.perf_counter()
                save_progress(recovery, model, optimizer, scaler, progress, config)
                print(f"Epoch {progress['epoch']+1}/{args.epochs}, batch {batch}, loss {float(loss.detach()):.4f}", flush=True)
            check_pause()
        logits, targets = logits_for(model, val_rows, 'cuda', args.batch_size)
        value = validation_loss(logits, targets, val_rows)
        metrics = domain_metrics(logits, targets, val_rows)
        progress['history'].append({'epoch': progress['epoch'] + 1, 'validation_loss': value,
                                    'known_f1': {d: m['known_macro_f1_present'] for d, m in metrics.items()}})
        village = metrics.get('PlantVillage', {}).get('known_macro_f1_present', 0.)
        if value < progress['best_loss'] and village >= args.min_village_f1:
            progress['best_loss'] = value; progress['best_state'] = cpu_state(model)
            progress['selected_stage'] = f"continued/{progress['epoch']+1}"
        print(progress['history'][-1], flush=True)
        progress['epoch'] += 1; progress['next_sample'] = 0; progress['order'] = []
        progress['active_seconds'] += time.perf_counter() - started; started = time.perf_counter()
        save_progress(recovery, model, optimizer, scaler, progress, config)
    model.load_state_dict(progress['best_state'])
    logits, targets = logits_for(model, val_rows, 'cuda', args.batch_size)
    temperature, calibration = calibrate_balanced(logits, targets, val_rows) if args.balanced_calibration else calibrate(logits, targets)
    report = {'config': config, 'architecture': FIELD_ARCHITECTURE, 'selected_stage': progress['selected_stage'],
              'history': progress['history'], 'validation': domain_metrics(logits, targets, val_rows, temperature),
              'temperature': temperature, 'training_seconds': progress['active_seconds'] + time.perf_counter() - started,
              'test_used_for_selection': False, 'test_evaluated': False,
              'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'parent': str(args.source), 'initial_optimizer': 'fresh AdamW warm-start from parent weights; subsequent recovery restores this run optimizer',
              'recovery': 'This new run saves model, optimizer, scaler, sampler order/cursor and RNG every 50 batches and on cooperative pause.'}
    atomic_save({'architecture': FIELD_ARCHITECTURE, 'class_names': FIELD_CLASSES,
                 'preprocessing': FIELD_PREPROCESSING, 'state_dict': progress['best_state'],
                 'temperature': temperature, 'calibration': calibration, 'training': report}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved continued candidate:', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('models/mobilenetv3_field_20261103.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/mobilenetv3_field_continued_20261103.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/field_mobile_v1'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--expanded-data', action='store_true', help='Require parent-bound expansion audit and preserved holdouts')
    parser.add_argument('--samples-per-epoch', type=int)
    parser.add_argument('--unfreeze-blocks', type=int, default=3)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--top-backbone-lr', type=float)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--min-village-f1', type=float, default=0.)
    parser.add_argument('--balanced-calibration', action='store_true')
    train(parser.parse_args())
