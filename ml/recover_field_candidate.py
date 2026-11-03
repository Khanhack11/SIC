"""Calibrate a completed-epoch checkpoint after an interrupted field run.

This does not resume an optimizer or claim that the interrupted epoch finished.
Only the saved, validation-selected weights and validation rows are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from backend.mobile_model import FIELD_ARCHITECTURE, FIELD_CLASSES, FIELD_PREPROCESSING, build_field_model
from ml.train import calibrate
from ml.train_field import domain_metrics, logits_for
from ml.train_gpu import atomic_save


def recover(source, output, data):
    if output.exists():
        raise ValueError('Preserve existing candidate; choose a new output')
    torch.set_num_threads(2)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    config = saved['config']
    manifest = data / 'records.json'
    audit = json.loads((data / 'audit.json').read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != config['records_sha256'] or digest != audit['records_sha256']:
        raise ValueError('Manifest differs from the original training run')
    for name, key in [('ml/train_field.py', 'script_sha256'), ('backend/mobile_model.py', 'transform_sha256')]:
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != config[key]:
            raise ValueError('Original training contract changed: ' + name)
    rows = [r for r in json.loads(manifest.read_text()) if r['split'] == 'validation']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_field_model().to(device).eval()
    model.load_state_dict(saved['state_dict'])
    logits, targets = logits_for(model, rows, device, 32)
    temperature, calibration = calibrate(logits, targets)
    report = {
        'config': config, 'architecture': FIELD_ARCHITECTURE,
        'selected_stage': saved['stage'], 'history': saved['history'],
        'validation': domain_metrics(logits, targets, rows, temperature),
        'temperature': temperature, 'training_seconds': None,
        'test_used_for_selection': False, 'training_interrupted': True,
        'recovery': {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'completed_finetune_epochs': sum(r['stage'] == 'finetune' for r in saved['history']),
                     'note': 'Epoch 3 interrupted; unsaved partial updates discarded. No optimizer or batch resume.'},
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'torch': str(torch.__version__),
        'pretraining': 'Torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2',
        'head': '960 -> 256 Hardswish Dropout(0.2) -> 12',
    }
    atomic_save({'architecture': FIELD_ARCHITECTURE, 'class_names': FIELD_CLASSES,
                 'preprocessing': FIELD_PREPROCESSING, 'state_dict': saved['state_dict'],
                 'temperature': temperature, 'calibration': calibration, 'training': report}, output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Recovered candidate:', output, 'selected:', saved['stage'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/training_runs/recovered/model.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/field_mobile_v1'))
    args = parser.parse_args()
    recover(args.source, args.output, args.data)
