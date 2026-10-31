"""Select traceable validation examples for a live demo, never for measuring model quality."""

import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image

from backend.inference_engine import CropDiseaseInferenceEngine
from backend.preprocessing import orient_rgb
from ml.dataset import ManifestDataset


def main():
    torch.set_num_threads(2)
    dataset = ManifestDataset(Path('data/processed/plantvillage/validation.csv'))
    engine = CropDiseaseInferenceEngine(device='cpu', runtime='eager')
    if not engine.weights_loaded:
        raise RuntimeError(engine.load_error)
    names = {'Tomato_Early_blight': 'Cà chua, đốm vòng', 'Corn_Common_rust': 'Ngô, gỉ sắt',
             'Potato_healthy': 'Khoai tây, lá khỏe'}
    folder = Path('sample_images/demo')
    folder.mkdir(parents=True, exist_ok=True)
    selected = {}
    for row in dataset.rows:
        label = row['label']
        if (label not in names or label in selected) and 'uncertain' in selected:
            continue
        with Image.open(dataset.source_root / row['path']) as original:
            buffer = io.BytesIO()
            orient_rgb(original).save(buffer, format='JPEG', quality=95)
        content = buffer.getvalue()
        prediction = engine.predict(content)
        key = None
        if label in names and label not in selected and prediction['predicted_class'] == label and prediction['confidence'] >= .85:
            key = label
        elif 'uncertain' not in selected and prediction['confidence'] < .70:
            key = 'uncertain'
        if key is None:
            continue
        filename = f'{key}.jpg'
        (folder / filename).write_bytes(content)
        selected[key] = {'filename': filename, 'caption': names.get(key, 'Ảnh khó, độ tin cậy thấp'),
                         'source_path': row['path'], 'label': label, 'split': 'validation',
                         'source_sha256': row['sha256'], 'demo_sha256': hashlib.sha256(content).hexdigest(),
                         'observed_prediction': prediction['predicted_class'], 'observed_confidence': prediction['confidence']}
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError('Could not find all demo scenarios; inspect validation examples')
    manifest = {'purpose': 'Examples selected from validation for demonstrating UI states, not an evaluation sample.',
                'source': 'https://github.com/spMohanty/PlantVillage-Dataset',
                'revision': '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a',
                'checkpoint_sha256': hashlib.sha256(Path('models/mobilenetv2_crop_disease.pt').read_bytes()).hexdigest(),
                'images': list(selected.values())}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: item['observed_confidence'] for key, item in selected.items()}), flush=True)


if __name__ == '__main__':
    main()
