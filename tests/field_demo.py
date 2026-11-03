"""Exercise the optional trained V3 through the real API, and measure warm CPU latency."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SIC_FIELD_MODEL_PATH'] = str(ROOT / 'models/mobilenetv3_field_20261103.pt')

from fastapi.testclient import TestClient
from backend.main import app, inference_engine


def main():
    output = ROOT / 'tests/results/field_demo'
    output.mkdir(parents=True, exist_ok=True)
    samples = sorted((ROOT / 'sample_images/demo').glob('*.jpg'))
    results = []
    with TestClient(app) as client:
        health = client.get('/api/health').json()
        assert health['weights_loaded'] and 'V3' in health['model_architecture'], health
        for sample in samples:
            raw = sample.read_bytes()
            response = client.post('/api/diagnose', files={'file': (sample.name, raw, 'image/jpeg')})
            assert response.status_code == 200, response.text
            result = response.json()
            if result['prediction']['is_healthy']:
                assert result['review_reason'] == 'field_healthy_unverified', result
            assert result['diagnostic_confirmation'] is False
            results.append({'file': sample.name, 'prediction': result['prediction'], 'review_reason': result['review_reason']})
        sample = samples[0]
        cropped = client.post('/api/diagnose?crop_box=0.1,0.1,0.8,0.8&visible_damage=true',
                              files={'file': (sample.name, sample.read_bytes(), 'image/jpeg')})
        assert cropped.status_code == 200, cropped.text
        bad = client.post('/api/diagnose', files={'file': ('bad.jpg', b'not an image', 'image/jpeg')})
        assert bad.status_code == 400, bad.text
        missing_crop = client.post('/api/diagnose?crop_box=0,0,0.0001,0.0001',
                                   files={'file': (sample.name, sample.read_bytes(), 'image/jpeg')})
        assert missing_crop.status_code == 400, missing_crop.text
    raw = samples[0].read_bytes()
    for _ in range(3): inference_engine.predict(raw)
    latencies = [inference_engine.predict(raw)['latency_ms'] for _ in range(30)]
    screenshot = ROOT / 'artifacts/field_rebuild/reported_leaf.jpg'
    qualitative = None
    if screenshot.is_file():
        qualitative = inference_engine.predict(screenshot.read_bytes())
        qualitative['scope'] = 'Screenshot crop only, not original upload; no expert label, not a quality metric or training sample.'
    report = {'health': health, 'samples': results, 'api_crop_valid': True, 'api_invalid_image_rejected': True,
              'api_tiny_crop_rejected': True, 'qualitative_user_screenshot': qualitative,
              'weights_sha256': hashlib.sha256(Path(os.environ['SIC_FIELD_MODEL_PATH']).read_bytes()).hexdigest(),
              'warm_cpu_latency_ms': {'n': len(latencies), 'median': statistics.median(latencies),
                                      'p95': sorted(latencies)[28], 'max': max(latencies)},
              'latency_scope': 'One JPEG fixture, 3 warmups then 30 sequential calls; decode+preprocess+CNN only. Not network, phone, concurrency or cold startup.'}
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Optional V3 API checks passed:', report['warm_cpu_latency_ms'])


if __name__ == '__main__':
    main()
