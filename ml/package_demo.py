"""Package the runnable source and deployed weights, excluding datasets and environments."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from backend.config import BASE_DIR, MODEL_WEIGHTS_PATH, ONNX_MODEL_PATH


def package(output):
    files = []
    for folder in ['backend', 'ml', 'tests', 'sample_images/demo', 'docs/final_release', 'docs/data_optimization']:
        files.extend(p for p in (BASE_DIR / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and 'node_modules' not in p.parts
                     and not p.is_relative_to(BASE_DIR / 'tests/results'))
    diagram_folder = BASE_DIR / '.archify/architecture-sic-pipeline-20261103-114010'
    files.extend(p for p in diagram_folder.rglob('*') if p.is_file())
    for name in ['frontend_demo.html', 'frontend.js', 'crop.js', 'run_demo.cmd', 'run_mobile_demo.cmd', 'README.md',
                 'requirements.txt', 'requirements-demo.txt', 'requirements-onnx.txt',
                 'data/advisory_knowledge_base.json',
                 'docs/API_AND_DATABASE_SPECIFICATION.md', 'docs/MOBILE_FIELD_PHASES.md', 'docs/FIELD_RELEASE_CRITERIA.json']:
        files.append(BASE_DIR / name)
    files.extend([MODEL_WEIGHTS_PATH, ONNX_MODEL_PATH])
    for name in ['small_cnn_20261031.pt', 'small_cnn_20261031.json',
                 'mobilenetv2_gpu_20261031.pt', 'mobilenetv2_gpu_20261031.json',
                 'mobilenetv2_gpu_20261103.pt', 'mobilenetv2_gpu_20261103.json',
                 'mobilenetv3_field_20261103.pt', 'mobilenetv3_field_20261103.json']:
        candidate = BASE_DIR / 'models' / name
        if candidate.is_file():
            files.append(candidate)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    files = sorted(set(files))
    manifest = {str(p.relative_to(BASE_DIR)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(BASE_DIR)))
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    with ZipFile(output) as archive:
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f'Package verification failed: {name}')
    print(f'Created {output}: {len(files)} files, {output.stat().st_size / 1024**2:.1f} MB; all checksums verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/SIC_demo_20261104.zip'))
    package(parser.parse_args().output)
