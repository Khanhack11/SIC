"""Download only Healthy images from CornLeafDiseaseCollection V1, checking publisher hashes."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from ml.download_recommended_datasets import ROOT, save_json, transfer


def publisher_json(url):
    with urlopen(Request(url, headers={'User-Agent': 'SIC-dataset-audit/1.0'}), timeout=30) as response:
        return json.load(response)


def download(root):
    folder = root / 'corn_ecuador_healthy_v1'
    files_path = folder / 'source_files.json'
    if files_path.exists():
        files = json.loads(files_path.read_text(encoding='utf-8'))
    else:
        files = publisher_json('https://data.mendeley.com/public-api/datasets/w56xxnykcc/files?'
                               'folder_id=5ef4f29e-96cc-4445-80f0-1b63fc13b726&version=1&$start=0&$limit=1000')
        save_json(files_path, files)
    expected = {f'healthy_{index:03}.jpg' for index in range(1, 233)}
    if len(files) != 232 or {item['filename'] for item in files} != expected:
        raise ValueError('Publisher Healthy V1 inventory differs from the audited 232 files')
    metadata_path = folder / 'publisher_metadata.json'
    if not metadata_path.exists():
        save_json(metadata_path, publisher_json(
            'https://data.mendeley.com/api/datasets-v2/datasets/w56xxnykcc?fields=repository&version=1'))
    for item in files:
        details = item['content_details']
        transfer(details['download_url'], folder/'original'/'Healthy'/item['filename'],
                 expected_size=details['size'], expected_hash=details['sha256_hash'])
    save_json(folder/'source.json', {
        'homepage': 'https://data.mendeley.com/datasets/w56xxnykcc/1',
        'doi': '10.17632/w56xxnykcc.1',
        'subset': 'Healthy only; ambiguous common-rust/Southern-rust labels excluded',
        'files': 232, 'sha256_upstream_verified': True})
    print('Verified 232 Healthy images against publisher SHA-256 hashes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    download(parser.parse_args().root)
