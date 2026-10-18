"""Download a pinned RGB PlantVillage subset without downloading other crops.

Run: python -m ml.download_dataset --output data/raw/plantvillage
Existing files are verified against Git's blob SHA-1 before being reused.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from ml.labels import PLANTVILLAGE_LABELS

REPOSITORY = "spMohanty/PlantVillage-Dataset"
REVISION = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
API = f"https://api.github.com/repos/{REPOSITORY}/git/trees/"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"


def fetch(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "SIC-data-pipeline/1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def tree(revision):
    result = json.loads(fetch(API + revision))
    if result.get("truncated"):
        raise ValueError("GitHub returned a truncated tree; refusing an incomplete dataset.")
    return result["tree"]


def blob_hash(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def download(output, cap=1500, seed=42, workers=16):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    roots = {entry["path"]: entry for entry in tree(REVISION)}
    raw = {entry["path"]: entry for entry in tree(roots["raw"]["sha"])}
    color = {entry["path"]: entry for entry in tree(raw["color"]["sha"])}
    leaf_bytes = fetch(RAW + "leaf-map.json")
    if blob_hash(leaf_bytes) != roots["leaf-map.json"]["sha"]:
        raise ValueError("Leaf map hash does not match the pinned repository revision.")
    (output / "leaf-map.json").write_bytes(leaf_bytes)
    selected = []
    counts = {}
    for folder in PLANTVILLAGE_LABELS:
        entries = sorted(
            (entry for entry in tree(color[folder]["sha"])
             if entry["type"] == "blob" and Path(entry["path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=lambda entry: entry["path"],
        )
        random.Random(f"{seed}:{folder}").shuffle(entries)
        entries = entries[:cap] if cap else entries
        counts[folder] = len(entries)
        for entry in entries:
            selected.append((folder, entry))
    print(f"Selected {len(selected)} images across {len(counts)} classes", flush=True)

    def transfer(item):
        folder, entry = item
        path = output / folder / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and blob_hash(path.read_bytes()) == entry["sha"]:
            return
        content = fetch(RAW + quote(f"raw/color/{folder}/{entry['path']}", safe="/"))
        if blob_hash(content) != entry["sha"]:
            raise ValueError(f"Image hash mismatch: {path}")
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(path)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transfer, item): item for item in selected}
        for index, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
            except Exception as exc:
                folder, entry = pending[future]
                failures.append({"path": f"{folder}/{entry['path']}", "error": str(exc)})
            if index % 500 == 0 or index == len(selected):
                print(f"Downloaded/verified {index}/{len(selected)}; failures={len(failures)}", flush=True)
    provenance = {
        "repository": f"https://github.com/{REPOSITORY}", "revision": REVISION,
        "variant": "raw/color", "seed": seed, "max_per_class": cap,
        "selected_counts": counts, "hash_verification": "git blob SHA-1",
        "leaf_map_sha256": hashlib.sha256(leaf_bytes).hexdigest(),
        "files": [{"path": f"{folder}/{entry['path']}", "git_blob_sha1": entry["sha"]}
                  for folder, entry in selected],
        "failures": failures,
    }
    (output / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to resume verified files.")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--max-per-class", type=int, default=1500, help="0 means all target images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.max_per_class < 0 or args.workers < 1:
        parser.error("max-per-class must be nonnegative and workers must be positive")
    download(args.output, args.max_per_class, args.seed, args.workers)


if __name__ == "__main__":
    main()
