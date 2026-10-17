"""Flag conservative cross-split perceptual matches for review, without changing splits."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ml.runtime import check_pause


class HashIndex:
    def __init__(self):
        self.root = None

    def add(self, value, record):
        if self.root is None:
            self.root = [value, [record], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(record)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [record], {}]
                return
            node = node[2][distance]

    def neighbors(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for record in node[1]:
                    yield distance, record
            pending.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)


def perceptual_hashes(image):
    # DCT low frequencies plus independent horizontal-gradient hash.
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64)
    gray = image.convert("L")
    orientations = [gray, *[gray.transpose(operation) for operation in
                    (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
                     Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                     Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE)]]
    values = []
    for oriented in orientations:
        pixels = np.asarray(oriented.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        coefficients = basis @ pixels @ basis.T
        bits = coefficients.ravel() > np.median(coefficients.ravel()[1:])
        bits[0] = False
        phash = int.from_bytes(np.packbits(bits).tobytes(), "big")
        small = np.asarray(oriented.resize((9, 8), Image.Resampling.LANCZOS))
        dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
        values.append((phash, dhash))
    return min(values)


def audit(data, source, output):
    index = HashIndex()
    examples, counts = [], Counter()
    manifests = {}
    scanned = 0
    expected = json.loads((data / "audit.json").read_text(encoding="utf-8"))["manifest_sha256"]
    for split in ("train", "validation", "test"):
        manifest = data / f"{split}.csv"
        manifests[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifests[split] != expected[split]:
            raise ValueError("Manifest changed after curation")
        with manifest.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                check_pause()
                content = (source / row["path"]).read_bytes()
                if hashlib.sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"Image changed after curation: {row['path']}")
                with Image.open(io.BytesIO(content)) as image:
                    phash, dhash = perceptual_hashes(ImageOps.exif_transpose(image).convert("RGB"))
                row.update(split=split, dhash=dhash)
                for distance, previous in index.neighbors(phash, 2):
                    if split == previous["split"] or row["group"] == previous["group"]:
                        continue
                    gradient_distance = (dhash ^ previous["dhash"]).bit_count()
                    if gradient_distance <= 4:
                        category = "same_label" if row["label"] == previous["label"] else "different_labels"
                        counts[category] += 1
                        if len(examples) < 200:
                            examples.append({"left": {key: previous[key] for key in ("path", "label", "group", "split")},
                                             "right": {key: row[key] for key in ("path", "label", "group", "split")},
                                             "phash_distance": distance, "dhash_distance": gradient_distance})
                index.add(phash, row)
                scanned += 1
                if scanned % 1000 == 0:
                    print(f"Scanned {scanned}; flagged pairs={sum(counts.values())}", flush=True)
    report = {"scanned_images": scanned, "manifest_sha256": manifests, "pair_counts": dict(counts),
              "examples": examples, "method": "canonical rotation/reflection; 63-bit DCT pHash <=2 and dHash <=4",
              "splits_modified": False, "limitations": ["Candidate matches need manual review; hashes do not prove leaf identity.",
                                                         "Conservative thresholds and canonical orientation can miss similar views.",
                                                         "No candidates does not prove the absence of physical-leaf leakage."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"images": scanned, "pair_counts": dict(counts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--source", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_optimization/results/optimization_20261017/near_duplicates.json"))
    args = parser.parse_args()
    audit(args.data, args.source, args.output)


if __name__ == "__main__":
    main()
