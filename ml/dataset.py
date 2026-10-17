"""Manifest-backed dataset; backend and training share exactly one input contract."""

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from backend.config import CLASS_NAMES
from backend.preprocessing import image_transform, orient_rgb


class ManifestDataset(Dataset):
    def __init__(self, manifest, source_root=None, training=False):
        self.manifest = Path(manifest).resolve()
        labels = json.loads((self.manifest.parent / "labels.json").read_text(encoding="utf-8"))
        if labels["class_names"] != CLASS_NAMES:
            raise ValueError("Manifest label order differs from backend.config.CLASS_NAMES")
        audit = json.loads((self.manifest.parent / "audit.json").read_text(encoding="utf-8"))
        name = self.manifest.stem
        if hashlib.sha256(self.manifest.read_bytes()).hexdigest() != audit["manifest_sha256"].get(name):
            raise ValueError("Manifest changed after audit; prepare a new dataset version.")
        self.source_root = Path(source_root or audit["source_root"]).resolve()
        with self.manifest.open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError("Manifest contains no images")
        self.transform = image_transform(training)
        for row in self.rows:
            candidate = (self.source_root / row["path"]).resolve()
            if not candidate.is_relative_to(self.source_root):
                raise ValueError("Manifest image path escapes source_root")
            if CLASS_NAMES[int(row["class_index"])] != row["label"]:
                raise ValueError("Manifest label/index mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.source_root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed after curation: {row['path']}")
        with Image.open(path) as image:
            tensor = self.transform(orient_rgb(image))
        return tensor, int(row["class_index"])
