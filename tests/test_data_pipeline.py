"""Data correctness checks using synthetic fixtures, never performance claims."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from unittest.mock import patch

from backend.config import CLASS_NAMES
from backend.inference_engine import (CropDiseaseInferenceEngine, ModelNotReadyError,
                                      InputValidationError, preprocess_image, validate_image_file)
from backend.preprocessing import image_transform, orient_rgb
from ml.evaluate import classification_metrics
from ml.labels import PLANTVILLAGE_LABELS
from ml.prepare_dataset import prepare


class DataPipelineTests(unittest.TestCase):
    def test_mapping_covers_backend_labels_exactly(self):
        self.assertEqual(set(PLANTVILLAGE_LABELS.values()), set(CLASS_NAMES))
        self.assertEqual(len(PLANTVILLAGE_LABELS), len(CLASS_NAMES))

    def test_preprocessing_matches_serving_and_is_deterministic(self):
        image = Image.new("RGB", (110, 80), (180, 25, 90))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        serving = preprocess_image(buffer.getvalue())
        training_eval = image_transform()(orient_rgb(image)).unsqueeze(0)
        self.assertEqual(tuple(serving.shape), (1, 3, 224, 224))
        self.assertTrue(torch.equal(serving, training_eval))
        self.assertTrue(torch.equal(serving, preprocess_image(buffer.getvalue())))
        torch.manual_seed(7)
        first = image_transform(True)(image)
        torch.manual_seed(7)
        self.assertTrue(torch.equal(first, image_transform(True)(image)))

    def test_invalid_resolution_and_disguised_content(self):
        for image, format_name in ((Image.new("RGB", (10, 10)), "PNG"),
                                   (Image.new("RGB", (64, 64)), "GIF")):
            buffer = io.BytesIO()
            image.save(buffer, format=format_name)
            with self.assertRaises(InputValidationError):
                validate_image_file(buffer.getvalue(), "disguised.png")

    def test_missing_weights_cannot_produce_a_diagnosis(self):
        engine = CropDiseaseInferenceEngine(weights_path=None, device="cpu")
        with self.assertRaises(ModelNotReadyError):
            engine.predict(b"not important without a model")

    def test_checkpoint_label_order_mismatch_is_rejected(self):
        temporary_root = Path(__file__).resolve().parents[1] / "artifacts" / "test_temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            path = Path(temporary) / "wrong_order.pt"
            torch.save({"state_dict": {}, "class_names": list(reversed(CLASS_NAMES))}, path)
            engine = CropDiseaseInferenceEngine(weights_path=path, device="cpu")
            self.assertFalse(engine.weights_loaded)
            self.assertIn("class_names", engine.load_error)

    def test_api_reports_missing_model_as_503(self):
        from fastapi.testclient import TestClient
        from backend.main import app, inference_engine
        image = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image, format="PNG")
        with patch.object(inference_engine, "weights_loaded", False), TestClient(app) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "MODEL_NOT_READY")
            response = client.post("/api/diagnose", files={"file": ("leaf.png", image.getvalue(), "image/png")})
            self.assertEqual(response.status_code, 503)

    def test_metrics_count_all_classes_and_confusion_orientation(self):
        report = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(report["accuracy"], .75)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + .8) / 2)

    def test_curation_is_reproducible_and_leaf_disjoint(self):
        temporary_root = Path(__file__).resolve().parents[1] / "artifacts" / "test_temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            source = root / "raw"
            leaf_map = {}
            for index, folder in enumerate(PLANTVILLAGE_LABELS):
                directory = source / folder
                directory.mkdir(parents=True)
                for leaf in range(10):
                    for view in range(2):
                        identifier = f"leaf{index}_{leaf}_{view}"
                        leaf_map[identifier] = [f"{folder}:::{leaf}"]
                        name = f"uuid___{identifier}.png"
                        Image.new("RGB", (48, 64), (index * 20, leaf * 20, view * 50)).save(directory / name)
            first_folder = source / next(iter(PLANTVILLAGE_LABELS))
            original = next(first_folder.glob("*.png"))
            (first_folder / original.name.replace(".png", "copy.png")).write_bytes(original.read_bytes())
            (first_folder / "broken.jpg").write_bytes(b"broken")
            (source / "leaf-map.json").write_text(json.dumps(leaf_map), encoding="utf-8")
            report = prepare(source, root / "one")
            repeated = prepare(source, root / "two")
            self.assertEqual(report["curated_count"], 220)
            self.assertEqual(report["excluded_reasons"], {"duplicate_rgb_pixels": 1, "invalid_image": 1})
            self.assertEqual(report["manifest_sha256"], repeated["manifest_sha256"])
            groups = []
            for split in ("train", "validation", "test"):
                with (root / "one" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                groups.append({row["group"] for row in rows})
                self.assertEqual({row["label"] for row in rows}, set(CLASS_NAMES))
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            with self.assertRaises(ValueError):
                prepare(source, root / "one")


if __name__ == "__main__":
    unittest.main()
