"""
Unit tests for automatic leaf detection & auto-diagnose API endpoints.
"""
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.leaf_detector import detect_leaf_regions, auto_crop_leaf


class LeafDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.demo_dir = Path("sample_images/demo")

    def test_detect_leaf_regions_returns_boxes_and_preview(self):
        img_path = self.demo_dir / "Tomato_Early_blight.jpg"
        self.assertTrue(img_path.exists())
        img_bytes = img_path.read_bytes()

        result = detect_leaf_regions(img_bytes)
        self.assertIn("count", result)
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("primary_crop_str", result)
        self.assertIn("boxes", result)
        self.assertIn("annotated_image_preview", result)
        self.assertTrue(result["annotated_image_preview"].startswith("data:image/jpeg;base64,"))
        self.assertLess(result["execution_ms"], 1000)

        # Check box structure
        box = result["boxes"][0]
        self.assertIn("pixel_box", box)
        self.assertIn("norm_box", box)
        self.assertEqual(len(box["norm_box"]), 4)
        for coord in box["norm_box"]:
            self.assertGreaterEqual(coord, 0.0)
            self.assertLessEqual(coord, 1.0)

    def test_auto_crop_leaf_produces_valid_image(self):
        img_path = self.demo_dir / "Corn_Common_rust.jpg"
        img_bytes = img_path.read_bytes()

        cropped_bytes = auto_crop_leaf(img_bytes)
        self.assertGreater(len(cropped_bytes), 500)
        # Check JPEG magic bytes
        self.assertTrue(cropped_bytes.startswith(b"\xff\xd8\xff"))

    def test_detect_leaves_endpoint(self):
        img_path = self.demo_dir / "Corn_Common_rust.jpg"
        with img_path.open("rb") as f:
            response = self.client.post(
                "/api/detect-leaves",
                files={"file": ("corn.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)
        self.assertIn("primary_crop_str", data)
        self.assertIn("boxes", data)

    def test_auto_diagnose_endpoint(self):
        img_path = self.demo_dir / "Tomato_Early_blight.jpg"
        with img_path.open("rb") as f:
            response = self.client.post(
                "/api/auto-diagnose",
                files={"file": ("tomato.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("diagnosis_status", data)
        self.assertIn("detection_info", data)
        self.assertIn("prediction", data)
        self.assertIn("disease_id", data["prediction"])


if __name__ == "__main__":
    unittest.main()
