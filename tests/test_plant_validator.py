"""
Unit tests for Plant / Leaf Presence Validation & Out-Of-Distribution Guardrail.
Ensures non-plant images (human faces, furniture, empty walls) are rejected
and cannot produce false disease diagnoses or erroneous bounding boxes.
"""
import io
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.plant_validator import validate_plant_leaf
from backend.leaf_detector import detect_leaf_regions


def create_solid_color_jpeg(color_rgb, size=(300, 300)):
    """Creates a JPEG in memory with a given solid RGB color."""
    img = Image.new("RGB", size, color_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def create_human_face_synthetic_jpeg(size=(300, 300)):
    """
    Creates an image simulating human face skin tones
    (R > 180, G in [120, 160], B in [90, 130] with skin conditions).
    """
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # Background neutral gray/office
    arr[:, :] = [180, 185, 190]
    # Human face oval in center
    cy, cx = size[1] // 2, size[0] // 2
    y, x = np.ogrid[:size[1], :size[0]]
    mask = ((x - cx) ** 2) / (70 ** 2) + ((y - cy) ** 2) / (100 ** 2) <= 1
    # Vietnamese/East Asian skin tone: R=215, G=165, B=130
    arr[mask] = [215, 165, 130]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class PlantValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.demo_dir = Path("sample_images/demo")

    def test_real_crop_leaves_pass_validation(self):
        """Verifies real field crop leaves (Tomato, Corn, Potato) pass validation."""
        for filename in ["Tomato_Early_blight.jpg", "Corn_Common_rust.jpg", "Potato_Late_blight.jpg"]:
            img_path = self.demo_dir / filename
            if not img_path.exists():
                continue
            img_bytes = img_path.read_bytes()
            res = validate_plant_leaf(img_bytes)
            self.assertTrue(res["is_valid_plant"], f"Failed on real leaf: {filename}")
            self.assertIsNone(res["rejection_code"])
            self.assertGreater(res["plant_ratio"], 10.0)

    def test_non_plant_object_rejected(self):
        """Verifies office/room walls and neutral gray objects are rejected."""
        gray_bytes = create_solid_color_jpeg((170, 175, 180))
        res = validate_plant_leaf(gray_bytes)
        self.assertFalse(res["is_valid_plant"])
        self.assertEqual(res["rejection_code"], "NON_PLANT_OBJECT")

        white_bytes = create_solid_color_jpeg((240, 240, 240))
        res_white = validate_plant_leaf(white_bytes)
        self.assertFalse(res_white["is_valid_plant"])

    def test_human_skin_detected_and_rejected(self):
        """Verifies human face skin tones trigger HUMAN_DETECTED rejection."""
        face_bytes = create_human_face_synthetic_jpeg()
        res = validate_plant_leaf(face_bytes)
        self.assertFalse(res["is_valid_plant"])
        self.assertEqual(res["rejection_code"], "HUMAN_DETECTED")
        self.assertIn("khuôn mặt", res["message"].lower())

    def test_detect_leaves_returns_empty_boxes_for_human_face(self):
        """Verifies leaf detector does not hallucinate leaf boxes on human faces."""
        face_bytes = create_human_face_synthetic_jpeg()
        det = detect_leaf_regions(face_bytes)
        self.assertFalse(det.get("has_leaves", True))
        self.assertEqual(det.get("count", 0), 0)
        self.assertEqual(det.get("boxes", []), [])

    def test_diagnose_api_rejects_non_plant_with_zero_confidence(self):
        """Verifies /api/diagnose rejects non-plant images and gives 0.0% confidence."""
        face_bytes = create_human_face_synthetic_jpeg()
        response = self.client.post(
            "/api/diagnose?field_photo=true",
            files={"file": ("face_selfie.jpg", face_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("diagnosis_status"), "REJECTED_NON_PLANT")
        self.assertEqual(data.get("status"), "NON_PLANT_IMAGE")
        self.assertFalse(data.get("is_plant", True))
        self.assertEqual(data["prediction"]["confidence"], 0.0)
        self.assertEqual(data["prediction"]["disease_id"], "Non_plant_object")
        # Ensure no accidental disease recommendation
        self.assertNotIn("treatment_notice", data.get("advisory_details", {}))

    def test_auto_diagnose_api_rejects_human_face(self):
        """Verifies /api/auto-diagnose handles human face gracefully."""
        face_bytes = create_human_face_synthetic_jpeg()
        response = self.client.post(
            "/api/auto-diagnose?field_photo=true",
            files={"file": ("camera_person.jpg", face_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("diagnosis_status"), "REJECTED_NON_PLANT")
        self.assertFalse(data.get("is_plant", True))


if __name__ == "__main__":
    unittest.main()
