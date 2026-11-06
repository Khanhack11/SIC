"""
Unit tests for Crop Scope Guardrails.
Verifies that the chatbot strictly enforces the 3 trained target crops rule:
- Allowed: Tomato (Cà chua), Potato (Khoai tây), Corn/Maize (Ngô/Bắp)
- Rejected: All out-of-scope crops (e.g. Lúa, Sầu riêng, Cà phê, Cam/Bưởi, Dưa hấu, Ớt...)
"""

import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.crop_guardrails import detect_crop_scope, SUPPORTED_CROPS
from backend.chatbot_engine import AgronomyChatbot


class CropScopeGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.chatbot = AgronomyChatbot()

    def test_supported_tomato_in_scope(self):
        res = detect_crop_scope("Bệnh mốc sương trên cà chua dùng thuốc gì?")
        self.assertTrue(res["is_in_scope"])
        self.assertEqual(res["matched_supported_crop"], "tomato")
        self.assertIsNone(res["detected_crop"])

    def test_supported_potato_in_scope(self):
        res = detect_crop_scope("Đất trồng khoai tây cần độ pH bao nhiêu?")
        self.assertTrue(res["is_in_scope"])
        self.assertEqual(res["matched_supported_crop"], "potato")
        self.assertIsNone(res["detected_crop"])

    def test_supported_corn_in_scope(self):
        res = detect_crop_scope("Cách phòng trừ bệnh gỉ sắt trên cây ngô?")
        self.assertTrue(res["is_in_scope"])
        self.assertEqual(res["matched_supported_crop"], "corn")
        self.assertIsNone(res["detected_crop"])

    def test_supported_disease_without_explicit_crop_in_scope(self):
        res = detect_crop_scope("Phun thuốc Ridomil Gold trị sương mai liều lượng bao nhiêu?")
        self.assertTrue(res["is_in_scope"])

    def test_out_of_scope_rice_rejected(self):
        res = detect_crop_scope("Cây lúa bị bệnh đạo ôn và khô vằn dùng thuốc gì?")
        self.assertFalse(res["is_in_scope"])
        self.assertIsNotNone(res["detected_crop"])
        self.assertIn("Lúa", res["detected_crop"])
        self.assertIn("Cà chua", res["refusal_message"])
        self.assertIn("Khoai tây", res["refusal_message"])
        self.assertIn("Ngô", res["refusal_message"])

    def test_out_of_scope_durian_rejected(self):
        res = detect_crop_scope("Cây sầu riêng bị nứt thân xì mủ phun thuốc gì?")
        self.assertFalse(res["is_in_scope"])
        self.assertIn("Sầu riêng", res["detected_crop"])
        self.assertIn("TỪ CHỐI TƯ VẤN", res["refusal_message"])

    def test_out_of_scope_citrus_rejected(self):
        res = detect_crop_scope("Bưởi da xanh bị vàng lá gân xanh và loét cam chữa thế nào?")
        self.assertFalse(res["is_in_scope"])
        self.assertIn("TỪ CHỐI TƯ VẤN", res["refusal_message"])

    def test_out_of_scope_watermelon_rejected(self):
        res = detect_crop_scope("Trị bọ trĩ trên dưa hấu")
        self.assertFalse(res["is_in_scope"])
        self.assertIn("Dưa hấu", res["detected_crop"])

    def test_non_agricultural_rejected(self):
        res = detect_crop_scope("Viết code Python giải phương trình bậc hai")
        self.assertFalse(res["is_in_scope"])
        self.assertEqual(res["reason"], "non_agricultural")
        self.assertIn("NGOÀI PHẠM VI", res["refusal_message"])

    def test_chatbot_refusal_response_via_chat_engine(self):
        # Direct chat method call
        chat_res = self.chatbot.chat("Cây cà phê bị rỉ sắt và khô cành chữa sao?")
        self.assertFalse(chat_res.get("is_in_scope", True))
        self.assertIn("TỪ CHỐI TƯ VẤN", chat_res["reply"])
        self.assertIn("Cà phê", chat_res["reply"])
        self.assertIn("Cà chua", chat_res["reply"])
        self.assertIn("Khoai tây", chat_res["reply"])
        self.assertIn("Ngô", chat_res["reply"])

    def test_chatbot_api_endpoint_rejects_out_of_scope(self):
        # API request
        response = self.client.post("/api/chat", json={"message": "Cây lúa bị đạo ôn cổ bông dùng thuốc gì"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data.get("is_in_scope", True))
        self.assertIn("TỪ CHỐI TƯ VẤN", data["reply"])
        self.assertIn("Khuyến nông", data["reply"])


if __name__ == "__main__":
    unittest.main()
