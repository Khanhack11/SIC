"""
Unit tests for Agronomy AI Chatbot and Weather Epidemic Early Warning APIs.
"""

import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.chatbot_engine import AgronomyChatbot
from backend.weather_advisory import get_agro_weather_advisory, calculate_disease_risks


class ChatbotAndWeatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.chatbot = AgronomyChatbot()

    def test_chatbot_greeting(self):
        res = self.chatbot.chat("Xin chào")
        self.assertIn("reply", res)
        self.assertIn("Trợ lý", res["reply"])
        self.assertIn("SIC", res["reply"])
        self.assertIn("suggestions", res)

    def test_chatbot_disease_and_dosage_inquiry(self):
        res = self.chatbot.chat("Bệnh mốc sương trên cà chua dùng thuốc gì và liều lượng bao nhiêu?")
        self.assertIn("reply", res)
        self.assertEqual(res["disease_id"], "Tomato_Late_blight")
        self.assertIn("MỐC SƯƠNG", res["reply"].upper())
        self.assertIn("Liều lượng", res["reply"])
        self.assertIn("suggestions", res)

    def test_chatbot_chemical_inquiry(self):
        res = self.chatbot.chat("Bệnh mốc sương cà chua phun Ridomil Gold liều lượng thế nào?")
        self.assertIn("reply", res)
        self.assertIn("RIDOMIL GOLD", res["reply"].upper())
        self.assertIn("16 lít", res["reply"])

    def test_chatbot_api_endpoint(self):
        response = self.client.post("/api/chat", json={"message": "Cây ngô bị bệnh gỉ sắt"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("reply", data)
        self.assertEqual(data["disease_id"], "Corn_Common_rust")

    def test_weather_regions_endpoint(self):
        response = self.client.get("/api/weather/regions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("regions", data)
        self.assertGreaterEqual(len(data["regions"]), 3)

    def test_weather_advisory_endpoint(self):
        response = self.client.get("/api/weather/advisory?region=thai_nguyen")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("region_name", data)
        self.assertIn("weather", data)
        self.assertIn("disease_risks", data)
        self.assertIn("overall_advisory", data)
        # Check risk structure
        risks = data["disease_risks"]
        self.assertGreater(len(risks), 0)
        for r in risks:
            self.assertIn("disease_id", r)
            self.assertIn("risk_score", r)
            self.assertIn("risk_level", r)
            self.assertIn("preventive_action", r)


if __name__ == "__main__":
    unittest.main()
