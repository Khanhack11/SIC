"""Unit tests for agronomy profiles, weather advisory, and agricultural chatbot."""
import unittest
from fastapi.testclient import TestClient

from backend.main import app, advisory_engine


class AgronomyAndChatTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_agronomy_detail_returns_soil_env_and_prescriptions(self):
        res = self.client.get('/api/diseases/Tomato_Late_blight/agronomy')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        agro = data['agronomy_data']
        self.assertEqual(agro['disease_id'], 'Tomato_Late_blight')
        self.assertIn('soil_requirements', agro)
        self.assertIn('optimal_ph', agro['soil_requirements'])
        self.assertIn('environmental_conditions', agro)
        self.assertIn('pest_vectors_and_co_infections', agro)
        self.assertIn('comprehensive_curative_solutions', agro)
        solutions = agro['comprehensive_curative_solutions']
        self.assertTrue(len(solutions['chemical_prescriptions']) >= 2)
        self.assertIn('Ridomil Gold 68WG', solutions['chemical_prescriptions'][0]['trade_names'])
        self.assertIn('exact_dosage', solutions['chemical_prescriptions'][0])

    def test_weather_advisory_detects_late_blight_risk_in_cool_humid_conditions(self):
        # 18°C and 94% humidity triggers Tomato Late Blight risk
        res = self.client.post('/api/weather-advisory', json={'temperature_c': 18.0, 'humidity_pct': 94.0, 'crop': 'tomato'})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['status'], 'ALERT')
        self.assertGreaterEqual(data['total_risks_detected'], 1)
        alert_diseases = [a['disease_id'] for a in data['alerts']]
        self.assertIn('Tomato_Late_blight', alert_diseases)

    def test_weather_advisory_reports_safe_in_dry_warm_conditions(self):
        # 30°C and 45% humidity is safe from Late Blight
        res = self.client.post('/api/weather-advisory', json={'temperature_c': 30.0, 'humidity_pct': 45.0, 'crop': 'tomato'})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['status'], 'SAFE')
        self.assertEqual(data['total_risks_detected'], 0)

    def test_agricultural_chatbot_answers_dosage_and_soil_questions(self):
        # Test dosage query (with or without accents)
        res1 = self.client.post('/api/chat', json={'message': 'Lieu luong thuoc tri benh suong mai ca chua nhu the nao?'})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1['matched_disease'], 'Tomato_Late_blight')
        self.assertIn('Ridomil Gold', data1['answer'])
        self.assertIn('Liều lượng pha', data1['answer'])

        # Test soil query
        res2 = self.client.post('/api/chat', json={'message': 'Đất trồng khoai tây cần độ pH và độ ẩm bao nhiêu?'})
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertIn('Độ pH', data2['answer'])

    def test_dataset_insights_exposes_kaggle_comparison(self):
        res = self.client.get('/api/dataset-insights')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('kaggle_benchmark', data)
        self.assertEqual(data['kaggle_benchmark']['classes_count'], 38)
        self.assertIn('our_field_v2_dataset', data)
        self.assertEqual(data['our_field_v2_dataset']['total_records'], 20902)


if __name__ == '__main__':
    unittest.main()
