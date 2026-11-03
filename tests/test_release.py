import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app, advisory_engine, inference_engine


class ReleaseTests(unittest.TestCase):
    def test_unreviewed_treatments_are_withheld_without_mutating_database(self):
        record = advisory_engine.get_disease_by_id('Tomato_Early_blight')
        self.assertEqual(record['chemical_controls'], [])
        self.assertEqual(record['organic_biological_controls'], [])
        self.assertTrue(record['sources'])
        self.assertFalse(record['review']['expert_approved'])
        self.assertTrue(advisory_engine.diseases['Tomato_Early_blight']['chemical_controls'])
        record['sources'].clear()
        self.assertTrue(advisory_engine.get_disease_by_id('Tomato_Early_blight')['sources'])

    def test_every_class_has_traceable_reference_and_review_scope(self):
        for key in advisory_engine.diseases:
            record = advisory_engine.get_disease_by_id(key)
            self.assertTrue(record['sources'], key)
            self.assertTrue(all(s['url'].startswith('https://') for s in record['sources']))
            self.assertTrue(record['review']['scope'])

    def test_frontend_and_health_are_served_by_the_api(self):
        with TestClient(app) as client:
            response = client.get('/app')
            self.assertEqual(response.status_code, 200)
            self.assertIn('frontend.js', response.text)
            self.assertIn('capture="environment"', response.text)
            self.assertEqual(client.get('/frontend.js').status_code, 200)
            self.assertEqual(client.get('/crop.js').status_code, 200)
            self.assertEqual(client.get('/frontend_demo.html').text, response.text)
            self.assertIn(client.get('/api/health').json()['inference_runtime'], ['eager', 'torchscript', 'onnx'])
            self.assertEqual(client.get('/api/diseases/Tomato_Early_blight').json()['advisory_data']['chemical_controls'], [])

    def test_internal_error_is_not_exposed_to_the_client(self):
        with TestClient(app) as client, patch.object(inference_engine, 'predict', side_effect=RuntimeError('secret-internal-path')):
            import io
            from PIL import Image
            image = io.BytesIO()
            Image.new('RGB', (64, 64), 'green').save(image, format='PNG')
            response = client.post('/api/diagnose', files={'file': ('leaf.png', image.getvalue(), 'image/png')})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('secret-internal-path', response.text)


if __name__ == '__main__':
    unittest.main()
