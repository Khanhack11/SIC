import io
import unittest
from PIL import Image
from backend.inference_engine import crop_upload, InputValidationError
from backend.advisory_engine import AdvisoryKnowledgeBase
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app,inference_engine


class MobileInputTests(unittest.TestCase):
    def test_crop_uses_uploaded_pixels_and_checks_bounds(self):
        image=Image.new('RGB',(200,100),'red')
        image.paste('blue',(100,0,200,100))
        data=io.BytesIO(); image.save(data,format='PNG')
        output=crop_upload(data.getvalue(),'0.5,0,0.5,1')
        with Image.open(io.BytesIO(output)) as cropped:
            self.assertEqual(cropped.size,(100,100))
            self.assertGreater(cropped.getpixel((50,50))[2],240)
        for box in ['0,0,0,1','0.8,0,0.5,1','NaN,0,1,1','0,0,1','0,0,.01,1']:
            with self.assertRaises(InputValidationError): crop_upload(data.getvalue(),box)

    def test_odd_dimensions_match_browser_rounding_without_extra_border(self):
        for dimensions, expected in [((95,97),(47,97)),((97,95),(48,95))]:
            image=Image.new('RGB',dimensions,(70,170,80));data=io.BytesIO();image.save(data,format='PNG')
            for box in ['0.5,0,0.5,1','0.5,0,0.5000001,1']:
                with Image.open(io.BytesIO(crop_upload(data.getvalue(),box))) as cropped:
                    self.assertEqual(cropped.size,expected)
                    self.assertGreater(cropped.getpixel((cropped.width-1,40))[1],150)

    def test_reported_damage_prevents_high_confidence_healthy_advice(self):
        result=AdvisoryKnowledgeBase().process_diagnosis({'predicted_class':'Corn_healthy',
            'confidence':.99999,'is_healthy':True,'visible_damage':True})
        self.assertEqual(result['diagnosis_status'],'UNCERTAIN')
        self.assertFalse(result['prediction']['confidence_gate_passed'])
        self.assertEqual(result['review_reason'],'visible_damage_conflict')

    def test_unknown_class_never_receives_confirmed_advisory(self):
        result=AdvisoryKnowledgeBase().process_diagnosis({'predicted_class':'Unsupported',
            'confidence':.99999,'is_healthy':False})
        self.assertEqual(result['diagnosis_status'],'UNCERTAIN')
        self.assertEqual(result['review_reason'],'unsupported_class')

    def test_api_crop_and_damage_options_use_the_processed_upload(self):
        image=Image.new('RGB',(200,100),'red');data=io.BytesIO();image.save(data,format='PNG')
        dimensions=[]
        def predict(content):
            with Image.open(io.BytesIO(content)) as crop: dimensions.append(crop.size)
            return {'predicted_class':'Corn_healthy','confidence':.9999,'is_healthy':True}
        with TestClient(app) as client,patch.object(inference_engine,'predict',side_effect=predict):
            response=client.post('/api/diagnose?crop_box=0.5,0,0.5,1&visible_damage=true',
                files={'file':('leaf.png',data.getvalue(),'image/png')})
            self.assertEqual(response.status_code,200)
            self.assertEqual(dimensions,[(100,100)])
            self.assertEqual(response.json()['review_reason'],'visible_damage_conflict')
            self.assertEqual(client.post('/api/diagnose?crop_box=bad',files={'file':('leaf.png',data.getvalue(),'image/png')}).status_code,400)
            result=client.post('/api/diagnose',files={'file':('field.png',data.getvalue(),'image/png')}).json()
            self.assertEqual(result['review_reason'],'field_healthy_unverified')
            self.assertEqual(result['diagnosis_status'],'UNCERTAIN')
            self.assertEqual(result['prediction']['confidence'],.9999)
