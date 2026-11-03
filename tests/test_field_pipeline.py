import unittest
import io
import tempfile
from pathlib import Path
from PIL import Image
import torch
from backend.mobile_model import build_field_model, field_transform, FIELD_CLASSES, letterbox
from backend.mobile_model import FIELD_ARCHITECTURE, FIELD_PREPROCESSING
from backend.field_inference import FieldInferenceEngine
from backend.advisory_engine import AdvisoryKnowledgeBase
from ml.download_plantdoc import local_path
from ml.train_field import weights_for


class FieldPipelineTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_rgb_contract(self):
        image=Image.new('RGB',(200,100),'red')
        boxed=letterbox(image)
        self.assertEqual(boxed.size,(224,224))
        self.assertEqual(boxed.getpixel((112,0)),(114,114,114))
        self.assertEqual(boxed.getpixel((112,112)),(255,0,0))
        self.assertEqual(tuple(field_transform()(image).shape),(3,224,224))

    def test_training_sampler_balances_domains_and_classes(self):
        rows=[{'domain':'PlantVillage','label':'healthy'}]*100 + [{'domain':'PlantDoc','label':'other'}]*2
        weights=weights_for(rows)
        self.assertAlmostEqual(float(weights[:100].sum()),float(weights[100:].sum()))

    def test_windows_invalid_source_names_preserve_original_provenance(self):
        entry={'path':'train/Potato leaf/x.jpg?query=1.jpg','sha':'abcdef'}
        self.assertEqual(local_path(entry).name,'abcdef.jpg')
        self.assertIn('?',entry['path'])

    def test_candidate_factory_has_separate_reject_class(self):
        model=build_field_model().eval()
        with torch.inference_mode():values=model(torch.zeros(1,3,64,64))
        self.assertEqual(tuple(values.shape),(1,len(FIELD_CLASSES)))
        self.assertEqual(FIELD_CLASSES[-1],'Unsupported')

    def test_serving_checks_contract_and_routes_reject_to_review(self):
        model=build_field_model()
        with torch.no_grad():
            model.classifier[-1].weight.zero_();model.classifier[-1].bias.fill_(-10)
            model.classifier[-1].bias[-1]=10
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.pt'
            checkpoint={'architecture':FIELD_ARCHITECTURE,'class_names':FIELD_CLASSES,
                'preprocessing':FIELD_PREPROCESSING,'temperature':1.,
                'state_dict':model.state_dict(),'calibration':{'fixture':True}}
            torch.save(checkpoint,path)
            engine=FieldInferenceEngine(path)
            self.assertTrue(engine.weights_loaded)
            content=io.BytesIO();Image.new('RGB',(80,40),'green').save(content,format='PNG')
            prediction=engine.predict(content.getvalue())
            self.assertEqual(prediction['predicted_class'],'Unsupported')
            self.assertEqual(AdvisoryKnowledgeBase().process_diagnosis(prediction)['diagnosis_status'],'UNCERTAIN')
            checkpoint['preprocessing']='incorrect'
            torch.save(checkpoint,path)
            self.assertFalse(FieldInferenceEngine(path).weights_loaded)
