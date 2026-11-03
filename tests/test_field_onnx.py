import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from backend.field_inference import FieldInferenceEngine
from backend.mobile_model import FIELD_CLASSES, FIELD_ARCHITECTURE, FIELD_PREPROCESSING, build_field_model


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX dependencies')
class FieldOnnxTests(unittest.TestCase):
    def test_export_parity_and_mismatched_checkpoint_are_checked(self):
        from ml.export_field_onnx import export
        torch.set_num_threads(2);torch.manual_seed(23)
        model=build_field_model()
        with torch.no_grad():
            model.classifier[-1].bias[-1]=5
        checkpoint={'architecture':FIELD_ARCHITECTURE,'class_names':FIELD_CLASSES,
                    'preprocessing':FIELD_PREPROCESSING,'temperature':1.2,
                    'calibration':{'fixture':True},'state_dict':model.state_dict()}
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'fixture.pt';graph=Path(folder)/'fixture.onnx'
            torch.save(checkpoint,weights);export(weights,graph)
            eager=FieldInferenceEngine(weights);onnx=FieldInferenceEngine(weights,onnx_path=graph)
            self.assertTrue(onnx.weights_loaded,onnx.load_error)
            self.assertEqual(onnx.runtime,'onnx')
            with torch.inference_mode():
                for batch in [1,4]:
                    images=torch.randn(batch,3,224,224)
                    self.assertTrue(torch.allclose(eager.logits(images),onnx.logits(images),atol=1e-4,rtol=1e-4))
            image=io.BytesIO();Image.new('RGB',(95,97),(60,130,70)).save(image,format='PNG')
            a=eager.predict(image.getvalue());b=onnx.predict(image.getvalue())
            self.assertEqual(a['predicted_class'],b['predicted_class'])
            self.assertAlmostEqual(a['confidence'],b['confidence'],places=5)
            checkpoint['temperature']=2.;torch.save(checkpoint,weights)
            invalid=FieldInferenceEngine(weights,onnx_path=graph)
            self.assertFalse(invalid.weights_loaded)
            self.assertIn('mismatch',invalid.load_error)
            with self.assertRaises(ValueError):export(weights,graph)
