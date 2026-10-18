from importlib.util import find_spec
from pathlib import Path
import unittest

import torch
from backend.inference_engine import CropDiseaseInferenceEngine


@unittest.skipUnless(find_spec("onnxruntime") and Path("models/mobilenetv2_crop_disease.onnx").exists()
                     and Path("models/mobilenetv2_crop_disease.pt").exists(), "Optional ONNX artifacts unavailable")
class OnnxRuntimeTests(unittest.TestCase):
    def test_batch_shapes_and_logits_agree_with_fp32_checkpoint(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            eager = CropDiseaseInferenceEngine(device="cpu", runtime="eager")
            onnx_engine = CropDiseaseInferenceEngine(device="cpu", runtime="onnx")
            self.assertTrue(onnx_engine.weights_loaded, onnx_engine.load_error)
            generator = torch.Generator().manual_seed(31)
            with torch.inference_mode():
                for batch in (1, 4):
                    images = torch.randn(batch, 3, 224, 224, generator=generator)
                    expected = eager.logits(images)
                    actual = onnx_engine.logits(images)
                    self.assertEqual(actual.shape, (batch, 11))
                    torch.testing.assert_close(actual, expected, atol=.0001, rtol=.0001)
            self.assertEqual(eager.temperature, onnx_engine.temperature)
        finally:
            torch.set_num_threads(previous_threads)

    def test_graph_from_another_checkpoint_is_rejected(self):
        engine = CropDiseaseInferenceEngine(weights_path=Path("models/does_not_exist.pt"), runtime="onnx", device="cpu")
        self.assertFalse(engine.weights_loaded)


if __name__ == "__main__":
    unittest.main()
