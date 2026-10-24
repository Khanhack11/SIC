import unittest
import torch
from torchvision.models import mobilenet_v2
from ml.finetune_projection import tail_logits


class ProjectionTests(unittest.TestCase):
    def test_cached_tail_matches_full_model(self):
        torch.manual_seed(42)
        model = mobilenet_v2(weights=None, num_classes=11).eval()
        image = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            expected = model(image)
            actual = tail_logits(model, model.features[:-1](image))
        torch.testing.assert_close(expected, actual)

    def test_only_projection_and_head_receive_gradients(self):
        model = mobilenet_v2(weights=None, num_classes=11).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.features[-1][0].parameters():
            parameter.requires_grad_(True)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
        tail_logits(model, torch.randn(2, 320, 7, 7)).sum().backward()
        for name, parameter in model.named_parameters():
            trainable = name.startswith("features.18.0.") or name.startswith("classifier.")
            self.assertEqual(parameter.grad is not None, trainable, name)
        self.assertFalse(model.features[-1][1].training)


if __name__ == "__main__":
    unittest.main()
