import unittest
import torch
from pathlib import Path
import tempfile
from torchvision.models import mobilenet_v2
from ml.finetune_projection import CachedSamples, tail_logits


class ProjectionTests(unittest.TestCase):
    def test_individual_samples_can_mix_across_class_sorted_shards(self):
        root = Path(__file__).resolve().parents[1] / "artifacts" / "test_temp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            paths = []
            for label in range(3):
                path = Path(temporary) / f"{label}.pt"
                torch.save({"features": torch.full((4, 2, 2, 2), label, dtype=torch.float16),
                            "targets": torch.full((4,), label)}, path)
                paths.append(path)
            dataset = CachedSamples(paths)
            features, labels = dataset.batch([0, 4, 8, 1, 5, 9])
            self.assertEqual(labels.tolist(), [0, 1, 2, 0, 1, 2])
            self.assertEqual(features[:, 0, 0, 0].tolist(), labels.float().tolist())
            order = torch.randperm(len(dataset.references), generator=torch.Generator().manual_seed(42)).tolist()
            self.assertEqual(sorted(order), list(range(12)))
            dataset.load.cache_clear()

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
