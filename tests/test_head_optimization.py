import unittest
import torch
from ml.optimize_head import fold_standardization


class HeadOptimizationTests(unittest.TestCase):
    def test_standardization_folds_into_linear_layer(self):
        generator = torch.Generator().manual_seed(42)
        features = torch.randn(8, 10, generator=generator)
        weight = torch.randn(3, 10, generator=generator)
        bias = torch.randn(3, generator=generator)
        mean = torch.randn(10, generator=generator)
        scale = torch.rand(10, generator=generator) + .01
        actual_weight, actual_bias = fold_standardization(weight, bias, mean, scale)
        torch.testing.assert_close(torch.nn.functional.linear(features, actual_weight, actual_bias),
                                   torch.nn.functional.linear((features - mean) / scale, weight, bias))


if __name__ == "__main__":
    unittest.main()
