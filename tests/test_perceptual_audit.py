import unittest
import numpy as np
from PIL import Image
from ml.near_duplicate_audit import HashIndex, perceptual_hashes


class PerceptualAuditTests(unittest.TestCase):
    def test_index_returns_only_hashes_within_radius(self):
        index = HashIndex()
        index.add(0, "zero")
        index.add(1, "one")
        index.add(255, "eight")
        index.add(1, "duplicate")
        self.assertEqual(set(record for _, record in index.neighbors(0, 1)), {"zero", "one", "duplicate"})

    def test_rotation_and_reflection_have_same_canonical_hash(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 255, (96, 96, 3), dtype=np.uint8))
        expected = perceptual_hashes(image)
        for operation in (Image.Transpose.ROTATE_90, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.TRANSVERSE):
            self.assertEqual(perceptual_hashes(image.transpose(operation)), expected)


if __name__ == "__main__":
    unittest.main()
