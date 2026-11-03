import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ml.continue_field import restore_progress, save_progress


class FieldRecoveryTests(unittest.TestCase):
    def test_recovery_preserves_next_dropout_step_optimizer_rng_and_cursor(self):
        torch.manual_seed(81); random.seed(81)
        model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(.4), nn.Linear(5, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        images = torch.ones(4, 3); labels = torch.tensor([0, 1, 0, 1])

        def step():
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()

        step()
        progress = {'epoch': 0, 'next_sample': 4, 'order': [1, 0, 2, 3, 0, 2]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.pt'
            save_progress(path, model, optimizer, scaler, progress, {'fixture': 1})
            step(); expected = {k: v.clone() for k, v in model.state_dict().items()}
            expected_random = random.random()
            restored = restore_progress(path, model, optimizer, scaler, {'fixture': 1})
            self.assertEqual(restored, progress)
            step()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            self.assertEqual(random.random(), expected_random)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                restore_progress(path, model, optimizer, scaler, {'fixture': 2})
