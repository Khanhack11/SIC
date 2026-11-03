import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ml.continue_field import build_cosine_scheduler, restore_progress, save_progress


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

    def test_recovery_preserves_cosine_scheduler_state_and_rejects_mismatch(self):
        torch.manual_seed(19); random.seed(19)
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        optimizer = torch.optim.AdamW([{'params': model[0].parameters(), 'lr': 3e-6},
                                       {'params': model[1].parameters(), 'lr': 2.5e-5}])
        scheduler = build_cosine_scheduler(optimizer, epochs=4, min_ratio=0.2)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        images = torch.ones(2, 3); labels = torch.tensor([0, 1])
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
        scheduler.step()
        saved_lrs = [g['lr'] for g in optimizer.param_groups]
        progress = {'epoch': 1, 'next_sample': 0, 'order': []}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress_cosine.pt'
            save_progress(path, model, optimizer, scaler, progress, {'cosine_lr': True}, scheduler=scheduler)
            optimizer.step(); scheduler.step()
            expected_lrs_after_epoch2 = [g['lr'] for g in optimizer.param_groups]
            restored = restore_progress(path, model, optimizer, scaler, {'cosine_lr': True}, scheduler=scheduler)
            self.assertEqual(restored, progress)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], saved_lrs)
            optimizer.step(); scheduler.step()
            self.assertEqual([g['lr'] for g in optimizer.param_groups], expected_lrs_after_epoch2)
            with self.assertRaisesRegex(ValueError, 'scheduler state mismatch'):
                restore_progress(path, model, optimizer, scaler, {'cosine_lr': True}, scheduler=None)
