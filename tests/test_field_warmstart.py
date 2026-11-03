import unittest

import torch
from torch import nn

from ml.continue_field import build_cosine_scheduler, calibrate_balanced, finetuning_groups


class WarmstartTests(unittest.TestCase):
    def test_optimizer_updates_only_selected_blocks_and_head(self):
        torch.manual_seed(7)
        model = nn.Module()
        model.features = nn.Sequential(*(nn.Linear(2, 2) for _ in range(12)))
        model.classifier = nn.Linear(2, 2)
        groups = finetuning_groups(model, 9, 5e-6, 3e-5, 1e-5)
        self.assertEqual([g['lr'] for g in groups], [5e-6, 1e-5, 3e-5])
        parameters = [p for g in groups for p in g['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        old = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(groups)
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        for index, block in enumerate(model.features):
            self.assertEqual(torch.equal(block.weight, old[f'features.{index}.weight']), index < 3)
        self.assertFalse(torch.equal(model.classifier.weight, old['classifier.weight']))
        with self.assertRaises(ValueError): finetuning_groups(model, 13, 1e-5, 1e-4)
        with self.assertRaises(ValueError): finetuning_groups(model, 3, float('nan'), 1e-4)

    def test_cosine_scheduler_preserves_parameter_group_lr_ratios(self):
        model = nn.Module()
        model.features = nn.Sequential(*(nn.Linear(2, 2) for _ in range(12)))
        model.classifier = nn.Linear(2, 2)
        base_lrs = [3e-6, 8e-6, 2.5e-5]
        optimizer = torch.optim.AdamW(finetuning_groups(model, 9, base_lrs[0], base_lrs[2], base_lrs[1]))
        scheduler = build_cosine_scheduler(optimizer, epochs=4, min_ratio=0.2)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], base_lrs)
        prev_lrs = list(base_lrs)
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            current_lrs = [g['lr'] for g in optimizer.param_groups]
            for cur, prev in zip(current_lrs, prev_lrs):
                self.assertLessEqual(cur, prev)
            self.assertAlmostEqual(current_lrs[1] / current_lrs[0], base_lrs[1] / base_lrs[0], places=6)
            self.assertAlmostEqual(current_lrs[2] / current_lrs[0], base_lrs[2] / base_lrs[0], places=6)
            prev_lrs = current_lrs
        for cur, base in zip(prev_lrs, base_lrs):
            self.assertAlmostEqual(cur, base * 0.2, places=10)
        with self.assertRaises(ValueError):
            build_cosine_scheduler(optimizer, epochs=0)
        with self.assertRaises(ValueError):
            build_cosine_scheduler(optimizer, epochs=4, min_ratio=0.0)

    def test_balanced_temperature_is_not_dominated_by_larger_domain(self):
        logits = torch.tensor([[8.,-8.],[8.,-8.],[2.,-2.],[-2.,2.]])
        targets = torch.tensor([0,1,0,1])
        rows = [{'domain':'a'}]*2 + [{'domain':'b'}]*2
        temperature, metrics = calibrate_balanced(logits, targets, rows)
        duplicated_logits = torch.cat([logits[:2].repeat(15,1),logits[2:]])
        duplicated_targets = torch.cat([targets[:2].repeat(15),targets[2:]])
        other, _ = calibrate_balanced(duplicated_logits, duplicated_targets,
                                     [{'domain':'a'}]*30 + [{'domain':'b'}]*2)
        self.assertAlmostEqual(temperature, other, places=3)
        self.assertLessEqual(metrics['validation_nll_after'],metrics['validation_nll_before'])
        self.assertEqual(metrics['method'],'equal_domain_temperature_nll_validation_only')


if __name__ == '__main__':
    unittest.main()
