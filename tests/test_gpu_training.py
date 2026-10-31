from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torchvision.models import mobilenet_v2
from backend.config import CLASS_NAMES, MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from ml.train_gpu import make_model, small_cnn, train_mode, train, evaluate_checkpoint


class GpuTrainingTests(unittest.TestCase):
    def test_finetune_only_updates_selected_blocks_and_keeps_batchnorm_frozen(self):
        source = {'state_dict': mobilenet_v2(weights=None, num_classes=11).state_dict(),
                  'architecture': 'mobilenet_v2', 'class_names': CLASS_NAMES,
                  'input_size': list(MODEL_INPUT_SIZE), 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}
        model = make_model('mobilenet_v2', source)
        train_mode(model)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith(('features.17.', 'features.18.', 'classifier.')))
        self.assertTrue(all(not module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)))

    def test_small_cnn_trains_saves_and_reloads_on_cpu(self):
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for split in ('train', 'validation', 'test'):
                    (root / f'{split}.csv').write_text(split)
                class TinyDataset(torch.utils.data.Dataset):
                    def __init__(self, manifest, training=False):
                        self.manifest = manifest
                        self.rows = [{'class_index': i} for i in range(11)]
                    def __len__(self): return 11
                    def __getitem__(self, index):
                        return torch.full((3, 32, 32), index / 11), index
                output = root / 'cnn.pt'
                args = Namespace(epochs=1, batch_size=11, patience=3, lr=.001, output=output,
                                 device='cpu', seed=42, architecture='small_cnn', data=root, weights=None)
                with patch('ml.train_gpu.ManifestDataset', TinyDataset):
                    train(args)
                    result = evaluate_checkpoint(output, root, 'test', 'cpu', 11)
                self.assertEqual(result['count'], 11)
                self.assertEqual(len(result['confusion_matrix']), 11)
                saved = torch.load(output, weights_only=True)
                self.assertTrue(all(value.device.type == 'cpu' for value in saved['state_dict'].values()))
                self.assertFalse(saved['training']['test_used_for_selection'])
                self.assertGreater(saved['temperature'], 0)
        finally:
            torch.set_num_threads(previous)


if __name__ == '__main__':
    unittest.main()
