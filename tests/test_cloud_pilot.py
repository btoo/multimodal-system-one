import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mmso.audio import choose_device, synchronize
from mmso.cloud_pilot import DivisibleAveragePool, restore_training, save_training, tensor_tree_difference


class DeviceTests(unittest.TestCase):
    def test_auto_and_explicit_devices(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(choose_device().type, 'cuda')
            self.assertEqual(choose_device('cpu').type, 'cpu')
        with patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(choose_device().type, 'mps')
            with self.assertRaisesRegex(ValueError, 'CUDA requested but unavailable'):
                choose_device('cuda')
        with patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(choose_device().type, 'cpu')
        with self.assertRaises(ValueError):
            choose_device('typo')

    def test_cuda_timing_waits_for_the_selected_device(self):
        device = torch.device('cuda')
        with patch('torch.cuda.synchronize') as sync:
            synchronize(device)
            sync.assert_called_once_with(device)


class ResumeTests(unittest.TestCase):
    def test_fixed_pool_matches_adaptive_forward_and_backward(self):
        for shape, output_size in [((2, 64, 8, 16), (4, 4)), ((2, 64, 8, 8), 1)]:
            x = torch.randn(shape, requires_grad=True)
            reference = torch.nn.AdaptiveAvgPool2d(output_size)(x)
            actual = DivisibleAveragePool(output_size)(x)
            gradient = torch.randn_like(reference)
            first = torch.autograd.grad(reference, x, gradient, retain_graph=True)[0]
            second = torch.autograd.grad(actual, x, gradient)[0]
            self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=0))
            self.assertTrue(torch.equal(first, second))
        with self.assertRaises(ValueError):
            DivisibleAveragePool((4, 4))(torch.ones(1, 1, 7, 8))

    def test_optimizer_sampler_and_rng_restore_reproduce_next_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        sampler = torch.Generator().manual_seed(456)
        def update():
            x = torch.randn(8, 3, generator=sampler)
            y = torch.randint(2, (8,), generator=sampler)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward(); optimizer.step()
            return float(loss.detach())
        first = update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_training(path, model, optimizer, sampler, 1, [first], torch.device('cpu'))
            expected_loss = update()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            expected_sampler = sampler.get_state().clone()
            step, losses = restore_training(path, model, optimizer, sampler, torch.device('cpu'))
            self.assertEqual((step, losses), (1, [first]))
            self.assertEqual(update(), expected_loss)
            self.assertEqual(tensor_tree_difference(expected, model.state_dict()), 0.)
            self.assertTrue(torch.equal(sampler.get_state(), expected_sampler))

    def test_resume_comparison_rejects_corruption(self):
        with self.assertRaises(ValueError):
            tensor_tree_difference({'a': torch.ones(1)}, {'b': torch.ones(1)})
        with self.assertRaises(ValueError):
            tensor_tree_difference(torch.ones(1), torch.tensor([float('nan')]))
        self.assertGreater(tensor_tree_difference(torch.ones(1), torch.zeros(1)), 0)


if __name__ == '__main__':
    unittest.main()
