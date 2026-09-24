import copy
import unittest

import numpy as np
import torch

from mmso.joint_model import NativeDecisionModel, Vocabulary, encode_requests
from mmso.joint_world import COLORS, make_panel, render_panel
from mmso.scale_study import FactorizedVision, make_model, selection_score


class ScaleStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pixels(self, panel):
        return torch.from_numpy(np.array(render_panel(panel)).copy()).permute(2, 0, 1).float()[None] / 127.5 - 1

    def test_shape_prior_is_identical_under_palette_intervention(self):
        panel = make_panel(53)
        expected = FactorizedVision.ink_map(self.pixels(panel))
        for color in COLORS:
            altered = copy.deepcopy(panel)
            for tile in altered["tiles"]:
                tile["color"] = color
            self.assertTrue(torch.equal(expected, FactorizedVision.ink_map(self.pixels(altered))))

    def test_factorized_encoder_retains_color_information(self):
        panel = make_panel(53)
        altered = copy.deepcopy(panel)
        for tile in panel["tiles"]:
            tile["color"] = "red"
        for tile in altered["tiles"]:
            tile["color"] = "blue"
        encoder = FactorizedVision(32).eval()
        a = encoder(self.pixels(panel)[:, :, :64, :64])
        b = encoder(self.pixels(altered)[:, :, :64, :64])
        self.assertTrue(torch.equal(a[:, :24], b[:, :24]))
        self.assertGreater(float((a[:, 24:] - b[:, 24:]).abs().sum().detach()), 0)

    def test_baseline_factory_preserves_existing_architecture(self):
        torch.manual_seed(1)
        original = NativeDecisionModel(43, 32, 1)
        torch.manual_seed(1)
        actual = make_model({"width": 32, "layers": 1, "vision": "rgb"}, 43)
        self.assertEqual(set(original.state_dict()), set(actual.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[key]))

    def test_factorized_joint_model_backpropagates_and_is_candidate_equivariant(self):
        vocabulary = Vocabulary(["<pad>", "<cls>", "what", "color", "red", "blue"])
        q, c, m = encode_requests([{"question": "what color", "candidates": [{"id": "a", "text": "red"}, {"id": "b", "text": "blue"}]}], vocabulary)
        model = make_model({"width": 32, "layers": 1, "vision": "factorized"}, len(vocabulary.tokens))
        args = [self.pixels(make_panel(17)), torch.randn(1, 1, 64, 128), q, c, m]
        logits = model(*args)
        logits.sum().backward()
        self.assertGreater(float(model.image.shape[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.image.color[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.audio_projection.weight.grad.abs().sum()), 0)
        model.eval()
        with torch.no_grad():
            expected = model(*args)
            swapped = args.copy()
            swapped[3], swapped[4] = c[:, [1, 0]], m[:, [1, 0]]
            self.assertTrue(torch.allclose(expected[:, [1, 0]], model(*swapped), atol=1e-5, rtol=0))

    def test_selection_gives_equal_weight_to_id_and_composition(self):
        metrics = {"by_slice": {"in_distribution": {"joint_macro_normalized_nll": 0.2}, "compositional": {"joint_macro_normalized_nll": 1.4}}}
        self.assertAlmostEqual(selection_score(metrics), 0.8)


if __name__ == "__main__":
    unittest.main()
