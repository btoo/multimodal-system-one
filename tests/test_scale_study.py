import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mmso.joint_model import NativeDecisionModel, Vocabulary, encode_requests
from mmso.joint_world import COLORS, WORDS, make_panel, render_panel
from mmso.artifacts import read_manifest
from mmso.joint_data import MANIFEST as OLD_MANIFEST
from mmso.scale_study import FactorizedVision, audit_scale, make_model, selection_score, verify_lineage


class ScaleStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pixels(self, panel):
        return torch.from_numpy(np.array(render_panel(panel)).copy()).permute(2, 0, 1).float()[None] / 127.5 - 1

    def test_shape_prior_is_identical_under_palette_intervention(self):
        for word in WORDS:
            panel = make_panel(53)
            for tile in panel["tiles"]:
                tile["word"] = word
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

    def test_frozen_lineage_rejects_changed_manifest_or_protocol(self):
        with patch("mmso.scale_study.sha256", return_value="frozen"):
            verify_lineage({"manifest_sha256": "frozen", "protocol_sha256": "frozen"})
            for field in ["manifest_sha256", "protocol_sha256"]:
                changed = {"manifest_sha256": "frozen", "protocol_sha256": "frozen", field: "changed"}
                with self.assertRaisesRegex(ValueError, "changed"):
                    verify_lineage(changed)

    def test_audit_rejects_reusing_a_prior_confirmation_speaker(self):
        previous = read_manifest(OLD_MANIFEST)
        old_test = next(s for s in previous if s["split"] == "test")
        with self.assertRaisesRegex(ValueError, "prior manifest"):
            audit_scale([old_test], verify_media=False)

    def test_audit_rejects_held_pairs_in_training(self):
        previous = read_manifest(OLD_MANIFEST)
        training = copy.deepcopy(next(s for s in previous if s["split"] == "train"))
        training["panel"] = make_panel(17, compositional=True)
        with self.assertRaisesRegex(ValueError, "Held composition"):
            audit_scale([training], verify_media=False)


if __name__ == "__main__":
    unittest.main()
