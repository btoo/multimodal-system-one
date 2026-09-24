import unittest

import torch

from mmso.artifacts import ROOT, read_manifest
from mmso.joint_data import expand_scenes
from mmso.joint_model import NativeDecisionModel, Vocabulary
from mmso.optimization_study import PrimitiveCache, PrimitiveHeads, selection_key, audit_optimization


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_selection_requires_accuracy_gate_before_confidence_preference(self):
        def metric(a, b, loss):
            return {"by_slice": {"in_distribution": {"joint_macro_accuracy": a, "joint_macro_normalized_nll": loss},
                                 "compositional": {"joint_macro_accuracy": b, "joint_macro_normalized_nll": loss}}}
        self.assertGreater(selection_key(metric(.70, .65, 1.)), selection_key(metric(.52, .52, .7)))
        self.assertGreater(selection_key(metric(.70, .65, .8)), selection_key(metric(.70, .65, 1.)))
        self.assertEqual(selection_key(metric(.64, .8, .7))[0], 0)

    def test_direct_heads_send_gradients_to_both_observable_representations(self):
        heads = PrimitiveHeads(16)
        visual = torch.randn(3, 4, 16, requires_grad=True)
        acoustic = torch.randn(3, 16, requires_grad=True)
        losses = heads.losses(visual, acoustic, torch.arange(12).reshape(3, 4) % 8,
                             torch.arange(12).reshape(3, 4) % 4, torch.tensor([1, 3, 7]))
        sum(losses.values()).backward()
        self.assertGreater(float(visual.grad.norm()), 0)
        self.assertGreater(float(acoustic.grad.norm()), 0)
        self.assertEqual(set(losses), {"visual_word", "visual_color", "audio_word"})

    def test_training_sampler_matches_existing_resampling_and_labels(self):
        scenes = [s for s in read_manifest(ROOT / "evals/manifests/joint_panels_v1.jsonl") if s["split"] == "train"][:8]
        if not all((ROOT / s["audio"]["path"]).exists() for s in scenes):
            self.skipTest("Prepared speech assets required")
        vocabulary = Vocabulary.from_records(expand_scenes(scenes))
        data = PrimitiveCache(scenes, vocabulary, "train")
        indices = torch.arange(12)
        inputs, targets, primitives = data.training_batch(indices, "cpu", torch.Generator().manual_seed(15))
        expected, expected_targets = data.batch(indices, "cpu", pairing_generator=torch.Generator().manual_seed(15))
        for actual, reference in zip(inputs, expected):
            self.assertTrue(torch.equal(actual, reference))
        self.assertTrue(torch.equal(targets, expected_targets))
        self.assertTrue(torch.equal(primitives[0], data.visual_words[data.index[indices]]))
        self.assertTrue(torch.equal(primitives[1], data.visual_colors[data.index[indices]]))

    def test_temporary_heads_are_not_in_inference_state(self):
        model = NativeDecisionModel(43, 32, 1)
        before = set(model.state_dict())
        heads = PrimitiveHeads(32)
        self.assertEqual(set(model.state_dict()), before)
        self.assertNotIn("word.weight", before)
        self.assertGreater(sum(p.numel() for p in heads.parameters()), 0)

    def test_fresh_confirmation_cannot_reuse_old_speaker_or_audio(self):
        manifest = ROOT / "evals/manifests/optimization_panels_v1.jsonl"
        if not manifest.exists(): self.skipTest("New confirmation not yet prepared")
        scenes = read_manifest(manifest)
        scene = next(s for s in scenes if s["split"] == "test")
        scene["speaker"] = scenes[0]["speaker"]
        with self.assertRaises(ValueError): audit_optimization(scenes, verify_media=False)


if __name__ == "__main__":
    unittest.main()
