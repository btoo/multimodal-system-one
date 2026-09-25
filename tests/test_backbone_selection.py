import copy
import unittest
from collections import UserDict
import torch

from mmso.backbone_selection import choose_winner, quality_reasons, softmax, summarize
from mmso.backbone_study import move, prompt_for
from mmso.backbone_adapters import DecisionLoRA
from mmso.backbone_diagnostics import candidate_projection, clean_generated_text


class BackboneSelectionTests(unittest.TestCase):
    def test_generation_cleanup_only_strips_wrappers(self):
        raw = '```json\n{"probabilities": [0.1, 0.2], "choice": "B"}\n```<|tts_eos|>'
        clean = clean_generated_text(raw)
        self.assertEqual(clean, '{"probabilities": [0.1, 0.2], "choice": "B"}')
        self.assertIn('0.1, 0.2', clean)  # Invalid total is not silently normalized.

    def test_candidate_projection_preserves_selected_vocabulary_logits(self):
        torch.manual_seed(9)
        head = torch.nn.Linear(6, 31, bias=True)
        x = torch.randn(2, 3, 6)
        ids = [3, 7, 13, 25]
        short = candidate_projection(head, ids)
        torch.testing.assert_close(short(x), head(x)[..., ids])

    def test_adapter_starts_identical_and_merge_preserves_trained_function(self):
        torch.manual_seed(1)
        original = torch.nn.Linear(6, 4)
        weights = original.weight.detach().clone()
        inputs = torch.randn(3, 6)
        expected = original(inputs).detach()
        adapter = DecisionLoRA(original, rank=2, alpha=4)
        torch.testing.assert_close(adapter(inputs), expected)
        optimizer = torch.optim.AdamW([adapter.a, adapter.b], lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (adapter(inputs) - torch.ones(3, 4)).square().mean()
            loss.backward(); optimizer.step()
        self.assertIsNone(original.weight.grad)
        torch.testing.assert_close(original.weight, weights)
        trained = adapter(inputs).detach()
        self.assertFalse(torch.equal(trained, expected))
        merged = adapter.merge()
        torch.testing.assert_close(merged(inputs), trained, rtol=1e-5, atol=1e-6)

    def test_processor_mapping_tensors_move_to_requested_device(self):
        # HF BatchFeature/BatchEncoding are UserDict mappings, not dicts.
        batch = UserDict({"input_ids": torch.tensor([[1, 2]]),
                          "pixels": [torch.zeros(1, 3, 8, 8)]})
        result = move(batch, "meta")
        self.assertEqual(result["input_ids"].device.type, "meta")
        self.assertEqual(result["pixels"][0].device.type, "meta")

    def test_private_labels_and_annotations_are_not_prompt_inputs(self):
        row = {"question": "Which option?", "choices": ["One", "Two"], "target": 1,
               "transcript": "SECRET TRANSCRIPT", "target_bbox_xyxy": [1, 2, 3, 4]}
        prompt = prompt_for(row)
        self.assertNotIn("SECRET", prompt)
        self.assertNotIn("target", prompt)
        changed = {**row, "target": 0, "transcript": "OTHER"}
        self.assertEqual(prompt, prompt_for(changed))

    def test_failures_remain_in_accuracy_denominator(self):
        rows = [{"id": "a", "track": "speech", "target": 0, "choices": ["one", "two"]},
                {"id": "b", "track": "speech", "target": 1, "choices": ["one", "two"]}]
        predictions = [{"id": "a", "status": "ok", "timings": {"request_ms": 20}}]
        summary = summarize(rows, predictions, {"a": [.9, .1]})
        self.assertEqual(summary["macro_accuracy"], .5)
        self.assertEqual(summary["errors"], 1)

    def test_fast_but_wrong_candidate_cannot_win(self):
        protocol = {"selection": {"maximum_accuracy_gap_to_best_eligible": .03}}
        fast = {"key": "fast", "disqualification_reasons": ["speech:accuracy_below_floor"],
                "development": {"macro_accuracy": .6, "macro_nll": .5, "macro_track_p95_ms": 2}}
        good = {"key": "good", "disqualification_reasons": [],
                "development": {"macro_accuracy": .9, "macro_nll": .2, "macro_track_p95_ms": 20}}
        self.assertEqual(choose_winner([fast, good], protocol)["winner"], "good")
        self.assertIsNone(choose_winner([fast], protocol)["winner"])

    def test_accuracy_margin_is_applied_before_speed(self):
        protocol = {"selection": {"maximum_accuracy_gap_to_best_eligible": .03}}
        def candidate(key, accuracy, speed):
            return {"key": key, "disqualification_reasons": [], "development": {
                "macro_accuracy": accuracy, "macro_nll": .2, "macro_track_p95_ms": speed}}
        result = choose_winner([candidate("best", .95, 50), candidate("near", .93, 25), candidate("low", .90, 10)], protocol)
        self.assertEqual(result["winner"], "near")
        self.assertEqual(result["quality_reference"], "best")
        self.assertFalse(result["statistical_noninferiority_established"])

    def test_confident_wrong_probabilities_fail_proper_score_gate(self):
        protocol = {"tracks": {"audio": {"accuracy_floor": .5}}, "selection": {"macro_accuracy_floor": .5}}
        summary = {"errors": 0, "macro_accuracy": .75, "tracks": {"audio": {
            "accuracy": .75, "nll": 2.0, "uniform_nll": .693, "brier": .7, "uniform_brier": .5}}}
        reasons = quality_reasons(summary, protocol)
        self.assertIn("audio:nll_not_better_than_uniform", reasons)
        self.assertIn("audio:brier_not_better_than_uniform", reasons)

    def test_softmax_is_stable(self):
        p = softmax([10000, 9999])
        self.assertAlmostEqual(float(p.sum()), 1)
        self.assertGreater(p[0], p[1])


if __name__ == "__main__":
    unittest.main()
