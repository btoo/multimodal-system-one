import platform
import tempfile
from pathlib import Path
import unittest

from PIL import Image, ImageDraw, ImageFont

from mmso.grounding import VisionOCR, center, clean_proposals, rank_proposals


def proposal(text, box, kind="word", confidence=.8):
    return {"text": text, "bbox_xyxy": box, "kind": kind, "confidence": confidence}


class GroundingTests(unittest.TestCase):
    def test_click_coordinate_preserves_small_target_resolution(self):
        proposals = [proposal("Continue", [101, 155, 151, 171]), proposal("Cancel", [201, 155, 241, 171])]
        for variant in ["token_f1", "idf_coverage"]:
            result = rank_proposals(proposals, "continue to install", variant)
            self.assertEqual(result["point"], [126, 163])
            self.assertFalse(result["abstained"])

    def test_zero_proposals_and_no_lexical_evidence_abstain(self):
        self.assertTrue(rank_proposals([], "click the icon")["zero_proposals"])
        result = rank_proposals([proposal("File", [1, 2, 3, 4])], "launchpad")
        self.assertTrue(result["abstained"])
        self.assertIsNone(result["point"])

    def test_deduplication_keeps_distant_same_text_and_clips_to_image(self):
        result = clean_proposals([proposal("Save", [-1, 0, 50, 20]),
            proposal("Save", [0, 0, 50, 20], "line"), proposal("Save", [100, 0, 150, 20]),
            proposal("!!!", [0, 0, 20, 20]), proposal("bad", [5, 5, 5, 5])], 200, 100)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["bbox_xyxy"][0], 0)

    def test_instruction_changes_ranking_without_changing_proposals(self):
        proposals = [proposal("Save", [1, 2, 3, 4]), proposal("Cancel", [11, 12, 13, 14])]
        self.assertNotEqual(rank_proposals(proposals, "save")["point"], rank_proposals(proposals, "cancel")["point"])
        self.assertEqual(len(proposals), 2)

    def test_ties_use_declared_geometry_order(self):
        proposals = [proposal("File", [10, 0, 20, 10]), proposal("File", [0, 0, 10, 10])]
        result = rank_proposals(proposals, "file")
        self.assertEqual(result["point"], [5, 5])
        self.assertEqual(result["tie_count"], 2)

    @unittest.skipUnless(platform.system() == "Darwin", "Apple Vision requires macOS")
    def test_actual_vision_fixture_and_top_left_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.png"
            image = Image.new("RGB", (800, 400), "white")
            ImageDraw.Draw(image).text((100, 150), "Continue", fill="black",
                font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48))
            image.save(path)
            proposals, details = VisionOCR().proposals(path)
            found = [p for p in proposals if p["text"].lower() == "continue"]
            self.assertTrue(found)
            point = center(found[0]["bbox_xyxy"])
            self.assertTrue(100 <= point[0] <= 300 and 150 <= point[1] <= 200)
            self.assertEqual(details["recognition_level"], "fast")
            self.assertTrue(details["cpu_only_requested"])

    @unittest.skipUnless(platform.system() == "Darwin", "Apple Vision requires macOS")
    def test_tiled_ocr_preserves_full_image_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.png"
            image = Image.new("RGB", (1400, 2200), "white")
            drawing = ImageDraw.Draw(image)
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48)
            drawing.text((100, 150), "Continue", fill="black", font=font)
            drawing.text((1000, 1800), "Cancel", fill="black", font=font)
            image.save(path)
            proposals, details = VisionOCR("fixed_1024").proposals(path)
            found = [p for p in proposals if p["text"].lower() == "cancel"]
            self.assertTrue(found)
            for result in found:
                point = center(result["bbox_xyxy"])
                self.assertTrue(1000 <= point[0] <= 1200 and 1800 <= point[1] <= 1850)
            self.assertEqual(details["tile_count"], 6)


if __name__ == "__main__":unittest.main()
