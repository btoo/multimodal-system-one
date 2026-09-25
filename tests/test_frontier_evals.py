import unittest
import numpy as np
from PIL import Image

from mmso.frontier_evals import mmstar_question, model_input, point_inside, summarize
from mmso.v4_iteration import grid_overlay, REGIONS


class FrontierEvalTests(unittest.TestCase):
    def test_original_options_preserve_commas_and_missing_distractors(self):
        question, choices = mmstar_question('What is visible?\nOptions: A: One, two, or three., B: Nothing., C: nan, D: nan')
        self.assertEqual(question, 'What is visible?')
        self.assertEqual(choices, ['One, two, or three.', 'Nothing.', 'nan', 'nan'])
        self.assertEqual(mmstar_question('Question?\nChoices:\n(A) One\n(B) Two'), ('Question?', ['One','Two']))
        self.assertEqual(mmstar_question('Question?\n(A) One\n(B) Two'), ('Question?', ['One','Two']))

    def test_inference_boundary_excludes_answers_and_boxes(self):
        row = dict(question='Where?', choices=['Left', 'Right'], media=[], target=1, target_bbox_xyxy=[1,2,3,4], transcript='SECRET')
        self.assertEqual(set(model_input(row)), {'question','choices','media'})

    def test_failed_prediction_stays_in_denominator(self):
        rows = [dict(id=str(i), benchmark='public', stratum='one', group_id=str(i), choices=['a','b'], target=0) for i in range(2)]
        preds = [dict(id='0', status='ok', probabilities=[.9,.1], timings={'request_ms':10}), dict(id='1', status='unsupported')]
        result = summarize(rows,preds)['public']
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['selective']['0.8']['coverage'], .5)
        self.assertAlmostEqual(result['ece_successes_only'], .1)
        with self.assertRaises(ValueError): summarize(rows, preds[:1])
        preds[0]['probabilities']=[.8,.8]
        with self.assertRaises(ValueError): summarize(rows,preds)

    def test_coarse_region_is_not_successful_click(self):
        self.assertFalse(point_inside([1/6,1/6], [2,2,8,8], [300,300]))
        self.assertTrue(point_inside([.02,.02], [2,2,8,8], [300,300]))
        self.assertFalse(point_inside([float('nan'),0], [2,2,8,8], [300,300]))

    def test_overlay_is_geometry_only_and_preserves_input(self):
        original=Image.new('RGB',(300,300),'white')
        output=grid_overlay(original,REGIONS)
        self.assertEqual(original.getpixel((100,150)),(255,255,255))
        self.assertNotEqual(output.getpixel((100,150)),(255,255,255))
        reordered=grid_overlay(original,list(reversed(REGIONS)))
        self.assertFalse(np.array_equal(np.asarray(output),np.asarray(reordered)))


if __name__ == '__main__': unittest.main()
