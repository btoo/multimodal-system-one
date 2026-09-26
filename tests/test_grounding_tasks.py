import unittest
from types import SimpleNamespace
import torch
from mmso.grounding_tasks import task_prompt, safe_input, parse_point, chart_score, pointer_from_script
from mmso.grounding_training import teacher_forcing_inputs


class GroundingTaskTests(unittest.TestCase):
    def test_teacher_forcing_supervises_only_answer_prediction_positions(self):
        tokenizer=SimpleNamespace(encode=lambda answer,add_special_tokens:[3,4],eos_token_id=5)
        backbone=SimpleNamespace(tokenizer=tokenizer,family='qwen')
        original={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2),'position_ids':torch.tensor([[0,1]])}
        inputs,target=teacher_forcing_inputs(backbone,original,'answer')
        self.assertEqual(inputs['input_ids'].tolist(),[[1,2,3,4]])
        self.assertEqual(target.tolist(),[[3,4,5]])
        self.assertEqual(inputs['position_ids'].tolist(),[[0,1,2,3]])
        self.assertEqual(original['input_ids'].tolist(),[[1,2]])
    def test_point_contract_accepts_full_precision_without_snapping(self):
        self.assertEqual(parse_point('{"x":0.003427,"y":0.914827}'),[.003427,.914827])
        self.assertEqual(parse_point('```json\n{"x":0.2,"y":0.3}\n```'),[.2,.3])
        for bad in ('{"x":true,"y":0.5}','{"x":25,"y":75}','{"x":NaN,"y":0.5}', 'Click (0.2, 0.3)', '{"x":0.2,"y":0.3,"target":"secret"}'):
            self.assertIsNone(parse_point(bad))

    def test_task_prompt_cannot_leak_annotations(self):
        row=dict(task='point',question='Close the window',media=[],target_bbox_xyxy=[9,8,7,6],target_point=[.123,.456],transcript='SECRET')
        self.assertNotIn('SECRET',task_prompt(safe_input(row)))
        self.assertNotIn('0.123',task_prompt(safe_input(row)))
        self.assertEqual(set(safe_input(row)),{'task','question','media'})

    def test_source_code_is_parsed_as_literals_only(self):
        self.assertEqual(pointer_from_script('Task: Open help\nOutput Script:\npyautogui.click(3.5,12.0)'),('Open help',[3.5,12]))
        self.assertEqual(pointer_from_script('Task: Open help\npyautogui.moveTo(3.5,12.0)'),('Open help',[3.5,12]))
        for code in ['__import__("os").system("touch bad")','pyautogui.click(eval("3"),12)', 'pyautogui.click(3,12)\npyautogui.click(4,15)']:
            with self.assertRaises(ValueError):pointer_from_script('Task: Test\nOutput Script:\n'+code)

    def test_chart_score_preserves_original_task(self):
        self.assertTrue(chart_score('104.9',['100']))
        self.assertFalse(chart_score('106',['100']))
        self.assertTrue(chart_score('50%',['0.5']))
        self.assertFalse(chart_score('0.01',['0']))
        self.assertTrue(chart_score(' BLUE ',['blue']))
        self.assertFalse(chart_score('The answer is blue',['blue']))


if __name__=='__main__':unittest.main()
