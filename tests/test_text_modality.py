"""Same raw media and answer spaces, different native text input."""
import base64
from copy import deepcopy
import json
import unittest

from fastapi.testclient import TestClient
import torch

from mmso.api.app import create_app
from mmso.api.schema import DecisionRequest
from mmso.artifacts import ROOT


class NativeTextTests(unittest.TestCase):
    def test_text_blocks_reach_the_model_and_change_only_their_question(self):
        torch.set_num_threads(2)
        questions=json.loads((ROOT/'playground/data/native-questions.json').read_text())
        inputs=[]
        for kind,filename,mime in [('image','panel.png','image/png'),('audio','voice.wav','audio/wav')]:
            inputs.append({'type':kind,'source':{'type':'base64','media_type':mime,
                'data':base64.b64encode((ROOT/'playground/public/examples'/filename).read_bytes()).decode()}})
        for name,question in questions.items():
            inputs.append({'type':'text','id':name,'text':question.pop('question')})
            question['question_ref']=name
        first={'model':'mmso-joint-v3','input':inputs,'questions':questions}
        second=deepcopy(first)
        changed=next(b for b in second['input'] if b['type']=='text' and b['id']=='color')
        changed['text']='what color marks the opposite of the spoken command'
        self.assertEqual(first['input'][:2],second['input'][:2])
        self.assertEqual(first['questions'],second['questions'])
        self.assertNotEqual(DecisionRequest.model_validate(first).neural_requests()[0]['question'],
                            DecisionRequest.model_validate(second).neural_requests()[0]['question'])
        with TestClient(create_app(device='cpu',api_key='')) as client:
            a=client.post('/v1/decisions',json=first)
            b=client.post('/v1/decisions',json=second)
        self.assertEqual(a.status_code,200,a.text);self.assertEqual(b.status_code,200,b.text)
        x,y=a.json()['results'],b.json()['results']
        self.assertNotEqual(x['color']['prediction'],y['color']['prediction'])
        self.assertGreater(max(abs(x['color']['probabilities'][k]-y['color']['probabilities'][k])
                               for k in x['color']['probabilities']),.1)
        for name in ['present','presence_score','opposite_ranked']:
            for option,p in x[name]['probabilities'].items():
                self.assertAlmostEqual(p,y[name]['probabilities'][option],places=6)

if __name__=='__main__':unittest.main()
