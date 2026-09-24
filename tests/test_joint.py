import unittest
import numpy as np
import torch

from mmso.joint_world import COLORS,POSITIONS,OPPOSITE,WORDS,answer,make_panel,render_panel,held_pair
from mmso.joint_data import expand_scenes,questions_for_scene
from mmso.joint_model import NativeDecisionModel,Vocabulary,encode_requests
from mmso.joint_training import PairedCache,decision_metrics


class OracleTests(unittest.TestCase):
    def panel(self):
        return {"background":245,"tiles":[{"word":w,"color":c,"jitter":[0,0],"stroke":5} for w,c in zip(['left','right','yes','go'],COLORS)]}

    def test_question_changes_answer_with_same_inputs(self):
        p=self.panel()
        self.assertEqual(answer(p,'left','color'),'red')
        self.assertEqual(answer(p,'left','opposite_color'),'green')
        self.assertEqual(answer(p,'left','position'),'top left')
        self.assertEqual(answer(p,'left','opposite_position'),'top right')
        self.assertEqual(answer(p,'left','present'),'yes')
        self.assertEqual(answer(p,'left','absent'),'no')

    def test_absent_and_opposite_semantics(self):
        p=self.panel()
        self.assertEqual(answer(p,'stop','color'),'not present')
        self.assertEqual(answer(p,'stop','opposite_color'),'yellow')

    def test_moving_tiles_changes_location_not_color(self):
        p=self.panel();moved={**p,'tiles':p['tiles'][1:]+p['tiles'][:1]}
        self.assertEqual(answer(moved,'left','color'),'red')
        self.assertEqual(answer(moved,'left','position'),'bottom right')

    def test_train_and_compositional_pairs_are_disjoint(self):
        for seed in range(30):
            for compositional in [False,True]:
                p=make_panel(seed,compositional)
                self.assertTrue(all(held_pair(t['word'],t['color'])==compositional for t in p['tiles']))
                self.assertEqual(render_panel(p).size,(128,128))
                self.assertEqual(render_panel(p).tobytes(),render_panel(make_panel(seed,compositional)).tobytes())


class CandidateModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self):
        self.vocab=Vocabulary(['<pad>','<cls>','what','color','is','spoken','red','green','blue','yes','no'])
        requests=[{'question':'what color is spoken','candidates':[{'id':'a','text':'red'},{'id':'b','text':'green'},{'id':'c','text':'blue'}]}]
        q,c,m=encode_requests(requests,self.vocab)
        torch.manual_seed(53)
        model=NativeDecisionModel(len(self.vocab.tokens),width=32,layers=1).eval()
        args=[torch.randn(1,3,128,128),torch.randn(1,1,64,128),q,c,m]
        return model,args,requests

    def test_variable_candidates_and_permutation(self):
        model,args,_=self.make()
        with torch.no_grad():
            original=model(*args)
            swapped=args.copy();swapped[3]=args[3][:,[2,0,1]];swapped[4]=args[4][:,[2,0,1]]
            actual=model(*swapped)
            self.assertTrue(torch.allclose(original[:,[2,0,1]],actual,atol=1e-5,rtol=0))
            shorter=args.copy();shorter[3]=args[3][:,:2];shorter[4]=args[4][:,:2]
            self.assertTrue(torch.allclose(original[:,:2],model(*shorter),atol=1e-5,rtol=0))

    def test_ids_are_not_neural_inputs(self):
        _,_,requests=self.make()
        first=encode_requests(requests,self.vocab)
        for c in requests[0]['candidates']:c['id']='opaque-'+c['id']
        second=encode_requests(requests,self.vocab)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)))

    def test_unknown_words_and_truncation_are_explicit(self):
        _,_,_=self.make()
        with self.assertRaises(ValueError):self.vocab.encode('unseen concept')
        with self.assertRaises(ValueError):self.vocab.encode('red green',length=1)

    def test_gradients_reach_both_native_modalities_and_text(self):
        model,args,_=self.make();args[0].requires_grad_();args[1].requires_grad_()
        model(*args).sum().backward()
        self.assertGreater(float(args[0].grad.abs().sum()),0)
        self.assertGreater(float(args[1].grad.abs().sum()),0)
        self.assertGreater(float(model.text.embedding.weight.grad.abs().sum()),0)

    def test_question_batch_isolation(self):
        model,args,_=self.make()
        with torch.no_grad():
            one=model(*args)
            batch=[torch.cat([x,x],0) for x in args]
            batch[2][1]=torch.flip(batch[2][1],[0])
            self.assertTrue(torch.allclose(one,model(*batch)[:1],atol=1e-5,rtol=0))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_real_mps_cpu_agreement(self):
        model,args,_=self.make()
        with torch.no_grad():
            expected=model(*args)
            actual=model.to('mps')(*[x.to('mps') for x in args]).cpu()
        self.assertTrue(torch.allclose(expected,actual,atol=3e-4,rtol=0))

    def test_metric_rejects_dropped_rows(self):
        record={'candidates':[{'text':'red'},{'text':'blue'}],'target_index':0,'target_text':'red','id':'a','task':'color','slice':'test'}
        with self.assertRaises(ValueError):decision_metrics([record],[])
        with self.assertRaises(ValueError):decision_metrics([record],[[float('nan'),0.]])
        metrics,_=decision_metrics([record],[[5.,0.]])
        self.assertEqual(metrics['by_slice']['test']['joint_macro_accuracy'],1)


class PairingTests(unittest.TestCase):
    def fixture(self):
        data=PairedCache.__new__(PairedCache)
        data.index=torch.zeros(64,dtype=torch.long)
        data.images=torch.zeros(1,3,2,2)
        data.audio=torch.arange(8).float().reshape(8,1,1,1)
        data.targets=torch.zeros(64,dtype=torch.long)
        data.question=torch.zeros(64,2,dtype=torch.long)
        data.candidates=torch.zeros(64,8,2,dtype=torch.long)
        data.mask=torch.ones(64,8,dtype=torch.bool)
        data.audio_pools={w:[i] for i,w in enumerate(WORDS)}
        data.counterfactual_targets=torch.arange(8).repeat(64,1)
        return data

    def test_repair_changes_audio_and_updates_ground_truth(self):
        data=self.fixture();indices=torch.arange(64)
        x,y=data.batch(indices,torch.device('cpu'),pairing_generator=torch.Generator().manual_seed(27))
        self.assertTrue(torch.equal(x[1].flatten().long(),y))
        self.assertGreater(len(set(y.tolist())),4)
        self.assertTrue(torch.equal(x[0],data.images[data.index]))

    def test_matched_controls_use_identical_pairing_stream(self):
        data=self.fixture();indices=torch.arange(64)
        a,ya=data.batch(indices,torch.device('cpu'),'full',torch.Generator().manual_seed(27))
        b,yb=data.batch(indices,torch.device('cpu'),'image_only',torch.Generator().manual_seed(27))
        self.assertTrue(torch.equal(ya,yb))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[:5],b[:5])))
        self.assertTrue(a[5][:,1].all());self.assertFalse(b[5][:,1].any())


if __name__=='__main__':unittest.main()
