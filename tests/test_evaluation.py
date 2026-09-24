import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mmso.artifacts import local_media_path, sha256, validate_manifest
from mmso.audio import AudioCNN, LogMel, fit_temperature
from mmso.data import speaker_split
from mmso.metrics import (aligned_predictions, average_precision, categorical_metrics,
    clustered_accuracy_interval, grounding_metrics, joint_metrics, multilabel_metrics, point_hit)
from mmso.screen import grid_regions, region_center


class ProbabilityTests(unittest.TestCase):
    def test_known_categorical_distributions(self):
        perfect = categorical_metrics([0,1],[[1,0],[0,1]])
        self.assertEqual(perfect['accuracy'],1)
        self.assertEqual(perfect['nll'],0)
        self.assertEqual(perfect['ece'],0)
        self.assertEqual(perfect['brier_sum_over_classes'],0)
        uniform = categorical_metrics([0,1],[[.5,.5],[.5,.5]])
        self.assertAlmostEqual(uniform['nll'],math.log(2))
        self.assertAlmostEqual(uniform['normalized_nll'],1)
        self.assertAlmostEqual(uniform['brier_sum_over_classes'],.5)

    def test_confident_errors_are_penalized(self):
        bad = categorical_metrics([0,1],[[.01,.99],[.99,.01]])
        self.assertGreater(bad['nll'],4)
        self.assertGreater(bad['brier_sum_over_classes'],1.9)
        self.assertEqual(bad['accuracy'],0)

    def test_invalid_predictions_rejected(self):
        for values in [[[.3,.3]],[[float('nan'),.5]],[[-.1,1.1]]]:
            with self.assertRaises(ValueError):categorical_metrics([0],values)
        with self.assertRaises(ValueError):categorical_metrics([.5],[[.5,.5]])
        with self.assertRaises(ValueError):categorical_metrics([3],[[.5,.5]])

    def test_ece_includes_confidence_one(self):
        result=categorical_metrics([0,1],[[1,0],[1,0]])
        self.assertEqual(sum(x['count'] for x in result['reliability']),2)
        self.assertAlmostEqual(result['ece'],.5)

    def test_empty_acceptance_is_undefined_not_zero_risk(self):
        result=categorical_metrics([0,1],[[.5,.5],[.5,.5]])
        self.assertIsNone(result['risk_coverage'][-1]['risk'])
        self.assertEqual(result['risk_coverage'][-1]['accepted'],0)

    def test_ap_ties_grouped_without_label_order_advantage(self):
        self.assertAlmostEqual(average_precision([1,0],[.5,.5]),.5)
        self.assertEqual(average_precision([0,1],[.5,.5]),average_precision([1,0],[.5,.5]))
        self.assertEqual(average_precision([1,0],[.8,.2]),1)
        self.assertIsNone(average_precision([0,0],[.8,.2]))
        with self.assertRaises(ValueError):average_precision([.5,1],[.2,.8])

    def test_multilabel_does_not_require_sum_one(self):
        result=multilabel_metrics([[1,1],[0,0]],[[.9,.8],[.1,.2]])
        self.assertEqual(result['macro_ap'],1)
        self.assertEqual(result['exact_match_at_half'],1)

    def test_joint_accepts_multiple_valid_answers(self):
        a={'action':'click','target':'one'};b={'action':'click','target':'two'}
        result=joint_metrics([[a,b]],[{'answer':b,'confidence':.7}])
        self.assertEqual(result['joint_exact_match'],1)

    def test_joint_abstention_is_not_free_success(self):
        a={'action':'click','target':'one'};reject={'action':'abstain','target':None}
        result=joint_metrics([[a]],[{'answer':reject,'confidence':1}])
        self.assertEqual(result['joint_exact_match'],0)
        self.assertEqual(result['decision_coverage'],0)
        self.assertIsNone(result['risk_coverage'][0]['risk'])

    def test_joint_requires_target_even_when_action_matches(self):
        result=joint_metrics([[{'action':'click','target':'one'}]],
                            [{'answer':{'action':'click','target':'two'},'confidence':.9}])
        self.assertEqual(result['joint_exact_match'],0)

    def test_all_prediction_ids_required(self):
        records=[{'id':'a'},{'id':'b'}]
        self.assertEqual(aligned_predictions(records,[{'id':'b'},{'id':'a'}]),records)
        for wrong in [[{'id':'a'}],[{'id':'a'},{'id':'a'}],[{'id':'a'},{'id':'c'}]]:
            with self.assertRaises(ValueError):aligned_predictions(records,wrong)

    def test_cluster_interval_is_deterministic(self):
        one=clustered_accuracy_interval([1,0,1],['a','a','b'])
        self.assertEqual(one,clustered_accuracy_interval([1,0,1],['a','a','b']))
        self.assertEqual(one['clusters'],2)

    def test_calibration_recovers_known_temperature(self):
        # Four repeated conditions with exact observed positive rates.
        q=np.repeat([.2,.4,.6,.8],100)
        y=np.concatenate([np.r_[np.ones(n,dtype=int),np.zeros(100-n,dtype=int)] for n in [20,40,60,80]])
        logits=np.stack([np.zeros(len(q)),2*np.log(q/(1-q))],axis=1)
        self.assertAlmostEqual(fit_temperature(logits,y),2,places=3)


class ManifestTests(unittest.TestCase):
    def row(self,identifier='a',split='train',group='speaker:one',digest='a'*64):
        return {'id':identifier,'dataset':'example','kind':'categorical','split':split,'group_id':group,
                'labels':['yes','no'],'target':'yes','media':[{'path':'data/example.wav','sha256':digest}]}

    def test_group_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError,'Group leakage'):
            validate_manifest([self.row(),self.row('b','test',digest='b'*64)],verify_media=False)

    def test_content_leakage_rejected_even_with_different_ids(self):
        with self.assertRaisesRegex(ValueError,'Media-content leakage'):
            validate_manifest([self.row(),self.row('b','test','speaker:two')],verify_media=False)

    def test_label_column_order_cannot_change(self):
        other=self.row('b',group='speaker:two',digest='b'*64);other['labels']=['no','yes']
        with self.assertRaisesRegex(ValueError,'label order'):
            validate_manifest([self.row(),other],verify_media=False)

    def test_actual_media_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';p.mkdir();f=p/'example.wav';f.write_bytes(b'original')
            row=self.row(digest=sha256(f))
            self.assertTrue(validate_manifest([row],root=d)['media_verified'])
            f.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):validate_manifest([row],root=d)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):local_media_path('/tmp/root','../private.wav')
        with self.assertRaises(ValueError):local_media_path('/tmp/root','/tmp/private.wav')

    def test_speaker_split_not_recording_dependent(self):
        self.assertEqual(speaker_split('speaker-1'),speaker_split('speaker-1'))
        self.assertIn(speaker_split('speaker-1'),{'train','dev','calibration','test'})


class GroundingTests(unittest.TestCase):
    def test_pixel_coordinates_and_boundaries(self):
        self.assertTrue(point_hit([10,20],[10,20,30,40]))
        self.assertFalse(point_hit([.5,.5],[10,20,30,40]))
        self.assertEqual(grounding_metrics([[10,20,30,40]],[[20,30]])['hit_rate'],1)
        with self.assertRaises(ValueError):point_hit([0,0],[1,2,1,3])

    def test_grid_covers_full_image_without_target_input(self):
        boxes=grid_regions(100,80,5,4)
        self.assertEqual(boxes[0],(0,0,20,20))
        self.assertEqual(boxes[-1],(80,60,100,80))
        self.assertEqual(sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes),8000)
        self.assertEqual(region_center(boxes[0]),[10,10])


class AudioMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_waveform_to_logmel_finite_with_silence(self):
        f=LogMel()(torch.zeros(16000))
        self.assertTrue(torch.isfinite(f).all())
        self.assertEqual(f.shape[0:2],(1,64))
        self.assertEqual(f.shape[-1] % 32,0)

    def test_gradients_checkpoint_and_batch_isolation(self):
        torch.manual_seed(4)
        waveform=torch.randn(16000,requires_grad=True)
        x=LogMel()(waveform).unsqueeze(0)
        model=AudioCNN(3)
        model(x).sum().backward()
        self.assertGreater(float(waveform.grad.abs().sum()),0)
        model.eval()
        one=model(x.detach()).detach()
        batch=model(torch.cat([x.detach(),torch.randn_like(x)])).detach()
        self.assertTrue(torch.allclose(one,batch[:1],atol=1e-5,rtol=0))
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'model.safetensors');save_file(model.state_dict(),p)
            clone=AudioCNN(3);clone.load_state_dict(load_file(p));clone.eval()
            self.assertTrue(torch.equal(one,clone(x.detach()).detach()))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_actual_mps_cpu_agreement(self):
        torch.manual_seed(8)
        for samples in [16000,80000]:
            x=LogMel()(torch.randn(samples)).unsqueeze(0)
            model=AudioCNN(3).eval()
            cpu=model(x).detach()
            actual=model.to('mps')(x.to('mps')).detach().cpu()
            self.assertTrue(torch.allclose(cpu,actual,atol=2e-4,rtol=0))


if __name__=='__main__':unittest.main()
