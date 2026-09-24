from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from mmso.workflow_benchmark import (JevAdapter, MODEL, canonical_answers, comparable_frontier,
                                     digest, load_suite, smoke_action, summarize)


def golden(case):
    answers={}
    for name,q in case['request']['questions'].items():
        target=case['targets'][name];kind=q['type']
        if kind=='noul':
            answers[name]={'type':kind,'noul':float(target=='true')}
        else:
            labels=list(q['criteria']) if kind=='choice' else [str(i) for i in range(len(q['criteria']))]
            answer={'type':kind,'probabilities':{k:float(k==target) for k in labels},'confidence':1.}
            if kind=='choice':answer['choice']=target
            else:answer.update(score=float(target),legend={str(i):text for i,text in enumerate(q['criteria'])})
            answers[name]=answer
    return {'model':MODEL,'answers':answers,'usage':{'input_tokens':100,'output_tokens':20}}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.suite=load_suite();self.case=self.suite['cases'][0]

    def test_authored_labels_follow_the_actual_policy(self):
        for case in self.suite['cases']:
            answers=canonical_answers(case['request']['questions'],golden(case))
            self.assertEqual(smoke_action(answers),case['expected_action'])

    def test_invalid_and_incomplete_probabilities_are_not_repaired(self):
        response=golden(self.case)
        response['answers']['route']['probabilities'].pop('account')
        with self.assertRaises(ValueError):canonical_answers(self.case['request']['questions'],response)
        response=golden(self.case);response['answers']['route']['probabilities']['billing']=.2
        with self.assertRaises(ValueError):canonical_answers(self.case['request']['questions'],response)
        response=golden(self.case);response['answers']['intervention']['noul']=float('nan')
        with self.assertRaises(ValueError):canonical_answers(self.case['request']['questions'],response)
        response=golden(self.case);response['answers']['urgency']['score']=2
        with self.assertRaises(ValueError):canonical_answers(self.case['request']['questions'],response)
        response=golden(self.case);response['answers']['extra']={}
        with self.assertRaises(ValueError):canonical_answers(self.case['request']['questions'],response)

    def rows(self):
        return [{'id':c['id'],'request_sha256':digest(c['request']),'status':'completed',
                 'raw':golden(c),'latency_seconds':.1,'cost_usd':.001} for c in self.suite['cases']]

    def test_failures_stay_in_denominator_and_unknown_cost_is_not_zero(self):
        rows=self.rows();rows[0]={'id':self.case['id'],'request_sha256':digest(self.case['request']),'status':'error'}
        report=summarize(self.suite,rows)
        self.assertAlmostEqual(report['workflow_macro_success_including_failures'],11/12)
        self.assertAlmostEqual(report['coverage'],11/12)
        self.assertIsNone(report['total_cost_usd'])
        self.assertIsNone(report['cost_per_completed_workflow_usd'])
        self.assertFalse(report['pareto_eligible'])
        with self.assertRaises(ValueError):summarize(self.suite,rows[:-1])
        rows[1]['request_sha256']='changed'
        with self.assertRaises(ValueError):summarize(self.suite,rows)

    def test_unsupported_model_gets_no_invented_accuracy(self):
        rows=[{'id':c['id'],'request_sha256':digest(c['request']),'status':'unsupported'} for c in self.suite['cases']]
        report=summarize(self.suite,rows)
        self.assertEqual(report['coverage'],0)
        self.assertIsNone(report['workflow_macro_success_including_failures'])
        self.assertIsNone(report['p50_workflow_ms'])

    def test_budget_gate_precedes_request_and_version_is_pinned(self):
        with patch.dict(os.environ,{'TYPESAFE_API_KEY':'unit-test-placeholder'}):
            adapter=JevAdapter(max_spend_usd=.00001)
            with patch('mmso.workflow_benchmark.urlopen') as send:
                with self.assertRaises(RuntimeError):adapter.predict(self.case['request'])
                send.assert_not_called();self.assertEqual(adapter.calls,0)
            adapter=JevAdapter(max_spend_usd=.1)
            with patch('mmso.workflow_benchmark.urlopen') as send:
                send.return_value.__enter__.return_value.read.return_value=json.dumps(golden(self.case)).encode()
                result=adapter.predict(self.case['request'])
                request=send.call_args.args[0]
                self.assertEqual(json.loads(request.data)['model'],MODEL)
                self.assertEqual(request.full_url,'https://api.typesafe.ai/v1/systemone')
                self.assertEqual(adapter.calls,1)
                self.assertAlmostEqual(result['cost_usd'],.0000042)

    def test_explicit_private_key_file_overrides_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'TYPESAFE_API_KEY':'environment-placeholder'}):
            path=Path(directory)/'credential.json'
            path.write_text(json.dumps({'api_key':'file-placeholder'}));path.chmod(0o600)
            adapter=JevAdapter(key_file=path)
            self.assertEqual(adapter.key,'file-placeholder')
            self.assertEqual(adapter.calls,0)
            path.chmod(0o644)
            with self.assertRaises(ValueError):JevAdapter(key_file=path)

    def test_frontier_rejects_mixed_tasks_or_inference_timing_boundaries(self):
        base={'model':'a','eligible':True,'coverage':1,'quality':.8,'cost_usd':.01,'latency_ms':100,
              'benchmark_sha256':'b','harness_sha256':'h','reference_sha256':'r','metric':'workflow_exact',
              'input_track':'text','cost_accounting':'inference_marginal_usd','latency_boundary':'client_wall','concurrency':1}
        dominated={**base,'model':'b','cost_usd':.02,'latency_ms':120}
        self.assertEqual(comparable_frontier([base,dominated]),['a'])
        for field,value in [('benchmark_sha256','different'),('latency_boundary','warm_kernel'),('coverage',.9),('cost_usd',None)]:
            with self.assertRaises((ValueError,TypeError)):
                comparable_frontier([base,{**dominated,field:value}])


if __name__=='__main__':unittest.main()
