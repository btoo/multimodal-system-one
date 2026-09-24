"""Independent workflow smoke, strict Jev response adapter, and comparable metrics.

These authored fixtures are NOT TypeSafe's published benchmark. MiSO v3's lack
of text-state support is reported as unsupported, never a invented prediction.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np

from .artifacts import ROOT, sha256, write_json, write_manifest

MODEL = 'jev-1.13.0'
PRICE_PER_MILLION_INPUT = .042
MAX_REQUEST_TOKENS = 65536
SUITE = ROOT / 'evals/workflows/smoke-v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return float(value)


def distribution(value, labels):
    if not isinstance(value, dict) or set(value) != set(labels):
        raise ValueError('Probability keys must exactly match the supplied answer space')
    p = np.array([number(value[key]) for key in labels])
    if (p < 0).any() or (p > 1).any() or abs(p.sum() - 1) > 1e-5:
        raise ValueError('Invalid probability distribution; no silent renormalization')
    return dict(zip(labels, p.tolist()))


def canonical_answers(questions, response):
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError('Missing or unexpected question answers')
    canonical = {}
    for name, question in questions.items():
        answer = answers[name]
        if answer.get('type') != question['type']:
            raise ValueError('Answer type differs from the requested primitive')
        if question['type'] == 'noul':
            value = number(answer['noul'])
            if not 0 <= value <= 1:
                raise ValueError('Noul must be in [0,1]')
            canonical[name] = {'probabilities': {'false': 1-value, 'true': value}, 'value': value}
        else:
            labels = list(question['criteria']) if question['type'] == 'choice' else [str(i) for i in range(len(question['criteria']))]
            probabilities = distribution(answer['probabilities'], labels)
            confidence = number(answer['confidence'])
            if not 0 <= confidence <= 1:
                raise ValueError('Invalid provider confidence')
            if question['type'] == 'choice':
                value = answer['choice']
                if value not in probabilities or probabilities[value] < max(probabilities.values()) - 1e-6:
                    raise ValueError('Choice is not a maximum-probability option')
            else:
                value = number(answer['score'])
                expected = sum(int(k)*v for k, v in probabilities.items())
                if abs(value-expected) > 1e-4:
                    raise ValueError('Score disagrees with its probability-weighted levels')
                if set(answer['legend']) != set(labels):
                    raise ValueError('Score legend does not cover the levels')
            canonical[name] = {'probabilities': probabilities, 'value': value,
                               'provider_confidence': confidence}
    return canonical


def smoke_action(answers):
    """A project-authored routing policy, with no external side effects."""
    if answers['intervention']['value'] >= .8:
        return 'urgent_review'
    if answers['urgency']['value'] >= 1.5 or max(answers['route']['probabilities'].values()) < .6:
        return 'review'
    return 'route:' + answers['route']['value']


def load_suite(path=SUITE):
    suite = json.loads(Path(path).read_text())
    if suite['purpose'] != 'engineering_smoke' or suite['official_typesafe_benchmark'] is not False:
        raise ValueError('This runner only implements the independent smoke policy')
    ids = set()
    for case in suite['cases']:
        if case['id'] in ids:
            raise ValueError('Duplicate case ID')
        ids.add(case['id'])
        if case['split'] != 'smoke' or set(case['request']) != {'state', 'questions'}:
            raise ValueError('Invalid smoke case')
        questions = case['request']['questions']
        if {k:v['type'] for k,v in questions.items()} != {'route':'choice', 'intervention':'noul', 'urgency':'score'}:
            raise ValueError('Smoke policy requires its three declared questions')
        if set(case['targets']) != set(questions):
            raise ValueError('Missing reference labels')
        for name, question in questions.items():
            labels = (list(question['criteria']) if question['type']=='choice' else
                      ['false','true'] if question['type']=='noul' else [str(i) for i in range(len(question['criteria']))])
            if case['targets'][name] not in labels:
                raise ValueError('Reference target is outside the answer space')
    if not ids:
        raise ValueError('Empty suite')
    return suite


class JevAdapter:
    name = MODEL

    def __init__(self, max_spend_usd=.1, key_file=None):
        self.key = os.environ.get('TYPESAFE_API_KEY')
        if key_file is not None:
            path = Path(key_file).expanduser()
            if path.stat().st_mode & 0o077:
                raise ValueError('Credential file must be private to its owner (chmod 600)')
            self.key = json.loads(path.read_text()).get('api_key')
        if self.key is not None and not isinstance(self.key, str):
            raise ValueError('Invalid credential format')
        if not self.key:
            raise ValueError('Set TYPESAFE_API_KEY locally; do not put credentials in reports or Git')
        self.budget = number(max_spend_usd)
        if self.budget <= 0:
            raise ValueError('A positive explicit spend bound is required')
        self.spent = 0.
        self.calls = 0

    def predict(self, request):
        # Reserve a whole context window, even though smoke cases are short.
        # No retries: an uncertain/failed charge stops the run instead of
        # assuming zero and silently spending the remainder.
        reserve = MAX_REQUEST_TOKENS * PRICE_PER_MILLION_INPUT / 1e6
        if self.spent + reserve > self.budget:
            raise RuntimeError('Cost reservation would exceed the configured budget')
        payload = {'model': MODEL, **request}
        data = json.dumps(payload, allow_nan=False).encode()
        if len(data) > 64_000:
            raise ValueError('Smoke request exceeds the bounded payload size')
        started = time.perf_counter()
        http_request = Request('https://api.typesafe.ai/v1/systemone', data=data,
                               headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        self.calls += 1
        with urlopen(http_request, timeout=30) as response:
            raw = json.loads(response.read(2_000_001))
        elapsed = time.perf_counter() - started
        if raw.get('model') != MODEL:
            raise ValueError('Provider returned a different model version')
        usage = raw.get('usage', {})
        tokens = usage.get('input_tokens')
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= MAX_REQUEST_TOKENS:
            raise ValueError('Missing/invalid billable input-token count; stop with unknown cost')
        cost = tokens * PRICE_PER_MILLION_INPUT / 1e6
        self.spent += cost
        return {'raw':raw, 'answers':canonical_answers(request['questions'], raw), 'latency_seconds':elapsed,
                'cost_usd':cost, 'cost_basis':'published per-input-token API price', 'usage':usage}


def miso_v3_support(request):
    # Inspect the actual registered model contract, not a second guessed list.
    from .api.runtime import NativeRuntime
    from .api.registry import V3_REGISTRATION
    import torch
    torch.set_num_threads(2)
    runtime = NativeRuntime('cpu', V3_REGISTRATION)
    card = runtime.model_card()
    reasons = ['Text-only state is unsupported; model requires one raw image and one raw audio block',
               'Question text cannot substitute for a free-form state/context input']
    unknown = set()
    from .joint_model import words
    for question in request['questions'].values():
        criteria = question.get('criteria', [])
        descriptions = list(criteria.values()) if isinstance(criteria, dict) else criteria
        for text in [question['instructions'], *descriptions]:
            unknown.update(set(words(text)) - set(runtime.vocabulary.tokens))
    return {'supported':False, 'reasons':reasons, 'unknown_question_words':sorted(unknown),
            'checkpoint_sha256':card['checkpoint_sha256'], 'vocabulary_tokens':len(runtime.vocabulary.tokens)}


def summarize(suite, rows):
    cases = {c['id']:c for c in suite['cases']}
    if len(rows) != len(cases) or {r['id'] for r in rows} != set(cases):
        raise ValueError('Every case must have exactly one result, including failures')
    grouped = defaultdict(list); latency=[]; cost=[]; nll=[]; brier=[]; complete=0
    for row in rows:
        case=cases[row['id']]
        if row['request_sha256'] != digest(case['request']):
            raise ValueError('Result request changed')
        if row['status'] == 'completed':
            answers=canonical_answers(case['request']['questions'],row['raw'])
            action=smoke_action(answers)
            grouped[case['workflow']].append(float(action == case['expected_action']))
            complete += 1; latency.append(number(row['latency_seconds']));cost.append(number(row['cost_usd']))
            if latency[-1] < 0 or cost[-1] < 0:
                raise ValueError('Cost and latency cannot be negative')
            for name, answer in answers.items():
                p=answer['probabilities'];target=case['targets'][name]
                nll.append(-math.log(max(p[target],1e-12)))
                brier.append(sum((v-float(k==target))**2 for k,v in p.items()))
        else:
            if row['status'] not in {'unsupported','error','not_run'}:
                raise ValueError('Unrecognized status')
            grouped[case['workflow']].append(0.)
    return {'suite':suite['id'], 'suite_sha256':digest(suite),'attempted_cases':len(cases),'completed_cases':complete,
            'statuses':dict(Counter(r['status'] for r in rows)), 'coverage':complete/len(cases),
            'workflow_macro_success_including_failures':float(np.mean([np.mean(v) for v in grouped.values()])) if complete else None,
            'question_nll_on_completed':float(np.mean(nll)) if nll else None,
            'question_brier_on_completed':float(np.mean(brier)) if brier else None,
            'p50_workflow_ms':float(np.median(latency)*1000) if latency else None,
            'p95_workflow_ms':float(np.quantile(latency,.95)*1000) if latency else None,
            'known_successful_call_cost_usd':sum(cost),
            'total_cost_usd':sum(cost) if complete==len(cases) else None,
            'cost_per_completed_workflow_usd':sum(cost)/complete if complete==len(cases) and complete else None,
            'pareto_eligible':False, 'reference_kind':'project-authored engineering fixtures',
            'note':'Smoke only. Unsupported cases are coverage gaps, not measured wrong predictions. Errors retain denominator; unknown charges are not zero.'}


def comparable_frontier(points):
    """Require identical benchmark/metric/cost/latency contracts, no missing data."""
    if not points:
        return []
    required=('benchmark_sha256','harness_sha256','reference_sha256','metric','input_track',
              'cost_accounting','latency_boundary','concurrency')
    key=tuple(points[0][k] for k in required)
    for p in points:
        if not p['eligible'] or tuple(p[k] for k in required) != key or p['coverage'] != 1:
            raise ValueError('Incomparable or incomplete benchmark points')
        for name in ('quality','cost_usd','latency_ms'):
            number(p[name])
        if p['cost_usd'] < 0 or p['latency_ms'] < 0:
            raise ValueError('Negative cost or latency')
    return [p['model'] for p in points if not any(
        q['quality'] >= p['quality'] and q['cost_usd'] <= p['cost_usd'] and q['latency_ms'] <= p['latency_ms']
        and (q['quality'] > p['quality'] or q['cost_usd'] < p['cost_usd'] or q['latency_ms'] < p['latency_ms'])
        for q in points)]


def run(provider, run_id, execute=False, max_spend_usd=.1, key_file=None):
    import re
    if re.fullmatch(r'[a-zA-Z0-9_-]+',run_id) is None:
        raise ValueError('Invalid run ID')
    directory=ROOT/'reports'/run_id
    if directory.exists():
        raise ValueError('Reports are immutable; use a new run ID')
    suite=load_suite(); rows=[]
    if provider not in {'miso-v3','jev'}:
        raise ValueError('Unknown provider')
    if provider=='jev' and not execute:
        raise ValueError('Use --execute to make bounded external API requests')
    adapter=JevAdapter(max_spend_usd,key_file) if provider=='jev' else None
    support=miso_v3_support({'questions':{f"{c['id']}-{k}":v for c in suite['cases'] for k,v in c['request']['questions'].items()}}) if provider=='miso-v3' else None
    directory.mkdir(parents=True)
    for case in suite['cases']:
        row={'id':case['id'],'request_sha256':digest(case['request']),'provider':provider}
        if support:
            row.update(status='unsupported',reason=support['reasons'])
        else:
            try:
                response=adapter.predict(case['request'])
                row.update(status='completed',**response)
            except Exception as error:
                # Never echo server bodies, request headers, or credentials.
                row.update(status='error',error_type=type(error).__name__,cost_usd=None)
                rows.append(row)
                for remaining in suite['cases'][len(rows):]:
                    rows.append({'id':remaining['id'],'request_sha256':digest(remaining['request']),
                                 'provider':provider,'status':'not_run','reason':'Stopped after uncertain provider failure or budget exhaustion'})
                break
        rows.append(row)
    summary=summarize(suite,rows)
    summary.update(provider=provider,support=support,source={'runner_sha256':sha256(Path(__file__)),'suite_file_sha256':sha256(SUITE)},
                   maximum_calls=len(suite['cases']), max_spend_usd=max_spend_usd if adapter else 0,
                   api_calls_made=adapter.calls if adapter else 0)
    write_manifest(directory/'cases.jsonl',rows);write_json(directory/'result.json',summary)
    return summary
