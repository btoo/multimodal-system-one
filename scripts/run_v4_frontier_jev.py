"""Bounded text-only Jev reference on the exact frozen MMLU-Pro subset."""
from pathlib import Path
import argparse
import json
import time

from mmso.backbone_study import digest
from mmso.frontier_evals import summarize
from mmso.workflow_benchmark import JevAdapter, canonical_answers, digest as request_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--attempt', default='jev-mmlu-pro-v1')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt: raise ValueError('Invalid attempt name')
    folder = ROOT / 'reports/v4-iteration-v2/attempts' / args.attempt
    if folder.exists(): raise ValueError('Attempt already exists')
    rows = [json.loads(s) for s in (ROOT / 'data/v4-frontier/manifest.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['benchmark'] == 'mmlu_pro']
    acquisition = json.loads((ROOT / 'evals/acquisition/v4_frontier_v1.json').read_text())
    if digest(ROOT / 'data/v4-frontier/manifest.jsonl') != acquisition['manifest_sha256']: raise ValueError('Manifest changed')
    adapter = JevAdapter(max_spend_usd=.09, key_file=args.key_file)
    folder.mkdir(parents=True)
    records = []; started = time.time(); failed = False
    for i, row in enumerate(rows):
        request = {'state': row['question'], 'questions': {'answer': {
            'type': 'choice', 'instructions': 'Choose the best answer to the question in the state.',
            'criteria': {chr(65+j): c for j,c in enumerate(row['choices'])}}}}
        if failed:
            result = {'status': 'not_run_after_error'}
        else:
            try:
                raw = adapter.predict(request, validate=False)
                answer = raw['raw']['answers']['answer']
                labels = list(request['questions']['answer']['criteria'])
                chosen = labels.index(answer['choice'])
                try:
                    parsed = canonical_answers(request['questions'], raw['raw'])
                    probabilities = [parsed['answer']['probabilities'][label] for label in labels]
                    contract_valid, contract_error = True, None
                except (ValueError, KeyError, TypeError) as error:
                    probabilities = None; contract_valid = False; contract_error = type(error).__name__
                result = {'status': 'ok', 'choice': chosen, 'probabilities': probabilities,
                    'probability_contract_valid': contract_valid, 'probability_contract_error': contract_error,
                    'raw_answer': answer, 'timings': {'request_ms': raw['latency_seconds'] * 1000}, 'usage': raw['usage'], 'cost_usd': raw['cost_usd']}
            except Exception as error:
                # Do not print headers/credentials or retry a possibly billable failure.
                result = {'status': 'error', 'error_type': type(error).__name__}; failed = True
        record = {'id': row['id'], 'request_sha256': request_digest(request), **result}
        records.append(record)
        with (folder / 'predictions.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
        if (i+1) % 50 == 0: print(json.dumps({'completed': i+1, 'cases': len(rows), 'known_cost_usd': adapter.spent}), flush=True)
    result = {'status': 'failed' if failed else 'completed', 'phase': 'jev-reference', 'model': adapter.name,
        'manifest_sha256': acquisition['manifest_sha256'], 'metrics': summarize(rows, records),
        'known_cost_usd': adapter.spent, 'cost_unknown_after_error': failed, 'calls': adapter.calls, 'ceiling_usd': .1,
        'seconds': time.time()-started, 'timing_boundary': 'Client HTTP request through parsed response; includes network and provider service; not same-hardware latency',
        'supports_native_multimodal_inputs': False, 'official_typesafe_workflow_benchmark': False}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if failed: raise SystemExit(1)


if __name__ == '__main__': main()
