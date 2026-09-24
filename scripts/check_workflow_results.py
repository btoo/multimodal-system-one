"""Recompute original smoke metrics and verify the exact recorded source/inputs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmso.artifacts import ROOT,read_manifest,sha256
from mmso.workflow_benchmark import MODEL,PRICE_PER_MILLION_INPUT,SUITE,load_suite,summarize

suite=load_suite()
for run_id in ('miso-jev-readiness-v1','jev-engineering-smoke-v1'):
    directory=ROOT/'reports'/run_id
    result=json.loads((directory/'result.json').read_text())
    rows=read_manifest(directory/'cases.jsonl')
    assert result['source']['suite_file_sha256']==sha256(SUITE)
    commit=result['source']['source_commit']
    code=subprocess.check_output(['git','show',f'{commit}:mmso/workflow_benchmark.py'],cwd=ROOT)
    assert hashlib.sha256(code).hexdigest()==result['source']['runner_sha256']
    source_suite=subprocess.check_output(['git','show',f'{commit}:evals/workflows/smoke-v1.json'],cwd=ROOT)
    assert hashlib.sha256(source_suite).hexdigest()==sha256(SUITE)
    recomputed=summarize(suite,rows)
    assert all(result[k]==v for k,v in recomputed.items())
    assert result['pareto_eligible'] is False
    if result['provider']=='jev':
        assert result['api_calls_made']==len(rows)==12
        for row in rows:
            assert row['raw']['model']==MODEL
            assert row['raw']['usage']==row['usage']
            assert row['cost_usd']==row['usage']['input_tokens']*PRICE_PER_MILLION_INPUT/1e6
        assert result['total_cost_usd']<=result['max_spend_usd']
    else:
        assert result['api_calls_made']==0 and result['statuses']=={'unsupported':12}
        assert result['workflow_macro_success_including_failures'] is None
    assert 'apikey_' not in (directory/'cases.jsonl').read_text()
    print(f'{run_id}: all case IDs, source hashes, metrics and cost records verified')
