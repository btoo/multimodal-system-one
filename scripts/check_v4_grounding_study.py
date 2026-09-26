"""Verify recorded training, unchanged holdouts, scores and resource shutdown."""
from pathlib import Path
import json
import math
import subprocess

import numpy as np

from mmso.backbone_study import digest
from mmso.grounding_tasks import parse_point
from mmso.grounding_training import score_result

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/v4-grounding-training-v1'


def read(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    rows=read(ROOT/'data/v4-training/manifest.jsonl');cases={r['id']:r for r in rows}
    joint=read(ROOT/'evals/manifests/v4_joint_regression_v1.jsonl');cases.update({r['id']:r for r in joint})
    acquisition=json.loads((ROOT/'evals/acquisition/v4_grounding_training_v1.json').read_text())
    assert digest(ROOT/'data/v4-training/manifest.jsonl')==acquisition['manifest_sha256']
    cfg=ROOT/'artifacts/v4-grounding-training/minicpmo45/adapter/config.json';config=json.loads(cfg.read_text())
    nomination=json.loads((ROOT/'evals/v4-grounding-confirmation-nomination-v1.json').read_text())
    assert digest(cfg)==nomination['adapter_config_sha256']
    assert config['adapter_sha256']==nomination['adapter_sha256']==digest(cfg.parent/'adapter.safetensors')
    trace=read(REPORT/'attempts/mixed-training-v1/training.jsonl')
    assert [r['step'] for r in trace]==list(range(1,1025))
    assert len({r['id'] for r in trace})==config['unique_training_cases_seen']==1024
    assert all(cases[r['id']]['split']=='train' and r['split']=='train' and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace)
    assert any(r['gradient_norm']>0 for r in trace)
    expected={
        'minicpm-baseline-v2/evaluation':{r['id'] for r in rows if r['split'] in ('development','regression')},
        'qwen-baseline-v2/evaluation':{r['id'] for r in rows if r['split'] in ('development','regression')},
        'adapted-development-v1/evaluation':{r['id'] for r in rows if r['split'] in ('development','regression')},
        'confirmation-v1/base':{r['id'] for r in rows if r['split']=='confirmation'},
        'confirmation-v1/adapted':{r['id'] for r in rows if r['split']=='confirmation'},
        'confirmation-v1/base-joint':{r['id'] for r in joint},
        'confirmation-v1/adapted-joint':{r['id'] for r in joint},
        'qwen-native-coordinates-v1/development':{r['id'] for r in rows if r['task']=='point' and r['split']=='development'},
        'qwen-native-coordinates-v1/confirmation':{r['id'] for r in rows if r['task']=='point' and r['split']=='confirmation'},
        'image-ablation-v1/base-blank':{r['id'] for r in rows if r['dataset']=='screenspot_v2' and r['split']=='confirmation'},
        'image-ablation-v1/adapted-blank':{r['id'] for r in rows if r['dataset']=='screenspot_v2' and r['split']=='confirmation'},
    }
    predictions=0
    for folder,ids in expected.items():
        path=REPORT/'attempts'/folder/'predictions.jsonl';values=read(path)
        assert len(values)==len(ids) and {r['id'] for r in values}==ids,folder
        result=json.loads((path.parent/'evaluation.json').read_text());assert result['predictions_sha256']==digest(path)
        for row in values:
            case=cases[row['id']];assert row['status']=='ok',(folder,row['id'])
            assert score_result(case,row)['correct']==row['correct']
            if case['task']=='point':
                scale=1000 if folder.startswith('qwen-native') else 1
                assert parse_point(row['text'],coordinate_scale=scale)==row['point']
            if case['task']=='choice':
                p=np.asarray(row['probabilities']);assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-5
            t=row['timings'];assert abs(t['request_ms']-t['decode_and_processor_ms']-t['model_and_output_ms'])<.001
        predictions+=len(values)
    costs=json.loads((REPORT/'cost-and-shutdown.json').read_text())
    assert costs['all_apps_stopped'] and costs['gpu_jobs_reserved']<=9 and costs['gpu_reserved_upper_usd']<=45
    assert costs['openai_usage_unknown_calls']==0
    reference=read(REPORT/'attempts/openai-sol-development-v1/predictions.jsonl')
    reference_ids={r['id'] for r in rows if r['split']=='development' and r['dataset'] in {'screenspot_pro','chartqa','boolq'}}
    assert len(reference)==373 and {r['id'] for r in reference}==reference_ids
    for result in reference:
        case=cases[result['id']];assert result['status']=='ok' and result['model']=='gpt-6-sol'
        correct=result['choice']==case['target'] if case['task']=='choice' else score_result(case,result)['correct']
        assert correct==result['correct']
    cumulative=0
    for ledger in (ROOT/'reports/v4-iteration-v2/openai-spend-ledger.jsonl',REPORT/'openai-spend-ledger.jsonl'):
        cumulative+=sum(r['charged_upper_usd'] for r in read(ledger))
    assert cumulative<=2
    summary=json.loads((REPORT/'summary.json').read_text())
    for path,value in summary['source_hashes'].items():assert digest(ROOT/path)==value
    figures=json.loads((REPORT/'figures.json').read_text());assert figures['summary_sha256']==digest(REPORT/'summary.json')
    for fig in figures['figures']:assert digest(ROOT/fig['path'])==fig['sha256']
    public=set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
    assert 'artifacts/v4-grounding-training/minicpmo45/adapter/adapter.safetensors' not in public
    result={'status':'passed','training_updates':1024,'training_cases_seen':1024,'trainable_adapter_parameters':config['trainable_parameters'],
        'confirmation_cases':448,'verified_self_hosted_predictions':predictions,'joint_regression_cases':64,
        'verified_openai_reference_predictions':len(reference),'cumulative_openai_cost_upper_usd':cumulative,
        'all_study_apps_stopped':True,'checkpoint_tensor_in_git':False,'original_scores_preserved':True}
    (REPORT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
