"""Check coverage, leakage boundaries, numerical evidence and final shutdown."""
from pathlib import Path
import json
import numpy as np

from mmso.backbone_study import digest
from mmso.frontier_evals import model_input, point_inside

ROOT=Path(__file__).resolve().parents[1]


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    report=ROOT/'reports/v4-iteration-v2'
    manifest=ROOT/'data/v4-frontier/manifest.jsonl'
    cases=rows(manifest);by_id={r['id']:r for r in cases}
    assert len(cases)==len(by_id)==2948
    acquisition=json.loads((ROOT/'evals/acquisition/v4_frontier_v1.json').read_text())
    assert digest(manifest)==acquisition['manifest_sha256']
    index=rows(ROOT/'evals/manifests/v4_frontier_v1.index.jsonl')
    assert {r['id'] for r in index}==set(by_id)
    assert not acquisition['exact_media_overlap_with_miso_training']
    assert not acquisition['audio_over_120s']
    for row in cases:
        assert set(model_input(row))=={'question','choices','media'}
        assert 2<=len(row['choices'])<=10 and 0<=row['target']<len(row['choices'])
        for item in row['media']:assert digest(ROOT/item['path'])==item['sha256']
    total_predictions=0
    for attempt in ('frontier-base-v1','frontier-adapted-v1','frontier-qwen3-base-v1','frontier-gemma4-12b-v1'):
        folder=report/'attempts'/attempt;preds=rows(folder/'predictions.jsonl')
        assert len(preds)==2948 and {r['id'] for r in preds}==set(by_id)
        result=json.loads((folder/'summary.json').read_text());assert result['status']=='completed'
        assert result['manifest_sha256']==digest(manifest)
        assert result['predictions_sha256']==digest(folder/'predictions.jsonl')
        for p in preds:
            assert p['status']=='ok', (attempt,p['id'])
            probabilities=np.asarray(p['probabilities']);assert abs(probabilities.sum()-1)<1e-5
            assert len(probabilities)==len(by_id[p['id']]['choices']) and (probabilities>=0).all()
            t=p['timings'];parts=sum(t[k] for k in ('decode_ms','processor_ms','h2d_ms','forward_ms','score_and_d2h_ms'))
            assert abs(parts-t['request_ms'])<.001
        total_predictions+=len(preds)
    speed=rows(report/'attempts/shared-observation-v1/shared-observation.jsonl')
    assert len(speed)==15
    for record in speed:
        count=record['questions']
        assert record['comparison']['serial_shared_media']['argmax_agreement']==count
        assert record['comparison']['serial_shared_media']['max_probability_delta']==0
        assert record['comparison']['packed_shared_media']['argmax_agreement']==count
        assert record['encoder_invocations']['independent']=={'vpm':count,'apm':count}
        for mode in ('serial_shared_media','packed_shared_media'):
            assert record['encoder_invocations'][mode]=={'vpm':1,'apm':1}
            assert len(record['timings_ms'][mode])==5
    fresh=rows(ROOT/'evals/manifests/v4_fresh_screens_v2.jsonl')
    used=set()
    for name in ('screen_grounding.jsonl','screen_grounding_v2.jsonl','v4_selection_v1.jsonl'):
        used|={m['sha256'] for row in rows(ROOT/'evals/manifests'/name) for m in row['media'] if m['modality']=='image'}
    assert len(fresh)==117 and not used.intersection(r['media'][0]['sha256'] for r in fresh)
    screen=rows(report/'attempts/screen-confirmation-v1/predictions.jsonl');screen_by_id={r['id']:r for r in screen}
    assert set(screen_by_id)=={r['id'] for r in fresh}
    for case in fresh:
        p=screen_by_id[case['id']]
        assert p['click_correct']==point_inside(p['point'],case['target_bbox_xyxy'],case['image_size'])
    for attempt,n in (('jev-mmlu-pro-v2',448),('openai-sol-none-v1',568),('openai-astra-medium-v1',20)):
        predictions=rows(report/'attempts'/attempt/'predictions.jsonl')
        assert len(predictions)==n and all(p['status']=='ok' for p in predictions)
        total_predictions+=len(predictions)
    summary=json.loads((report/'summary.json').read_text())
    for path,fingerprint in summary['source_hashes'].items():assert digest(ROOT/path)==fingerprint
    figures=json.loads((report/'figures.json').read_text())
    assert figures['source_summary_sha256']==digest(report/'summary.json')
    for item in figures['figures']:assert digest(ROOT/item['path'])==item['sha256']
    cost=json.loads((report/'cost-and-shutdown.json').read_text())
    assert cost['all_apps_stopped'] and cost['gpu_jobs']<=8 and cost['gpu_reserved_upper_usd']<=45
    assert cost['openai_cost_upper_usd']<=2 and cost['openai_usage_unknown_calls']==0
    result={'status':'passed','new_public_audit_cases':2948,'new_screen_cases':117,
        'scored_public_reference_predictions':total_predictions,'screen_predictions':len(screen),
        'shared_observation_decisions':sum(r['questions'] for r in speed),
        'probability_equivalent_default_decisions':sum(r['questions'] for r in speed),
        'all_study_apps_stopped':True,'credentials_in_bundle_allowlist':False}
    (report/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
