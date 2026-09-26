"""Oracle checks and split checks for the frozen training/evaluation corpus."""
from collections import Counter,defaultdict
from pathlib import Path
import json

from mmso.backbone_study import digest
from mmso.frontier_evals import point_inside
from mmso.grounding_tasks import safe_input,task_prompt,parse_point,chart_score

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=ROOT/'data/v4-training/manifest.jsonl';rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    audit=json.loads((ROOT/'evals/acquisition/v4_grounding_training_v1.json').read_text())
    assert digest(manifest)==audit['manifest_sha256']
    assert len({r['id'] for r in rows})==len(rows)
    groups=defaultdict(set);media_splits=defaultdict(set);seen_media={};point_count=0;chart_count=0
    for row in rows:
        safe=safe_input(row)
        assert not set(safe).intersection({'target','answers','target_point','target_bbox_xyxy','source_paths'})
        modified={**row,'target_point':[.111111,.222222],'target_bbox_xyxy':[1,2,3,4],'answers':['SECRET_REFERENCE']}
        assert task_prompt(safe)==task_prompt(safe_input(modified))
        if row['split']!='regression':groups[row['group_id']].add(row['split'])
        for item in row['media']:
            if item['path'] not in seen_media:
                assert digest(ROOT/item['path'])==item['sha256'];seen_media[item['path']]=True
            if row['split']!='regression':media_splits[item.get('pixel_sha256',item['sha256'])].add(row['split'])
        if row['task']=='point':
            text=json.dumps(dict(zip(('x','y'),[round(x,6) for x in row['target_point']])))
            predicted=parse_point(text)
            assert point_inside(predicted,row['target_bbox_xyxy'],row['image_size']),row['id']
            point_count+=1
        elif row['task']=='chart':
            assert chart_score(str(row['answers'][0]),row['answers']),row['id'];chart_count+=1
    assert all(len(s)==1 for s in groups.values())
    assert all(len(s)==1 for s in media_splits.values())
    result={'status':'passed','cases':len(rows),'training_cases':sum(r['split']=='train' for r in rows),
        'point_oracle_cases':point_count,'point_oracle_accuracy':1.0,'chart_identity_control_cases':chart_count,
        'group_split_leakage':False,'exact_media_split_leakage':False,'label_blind_prompt_check':True,
        'counts':dict(Counter(r['dataset']+':'+r['split'] for r in rows)),
        'manual_source_label_spotcheck':{'cases':['omniact:2245','omniact:4657','omniact:1381','omniact:855'],
            'notes':'Points align with annotated visual elements. The AMC example has a modal overlay; this corpus measures localization, not whether a live click would execute.'}}
    path=ROOT/'reports/v4-grounding-training-v1/data-validation.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cases','training_cases','point_oracle_cases','point_oracle_accuracy')}))


if __name__=='__main__':main()
