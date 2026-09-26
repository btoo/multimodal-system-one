"""Freeze the protocol's model choice after both development baselines."""
from pathlib import Path
import json
from mmso.backbone_study import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    result={};paths={'minicpmo45':'minicpm-baseline-v2','qwen3-30ba3b':'qwen-baseline-v2'}
    for key,attempt in paths.items():
        path=ROOT/'reports/v4-grounding-training-v1/attempts'/attempt/'summary.json';raw=json.loads(path.read_text())
        if raw['status']!='completed':raise ValueError('Incomplete baseline')
        metrics=raw['evaluation']['metrics'];count=sum(metrics[k]['cases'] for k in ('omniact','screenspot_pro'))
        correct=sum(metrics[k]['correct'] for k in ('omniact','screenspot_pro'))
        if count!=213:raise ValueError('Mismatched development point population')
        result[key]={'correct':correct,'cases':count,'point_accuracy':correct/count,'summary_sha256':digest(path)}
    gap=result['qwen3-30ba3b']['point_accuracy']-result['minicpmo45']['point_accuracy']
    key='qwen3-30ba3b' if gap>=.10 else 'minicpmo45'
    out={'key':key,'baseline_points':result,'qwen_minus_mini_point_accuracy':gap,
        'rule':'Use smaller MiniCPM unless Qwen improves pooled development point accuracy by at least 10 percentage points. Both formatting failures and misses stay in the denominator.',
        'manifest_sha256':digest(ROOT/'data/v4-training/manifest.jsonl'),
        'protocol_sha256':digest(ROOT/'evals/v4-grounding-training-protocol-v1.json'),'confirmation_opened':False}
    path=ROOT/'evals/v4-grounding-training-nomination-v1.json'
    if path.exists():raise ValueError('Nomination is already frozen')
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
