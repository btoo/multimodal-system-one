"""Post-hoc formatting sensitivity, kept separate from frozen primary scores."""
from pathlib import Path
import json
import re

from mmso.grounding_tasks import chart_score
from mmso.backbone_study import digest

ROOT=Path(__file__).resolve().parents[1]


def numeric_display(text):
    match=re.fullmatch(r'\s*[$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)(?:\s*%|\s+[A-Za-z][A-Za-z\s%/-]*)?\s*',text)
    return match.group(1).replace(',','') if match else text


def main():
    cases={r['id']:r for r in (json.loads(s) for s in (ROOT/'data/v4-training/manifest.jsonl').read_text().splitlines())}
    report=ROOT/'reports/v4-grounding-training-v1';paths={
        'minicpm-base':'attempts/minicpm-baseline-v2/evaluation/predictions.jsonl',
        'qwen-base':'attempts/qwen-baseline-v2/evaluation/predictions.jsonl',
        'openai-sol-none':'attempts/openai-sol-development-v1/predictions.jsonl'}
    outputs={}
    for key,name in paths.items():
        path=report/name
        if not path.exists():continue
        rows=[r for r in (json.loads(s) for s in path.read_text().splitlines()) if r['dataset']=='chartqa']
        cases_changed=[];hits=0
        for row in rows:
            correct=row.get('status')=='ok' and chart_score(numeric_display(row.get('text','')),cases[row['id']]['answers'])
            hits+=correct
            if correct!=row['correct']:cases_changed.append({'id':row['id'],'raw_prediction':row.get('text'), 'numeric_display':numeric_display(row.get('text','')),'primary_correct':row['correct'],'diagnostic_correct':correct})
        outputs[key]={'cases':len(rows),'strict_correct':sum(r['correct'] for r in rows),'diagnostic_correct':hits,
            'changed_cases':cases_changed,'predictions_sha256':digest(path)}
    result={'status':'diagnostic_only','post_hoc':True,'primary_scores_unchanged':True,
        'method':'Strip a leading currency symbol, grouping commas and a simple trailing unit phrase/percent sign from a single numeric answer; do not rescale the number.',
        'limitations':'Not the official ChartQA metric. Units and scales can change meaning; this heuristic is only a sensitivity check and cannot establish semantic correctness or a model ranking.',
        'metric_reference':'https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py',
        'runs':outputs}
    (report/'chart-format-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ('cases','strict_correct','diagnostic_correct')} for k,v in outputs.items()}))


if __name__=='__main__':main()
