"""Build auditable task tables and figures without changing recorded scores."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmso.backbone_study import digest

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/v4-grounding-training-v1'


def read(path):return [json.loads(s) for s in path.read_text().splitlines()]


def paired(cases,left,right):
    a={r['id']:r for r in left};b={r['id']:r for r in right};groups=defaultdict(list)
    for row in cases:groups[row['group_id']].append(int(b[row['id']]['correct'])-int(a[row['id']]['correct']))
    totals=np.asarray([(sum(v),len(v)) for v in groups.values()])
    draws=totals[np.random.default_rng(20260925).integers(0,len(totals),(5000,len(totals)))].sum(1)
    return {'cases':int(totals[:,1].sum()),'accuracy_delta':float(totals[:,0].sum()/totals[:,1].sum()),
            'paired_group_bootstrap_95':np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist()}


def main():
    variants={'minicpm-base':'minicpm-baseline-v2','qwen-base':'qwen-baseline-v2','adapted':'adapted-development-v1','openai-sol-none':'openai-sol-development-v1'}
    loaded={};hashes={}
    for key,attempt in variants.items():
        path=REPORT/'attempts'/attempt/'summary.json'
        if not path.exists():continue
        raw=json.loads(path.read_text());loaded[key]=raw.get('evaluation',raw)
        hashes[str(path.relative_to(ROOT))]=digest(path)
    nomination_path=ROOT/'evals/v4-grounding-training-nomination-v1.json'
    nomination=json.loads(nomination_path.read_text()) if nomination_path.exists() else None
    confirmation_path=REPORT/'attempts/confirmation-v1/summary.json'
    confirmation=json.loads(confirmation_path.read_text()) if confirmation_path.exists() else None
    result={'development':loaded,'nomination':nomination,'confirmation':confirmation,'source_hashes':hashes,
            'served_model_unchanged':'mmso-joint-v3','frontier_parity_established':False}
    rows=read(ROOT/'data/v4-training/manifest.jsonl')
    if confirmation:
        path=confirmation_path.parent
        a=read(path/'base/predictions.jsonl');b=read(path/'adapted/predictions.jsonl');differences={}
        for dataset in ('screenspot_v2','omniact','chartqa','boolq'):
            cases=[r for r in rows if r['split']=='confirmation' and r['dataset']==dataset]
            differences[dataset]=paired(cases,a,b)
        result['confirmation_paired_differences']=differences
        hashes[str(confirmation_path.relative_to(ROOT))]=digest(confirmation_path)
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    names={'minicpm-base':'MiniCPM base','qwen-base':'Qwen base','adapted':'MiSO mixed adapter','openai-sol-none':'GPT-6 Sol none'}
    datasets=['screenspot_pro','omniact','chartqa','boolq','mmau','mmstar','mmlu_pro']
    lines=['# Grounding and broader-training results','',
        'Development and regression cases below are exposed; use the separate confirmation table for the fixed-checkpoint follow-up. Local and hosted latency boundaries differ.','',
        '| Task | Cases | '+ ' | '.join(names[k] for k in loaded)+' |','|---|---:|'+ '|'.join('---:' for k in loaded)+'|']
    for dataset in datasets:
        ms=[loaded[k]['metrics'].get(dataset) for k in loaded];count=next((m['cases'] for m in ms if m),0)
        cells=[f"{m['accuracy']:.2%} ({m['correct']}/{m['cases']})" if m else '—' for m in ms]
        lines.append('| '+dataset+' | '+str(count)+' | '+' | '.join(cells)+' |')
    lines += ['', 'ChartQA uses the frozen strict relaxed-numeric/exact-text scorer. The separately reported unit/format diagnostic is post hoc and is not a replacement leaderboard metric.','',
        '## Coordinate contract and latency','', '| Model | Point dataset | In-box / cases | Valid normalized JSON | Median ms | p95 ms |','|---|---|---:|---:|---:|---:|']
    for key,raw in loaded.items():
        for dataset in ('screenspot_pro','omniact'):
            m=raw['metrics'].get(dataset)
            if not m:continue
            t=m['latency_ms'];lines.append(f"| {names[key]} | {dataset} | {m['correct']}/{m['cases']} | {m['schema_valid']}/{m['cases']} | {t['median']:.1f} | {t['p95']:.1f} |")
    lines += ['', 'H100 timings include local media decode, preprocessing, full model generation and CPU output. GPT-6 Sol timings also include network and hosted service work. No matched hosted-serving speed claim is made.','', '## Fresh confirmation','']
    if confirmation:
        lines += ['| Dataset | Cases | Unchanged base | Fixed mixed adapter | Paired delta, 95% group interval |','|---|---:|---:|---:|---|']
        for ds,change in result['confirmation_paired_differences'].items():
            a=confirmation['base']['metrics'][ds];b=confirmation['adapted']['metrics'][ds];lo,hi=change['paired_group_bootstrap_95']
            lines.append(f"| {ds} | {a['cases']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {change['accuracy_delta']*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}] |")
    else:lines.append('Not yet evaluated. Checkpoint hashes must be frozen before these calls.')
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'#f7f7ef','axes.facecolor':'#f7f7ef','savefig.facecolor':'#f7f7ef'})
    figures=[]
    def save(fig,name):
        for suffix in ('png','svg'):
            path=ROOT/'docs/assets'/(name+'.'+suffix);fig.savefig(path,dpi=170,bbox_inches='tight')
            if suffix=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
        path=ROOT/'docs/assets'/(name+'.svg');figures.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path),'model_results':True});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8));keys=list(loaded);x=np.arange(len(keys));correct=[];valid=[]
    for k in keys:
        m=loaded[k]['metrics']['screenspot_pro'];correct.append(m['accuracy']*100);valid.append(m['schema_valid']/m['cases']*100)
    ax.bar(x,valid,color='#d6ddd7',label='Valid coordinate JSON');ax.bar(x,correct,color=['#718b7a','#b17b47','#365748','#397396'][:len(keys)],label='Point inside target')
    for i,v in enumerate(correct):ax.text(i,v+2,f'{v:.1f}%',ha='center')
    ax.set(ylim=(0,115),xticks=x,xticklabels=[names[k] for k in keys],ylabel='Percent of the same 117 screenshots')
    ax.set_title('A valid coordinate is only the first step',fontsize=17);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper left')
    fig.text(.1,-.03,'Original target boxes, unrestricted normalized x/y outputs. Formatting failures stay in the denominator.\nPreviously exposed development screenshots; no official full-benchmark or frontier-parity claim.',fontsize=9);fig.tight_layout();save(fig,'v4-valid-grounding')
    if confirmation:
        fig,ax=plt.subplots(figsize=(10,4.7));ds=list(result['confirmation_paired_differences']);x=np.arange(len(ds));width=.34
        for shift,key,color in [(-width/2,'base','#85998b'),(width/2,'adapted','#365748')]:
            values=[confirmation[key]['metrics'][k]['accuracy']*100 for k in ds];ax.bar(x+shift,values,width,label=key,color=color)
            for pos,val in zip(x+shift,values):ax.text(pos,val+2,f'{val:.1f}',ha='center',fontsize=10)
        ax.set(ylim=(0,110),xticks=x,xticklabels=ds,ylabel='Accuracy (%)');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
        ax.set_title('Broader training: frozen-checkpoint confirmation',fontsize=17)
        fig.text(.1,-.015,'Same untouched confirmation cases for base and adapter; point-in-box, chart relaxed accuracy, and binary decisions.\nPaired group intervals and corpus limitations are reported in the results table.',fontsize=9);fig.tight_layout();save(fig,'v4-grounding-confirmation')
    (REPORT/'figures.json').write_text(json.dumps({'figures':figures,'summary_sha256':digest(REPORT/'summary.json')},indent=2)+'\n')
    print(json.dumps({'variants':list(loaded),'confirmation':confirmation is not None}))


if __name__=='__main__':main()
